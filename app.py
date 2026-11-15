"""Banana ripeness grading web app (Flask + ONNX Runtime).

Run:  python app.py        -> http://localhost:5000
Model: models/banana_model.onnx  (make it with convert_model.py)
If the model file is missing the app runs in DEMO mode (colour heuristic) and says so in the UI.
"""
import io, json, os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
SIZE = CFG["image_size"]
MODEL_PATH = os.environ.get("BANANA_MODEL", os.path.join(BASE, "models", "banana_model.onnx"))
MAX_FILES, MAX_MB = 100, 15

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

session = None
if os.path.exists(MODEL_PATH):
    import onnxruntime as ort
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    INPUT_NAME = session.get_inputs()[0].name
DEMO = session is None


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB").resize((SIZE, SIZE))
    return np.asarray(img, dtype=np.float32)  # 0-255; preprocess_input is inside the model


def demo_probs(arr: np.ndarray) -> np.ndarray:
    """Rough colour heuristic so the UI can be shown without the model. NOT real accuracy."""
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    score = np.array([
        max(g - r, 0) + 20,                          # green -> unripe
        40 - abs((r - g) - 10),                      # yellow-green -> semi
        40 - abs((r - g) - 40),                      # yellow -> ripe
        max(60 - (r + g + b) / 3, 0) + max(r - g - 60, 0),  # dark/brown -> overripe
    ], dtype=np.float32)
    e = np.exp((score - score.max()) / 12)
    return e / e.sum()


def predict(arr: np.ndarray) -> np.ndarray:
    if DEMO:
        return demo_probs(arr)
    return session.run(None, {INPUT_NAME: arr[None]})[0][0]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/config")
def config():
    return jsonify({**CFG, "demo": DEMO})


@app.post("/api/predict")
def api_predict():
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="no_files"), 400
    if len(files) > MAX_FILES:
        return jsonify(error="too_many", max=MAX_FILES), 400
    out = []
    for f in files:
        item = {"name": f.filename}
        try:
            data = f.read()
            if len(data) > MAX_MB * 1024 * 1024:
                raise ValueError("file_too_large")
            probs = [float(p) for p in predict(load_image(data))]
            best = int(np.argmax(probs))
            item.update(index=best, confidence=probs[best], probs=probs,
                        low_confidence=probs[best] < CFG["low_confidence_threshold"])
        except Exception as e:  # unreadable image etc.
            item["error"] = "bad_image" if not isinstance(e, ValueError) else str(e)
        out.append(item)
    return jsonify(results=out, demo=DEMO)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
