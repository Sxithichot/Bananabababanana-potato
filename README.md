# Bananabababanana-potato — Banana Ripeness Grader

Flask web app that grades banana photos into 4 ripeness classes (MobileNetV2, ONNX Runtime),
summarises a lot and checks it against a customer spec. Thai / English UI.

## Run locally
```
pip install -r requirements.txt
python app.py        # http://localhost:5000
```
On Windows, `gunicorn` may fail to install; use `pip install flask onnxruntime pillow numpy` instead.

## Model
`models/banana_model.onnx` is converted from `best_optimized_banana_model.keras`:
```
pip install tensorflow tf2onnx
python convert_model.py best_optimized_banana_model.keras
```
Without the .onnx file the app runs in DEMO mode (colour heuristic, banner shown).

## Classes
Edit names/colours/threshold in `config.json`. Order must match the training folders (Class A, B, C, D).
In the Kaggle dataset `luischuquimarca/banana-ripeness` these are ripening-day groups, not trade grades
(A: days 1-6, B: 7-14, C: 15-22, D: 23-28). Validate them against real line photos before commercial use.

## Deploy
`docker build -t banana . && docker run -p 8000:8000 banana` (works on Render and similar hosts).
