"""
FastAPI service wrapping the XGBoost purchase-intent model.

Threshold note: /predict uses the standard 0.5 cutoff for will_purchase,
but purchase_probability is returned raw specifically so the caller can
apply their own business threshold - in a real setting that cutoff should
depend on the cost of a false positive (wasted retargeting spend) vs a
false negative (missed conversion), not default to 0.5 blindly.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features import add_month_cyclical
from src.serve.schemas import SessionFeatures, PredictionResponse

APP_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = APP_DIR / "model_artifact.joblib"
MANIFEST_PATH = APP_DIR / "model_manifest.json"

app = FastAPI(
    title="Purchase Intent Predictor",
    description="Predicts purchase likelihood from session behavior features.",
    version="1.0.0",
)

_model = None
_manifest = None


@app.on_event("startup")
def load_model():
    global _model, _manifest
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run src/train_final.py first.")
    _model = joblib.load(MODEL_PATH)
    _manifest = json.loads(MANIFEST_PATH.read_text())


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    if _manifest is None:
        raise HTTPException(503, "Model not loaded")
    return _manifest


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SessionFeatures):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    row = pd.DataFrame([features.model_dump()])
    row = add_month_cyclical(row)
    proba = float(_model.predict_proba(row)[:, 1][0])
    return PredictionResponse(
        purchase_probability=round(proba, 4),
        will_purchase=proba >= 0.5,
        model_version=_manifest["model_version"],
    )
