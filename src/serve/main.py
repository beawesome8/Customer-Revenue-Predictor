"""
FastAPI service wrapping the XGBoost purchase-intent model.

Threshold note: /predict uses the standard 0.5 cutoff for will_purchase,
but purchase_probability is returned raw specifically so the caller can
apply their own business threshold - in a real setting that cutoff should
depend on the cost of a false positive (wasted retargeting spend) vs a
false negative (missed conversion), not default to 0.5 blindly.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features import add_month_cyclical
from src.serve.schemas import SessionFeatures, PredictionResponse

APP_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = APP_DIR / "model_artifact.joblib"
MANIFEST_PATH = APP_DIR / "model_manifest.json"

_state = {"model": None, "manifest": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once, held for the process lifetime, instead
    # of on every request - reloading a 450KB joblib file per call would
    # be needlessly slow under real traffic.
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run src/train_final.py first.")
    _state["model"] = joblib.load(MODEL_PATH)
    _state["manifest"] = json.loads(MANIFEST_PATH.read_text())
    yield
    # Shutdown: nothing to clean up - no open connections or file handles
    # held beyond the loaded model object, which garbage-collects normally.
    _state["model"] = None
    _state["manifest"] = None


app = FastAPI(
    title="Purchase Intent Predictor",
    description="Predicts purchase likelihood from session behavior features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/model-info")
def model_info():
    if _state["manifest"] is None:
        raise HTTPException(503, "Model not loaded")
    return _state["manifest"]


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SessionFeatures):
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded")
    row = pd.DataFrame([features.model_dump()])
    row = add_month_cyclical(row)
    proba = float(_state["model"].predict_proba(row)[:, 1][0])
    return PredictionResponse(
        purchase_probability=round(proba, 4),
        will_purchase=proba >= 0.5,
        model_version=_state["manifest"]["model_version"],
    )
