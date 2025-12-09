"""
Simple FastAPI server to serve predictions using the saved pipeline.
POST /predict with JSON: {"instances": [[...], [...]]}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

ARTIFACT_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pkl")

app = FastAPI(title="Heart Disease Model API")

class PredictRequest(BaseModel):
    instances: list

def load_model(path=ARTIFACT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    data = joblib.load(path)
    return data["pipeline"]

try:
    MODEL = load_model()
except Exception:
    MODEL = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(req.instances)
    try:
        preds = MODEL.predict(X).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predictions": preds}
