from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

from inference.schemas import PredictionRequest, PredictionResponse
from inference.logging import log_inference


MODEL_PATH = "models/ensemble_model.pkl"

model = joblib.load(MODEL_PATH)

# 224 features on training
EXPECTED_DIM = model.svm_model.n_features_in_

app = FastAPI(
    title="Confidence-Gated Ensemble API",
    version="1.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):

    if len(req.features) != EXPECTED_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_DIM} features, got {len(req.features)}"
        )

    X = np.array(req.features).reshape(1, -1)

    prediction, svm_margin, assistant_used = model.evaluate_inference(X)

    log_inference(
        svm_margin=svm_margin,
        prediction=prediction,
        assistant_used=assistant_used
    )

    return PredictionResponse(
        prediction=prediction,
        svm_margin=svm_margin,
        assistant_used=assistant_used
    )
