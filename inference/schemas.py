from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    svm_margin: float
    assistant_used: bool
