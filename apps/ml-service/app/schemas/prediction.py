from pydantic import BaseModel
from typing import List

class PredictionItem(BaseModel):
    motif: str
    confidence: float

class PredictionResponse(BaseModel):
    success: bool
    predicted_motif: str
    confidence: float
    top_predictions: List[PredictionItem]
    processing_time_ms: int
