from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str          # "genuine" or "forged"
    confidence: float   # 0.0 to 1.0
    filename: str
    message: str