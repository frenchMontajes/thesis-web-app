# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Pydantic response models for the API.
# ============================================================

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str                  # "genuine" or "forged"
    confidence: float           # 0.0 - 1.0
    distance: float             # raw Euclidean distance (low = genuine)
    score: float                # inverted distance   (high = genuine)
    reference_filename: str     # uploaded reference signature filename
    test_filename: str          # uploaded test signature filename
    message: str                # result message