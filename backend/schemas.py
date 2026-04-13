# ============================================================
# Signature Verification — Response Schema
# ============================================================

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label               : str   
    confidence          : float  
    distance            : float  
    score               : float  
    reference_filename  : str
    test_filename       : str
    message             : str