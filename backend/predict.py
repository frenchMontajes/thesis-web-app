# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Inference logic for Siamese Network signature verification.
# Label mapping: 1 = Genuine, 0 = Forged
# Uses Euclidean distance + EER threshold for classification.
# TODO: Update EER_THRESHOLD once finalized from training metrics.
# ============================================================

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

from model import DeepCNN, SiameseNetwork
from config import MODEL_PATH

# ─────────────────────────────
# Load model once at startup
# ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = DeepCNN().to(device)
model = SiameseNetwork(backbone).to(device)

# Load saved weights
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ─────────────────────────────
# EER Threshold
# TODO: Replace this value with your actual EER threshold from training
# THIS IS MOCK ONLY 
# ─────────────────────────────
EER_THRESHOLD = -0.5  # placeholder pdate after checking your metrics notebook 

# ─────────────────────────────
# Preprocessing
# Must match training preprocessing exactly
# ─────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),          # 1 channel — matches Conv2d(1, ...)
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def preprocess(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


# ─────────────────────────────
# TODO: Remove mock and switch to real inference below
#       once retraining is complete
# ─────────────────────────────
async def classify_signature(reference_file, test_file):
    # MOCK RESPONSE — remove this block when model is ready
    return {
        "label": "genuine",
        "confidence": 0.8342,
        "distance": 0.3785,
        "score": -0.3785,
        "reference_filename": reference_file.filename,
        "test_filename": test_file.filename,
        "message": "MOCK RESPONSE — model not yet connected."
    }


# ─────────────────────────────
# REAL INFERENCE — uncomment this when retraining is done
# ─────────────────────────────
# async def classify_signature(reference_file, test_file):
#     ref_bytes  = await reference_file.read()
#     test_bytes = await test_file.read()
#
#     ref_tensor  = preprocess(ref_bytes)
#     test_tensor = preprocess(test_bytes)
#
#     with torch.no_grad():
#         distance, _, _ = model(ref_tensor, test_tensor)
#
#     distance_val = distance.item()
#     score = -distance_val
#
#     label = "genuine" if score >= EER_THRESHOLD else "forged"
#     confidence = round(float(torch.sigmoid(torch.tensor(score)).item()), 4)
#
#     return {
#         "label": label,
#         "confidence": confidence,
#         "distance": round(distance_val, 4),
#         "score": round(score, 4),
#         "reference_filename": reference_file.filename,
#         "test_filename": test_file.filename,
#         "message": f"Signature classified as {label}."
#     }