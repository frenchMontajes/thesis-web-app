# ============================================================
# Signature Verification — Inference
# Model    : siamese_auc0.9314.pth
# AUC      : 0.9314 (Val) / 0.9487 (Test)
# EER      : 14.08% (Val) / 12.10% (Test)
# Threshold: -0.5388
# Accuracy : 85.92% (Val) / 88.23% (Test)
# FAR      : 14.07% (Val) / 13.35% (Test)
# FRR      : 14.09% (Val) / 10.19% (Test)
# Label mapping: 1 = Genuine, 0 = Forged
# ============================================================

import cv2
import numpy as np
import io
import torch

from PIL import Image
from torchvision import transforms
from model import DeepCNN, SiameseNetwork
from config import MODEL_PATH

# ─────────────────────────────
# Device
# ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────
# Load model once at startup
# ─────────────────────────────
backbone = DeepCNN().to(device)
model    = SiameseNetwork(backbone).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# ─────────────────────────────
# EER Threshold
# ─────────────────────────────
# EER_THRESHOLD = -0.5388
EER_THRESHOLD = -0.2284

# ─────────────────────────────
# Base transform — matches Dataset.base_transform exactly
# No augmentation, same as val/test during training
# ─────────────────────────────
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ─────────────────────────────
# OpenCV preprocessing pipeline
# Must match training preprocess_signature() exactly
# Returns uint8 (155, 220) — same dtype/shape as HDF5 stored images
# ─────────────────────────────
def _opencv_preprocess(img_bytes: bytes,
                       target_size=(155, 220),
                       use_clahe=True) -> np.ndarray:

    # ── Step 1: Decode bytes → BGR image ──
    img_array = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot read image from uploaded file.")

    # ── Step 2: Grayscale ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # ── Step 3: Inversion ──
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # ── Step 4: Noise removal ──
    gray = cv2.medianBlur(gray, 3)

    # ── Step 5: Background normalization ──
    background = cv2.GaussianBlur(gray, (31, 31), 0)
    gray_norm  = cv2.divide(gray, background, scale=255)
    gray_norm  = np.clip(gray_norm, 0, 255).astype(np.uint8)

    # ── Step 6: Contrast enhancement ──
    if use_clahe:
        clahe     = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        gray_norm = clahe.apply(gray_norm)

    # ── Step 7: Adaptive thresholding ──
    block_size = 15 if gray_norm.shape[0] < 300 else 25
    C          = 10 if gray_norm.shape[0] < 300 else 12

    binary = cv2.adaptiveThreshold(
        gray_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # ── Step 8: Small noise removal ──
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ── Step 8b: Blank check ──
    if cv2.countNonZero(binary) == 0:
        raise ValueError("No signature detected in the uploaded image.")

    # ── Step 9: Tight ROI crop ──
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin     = max(5, int(0.02 * max(w, h)))
        x, y       = max(x - margin, 0), max(y - margin, 0)
        w          = min(w + 2 * margin, binary.shape[1] - x)
        h          = min(h + 2 * margin, binary.shape[0] - y)
        binary     = binary[y:y+h, x:x+w]

    # ── Step 10: Resize with aspect ratio + center pad ──
    h, w    = binary.shape
    scale   = min(target_size[0] / h, target_size[1] / w)
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded       = np.zeros(target_size, dtype=np.uint8)
    top          = (target_size[0] - new_h) // 2
    left         = (target_size[1] - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized

    # Return uint8 [0, 255] — do NOT divide by 255 here
    # ToTensor() in base_transform handles that division
    return padded


def preprocess_signature(img_bytes: bytes) -> torch.Tensor:
    """
    Full inference pipeline matching training exactly:
      _opencv_preprocess → uint8 (155,220)
      Image.fromarray   → PIL 'L'  (same as Dataset.__getitem__)
      base_transform    → ToTensor → [0,1] → Normalize → [-1,1]
      unsqueeze(0)      → [1, 1, 155, 220]
    """
    arr    = _opencv_preprocess(img_bytes)        # uint8 (155, 220)
    pil    = Image.fromarray(arr, mode='L')       # PIL grayscale
    tensor = base_transform(pil)                  # [1, 155, 220]
    return tensor.unsqueeze(0).to(device)         # [1, 1, 155, 220]



# ─────────────────────────────
# SANITIY CHECK
# ─────────────────────────────

#  Genuine pair (same person, different samples)
# distance  : 0.15 – 0.45
# score     : -0.15 to -0.45
# verdict   : GENUINE  ✓

# # Forged pair (different person imitating)
# distance  : 0.60 – 1.20
# score     : -0.60 to -1.20
# verdict   : FORGED   ✓

# # Same image vs itself (sanity check)
# distance  : ~0.0000
# score     : ~0.0000
# verdict   : GENUINE  ✓


# ─────────────────────────────
# Confidence calibration
# Anchored at EER threshold so boundary → 0.5 confidence
# ─────────────────────────────
_SCALE = abs(EER_THRESHOLD)   # 0.5388


def _compute_confidence(distance: float, label: str) -> float:
    p_genuine = float(torch.exp(torch.tensor(-distance / _SCALE)).item())
    p_genuine = max(0.0, min(1.0, p_genuine))
    return round(p_genuine if label == "genuine" else 1.0 - p_genuine, 4)


# ─────────────────────────────
# Main inference function
# ─────────────────────────────
async def classify_signature(reference_file, test_file) -> dict:
    ref_bytes  = await reference_file.read()
    test_bytes = await test_file.read()

    ref_tensor  = preprocess_signature(ref_bytes)
    test_tensor = preprocess_signature(test_bytes)

    with torch.no_grad():
        distance, _, _ = model(ref_tensor, test_tensor)

    distance_val = float(distance.item())
    score        = -distance_val

    label      = "Genuine" if score >= EER_THRESHOLD else "Forged"
    confidence = _compute_confidence(distance_val, label)

    return {
        "label"              : label,
        "confidence"         : confidence,
        "distance"           : round(distance_val, 4),
        "score"              : round(score, 4),
        "threshold"          : EER_THRESHOLD,
        "reference_filename" : reference_file.filename,
        "test_filename"      : test_file.filename,
        "message"            : f"Signature classified as {label}.",
    }