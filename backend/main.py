# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Main FastAPI application entry point.
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from predict import classify_signature
from schemas import PredictionResponse
from config import APP_NAME, DEBUG, ALLOWED_ORIGINS
from predict import classify_signature, preprocess_signature, EER_THRESHOLD, model
import torch
import numpy as np
from PIL import Image
import io


app = FastAPI(title=APP_NAME, debug=DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running!"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Signature Verification - Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f4f8;
            }
            .card {
                background: white;
                padding: 40px 60px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 { color: #2d6a4f; }
            p  { color: #555; }
            .status {
                margin-top: 20px;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            .ok    { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            button {
                margin-top: 20px;
                padding: 10px 24px;
                background: #2d6a4f;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 15px;
            }
            button:hover { background: #1b4332; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Signature Verification</h1>
            <p>Hello World!, Backend is up and running.</p>
            <p style="font-size:13px; color:#888;">
                This is a setup test page only — not the final frontend.
            </p>
            <button onclick="pingBackend()">Ping /health</button>
            <div id="status"></div>
        </div>
        <script>
            async function pingBackend() {
                const el = document.getElementById('status');
                try {
                    const res  = await fetch('/health');
                    const data = await res.json();
                    el.innerHTML = `<div class="status ok">✅ ${data.message}</div>`;
                } catch (e) {
                    el.innerHTML = `<div class="status error">❌ Could not reach backend.</div>`;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    reference: UploadFile = File(...),  # original/genuine signature
    test:      UploadFile = File(...)   # signature to verify
):
    result = await classify_signature(reference, test)
    return result

# Add this to main.py, below the /predict route

@app.post("/predict/debug")
async def debug_predict(
    reference: UploadFile = File(...),
    test:      UploadFile = File(...)
):
    import numpy as np
    from PIL import Image as PILImage
    import io

    ref_bytes  = await reference.read()
    test_bytes = await test.read()

    # ── What your heavy pipeline produces ──
    ref_tensor  = preprocess_signature(ref_bytes)
    test_tensor = preprocess_signature(test_bytes)

    with torch.no_grad():
        distance, emb_a, emb_b = model(ref_tensor, test_tensor)

    distance_val = float(distance.item())

    # ── Raw pixel stats BEFORE preprocessing ──
    ref_raw  = np.array(PILImage.open(io.BytesIO(ref_bytes)).convert("L"))
    test_raw = np.array(PILImage.open(io.BytesIO(test_bytes)).convert("L"))

    # ── What the tensor looks like AFTER preprocessing ──
    ref_np  = ref_tensor.squeeze().cpu().numpy()
    test_np = test_tensor.squeeze().cpu().numpy()

    return {
        "distance"              : round(distance_val, 4),
        "score"                 : round(-distance_val, 4),
        "threshold"             : EER_THRESHOLD,
        "verdict"               : "GENUINE" if -distance_val >= EER_THRESHOLD else "FORGED",
        "emb_a_norm"            : round(float(emb_a.norm().item()), 4),
        "emb_b_norm"            : round(float(emb_b.norm().item()), 4),

        "raw_ref_pixel_mean"    : round(float(ref_raw.mean()), 1),
        "raw_test_pixel_mean"   : round(float(test_raw.mean()), 1),

        "tensor_ref_mean"       : round(float(ref_np.mean()), 4),
        "tensor_ref_min"        : round(float(ref_np.min()), 4),
        "tensor_ref_max"        : round(float(ref_np.max()), 4),
        "tensor_ref_shape"      : list(ref_tensor.shape),

        "tensor_test_mean"      : round(float(test_np.mean()), 4),
        "tensor_test_min"       : round(float(test_np.min()), 4),
        "tensor_test_max"       : round(float(test_np.max()), 4),
    }
