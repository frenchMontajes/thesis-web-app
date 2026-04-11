# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Not final — model path, transforms, and class mapping
# should be updated once training is complete.
# ============================================================

import torch
from torchvision import transforms
from PIL import Image
import io

# TODO: Update this path to your final trained model file
model = torch.load("model/signature_model.pt", map_location="cpu")
model.eval()

# TODO: Ensure these transforms match exactly what was used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

async def classify_signature(file):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)

    # Label mapping: 1 = genuine, 0 = forged
    # TODO: Confirm this matches your training label encoding before finalizing
    label = "genuine" if pred.item() == 1 else "forged"

    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "filename": file.filename,
        "message": f"Signature classified as {label}."
    }