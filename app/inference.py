from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .model import CoordinateModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_model: CoordinateModel | None = None


def load_model() -> CoordinateModel:
    global _model
    if _model is None:
        _model = CoordinateModel().to(DEVICE)
        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            _model.load_state_dict(state)
        _model.eval()
    return _model


def predict_bytes(data: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = load_model()(tensor)[0].cpu()
    # Coordinates are normalized to [0, 1]; clamp protects malformed checkpoints.
    xy = output[:2].sigmoid().clamp(0, 1)
    log_sigma = output[2:].clamp(-5, 4)
    sigma_norm = log_sigma.exp()
    x, y = float(xy[0] * width), float(xy[1] * height)
    uncertainty = float(((sigma_norm[0] * width) ** 2 + (sigma_norm[1] * height) ** 2).sqrt())
    confidence = float(torch.exp(-uncertainty / max(width, height) * 10).clamp(0, 1))
    return {
        "x": x,
        "y": y,
        "confidence": confidence,
        "uncertainty_px": uncertainty,
        "width": width,
        "height": height,
        "model_version": "baseline-0.1.0",
        "warning": "Estimate only; not guaranteed and not a BOTB submission.",
    }
