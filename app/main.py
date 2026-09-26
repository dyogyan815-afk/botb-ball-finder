from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict_bytes

app = FastAPI(title="Human Coordinate Emulator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-research-only"}

@app.get("/model")
def model_info() -> dict[str, str]:
    return {"model_version": "baseline-0.1.0", "coordinate_system": "original-image-pixels"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload JPEG, PNG, or WebP")
    data = await image.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 15 MB")
    try:
        result = predict_bytes(data)
        result["sha256"] = hashlib.sha256(data).hexdigest()
        return result
    except Exception as exc:
        raise HTTPException(400, f"Unable to process image: {exc}") from exc
