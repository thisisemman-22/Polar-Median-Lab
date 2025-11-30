"""Flask application serving the Polar Median Lab web UI and API."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from .filters import brute_force_median, optimized_median_filter
from .metrics import psnr
from .noise import add_salt_pepper_noise

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path="/static",
)


@app.get("/")
def serve_index():
    if not WEB_DIR.exists():
        return ("Web directory missing.", 404)
    return send_from_directory(str(WEB_DIR), "index.html")


@app.post("/api/denoise")
def denoise_endpoint():
    file = request.files.get("image")
    if not file:
        return ("Image file is required", 400)
    kernel = _ensure_valid_kernel(int(request.form.get("kernel", 5)))
    strategy = request.form.get("strategy", "optimized").lower()
    noise = float(request.form.get("noise", 0.0) or 0.0)
    add_noise_flag = (request.form.get("add_noise", "false").lower() in {"true", "1", "yes", "on"})

    base_image = _load_image(file.read())
    noisy = add_salt_pepper_noise(base_image, amount=noise) if add_noise_flag and noise > 0 else base_image

    start = time.perf_counter()
    if strategy == "optimized":
        denoised = optimized_median_filter(noisy, kernel)
    elif strategy == "brute":
        denoised = brute_force_median(noisy, kernel)
    else:
        return ("Unknown strategy", 400)
    runtime_ms = (time.perf_counter() - start) * 1000

    payload = {
        "strategy": strategy,
        "kernel": kernel,
        "noise_amount": noise if add_noise_flag else 0.0,
        "runtime_ms": runtime_ms,
        "psnr": psnr(base_image, denoised),
        "original": _encode_image(base_image),
        "noisy": _encode_image(noisy),
        "processed": _encode_image(denoised),
    }
    return jsonify(payload)


def _load_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.array(img.convert("RGB"))


def _encode_image(array: np.ndarray) -> str:
    clamped = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(clamped).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _ensure_valid_kernel(kernel: int) -> int:
    if kernel < 3:
        kernel = 3
    if kernel % 2 == 0:
        kernel += 1
    return min(kernel, 31)


__all__ = ["app"]
