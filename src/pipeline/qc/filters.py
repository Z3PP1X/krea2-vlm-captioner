"""Quality Control filters: sRGB normalization, EXIF stripping, Laplacian sharpness, and watermark detection."""

from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Dict, Any


def strip_exif_and_to_srgb(pil_img: Image.Image) -> Image.Image:
    """Normalizes image to RGB color space, applies EXIF orientation if needed, and strips all EXIF metadata."""
    # 1. Transpose according to EXIF orientation so image isn't rotated
    img = ImageOps.exif_transpose(pil_img)

    # 2. Convert to RGB (sRGB standard)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 3. Stripping EXIF: Create a fresh new image copy without info/exif dicts
    clean_img = Image.new("RGB", img.size)
    clean_img.paste(img)

    return clean_img


def compute_laplacian_variance(pil_img: Image.Image) -> float:
    """Computes Laplacian variance as an indicator of focus sharpness.
    
    Higher values correspond to sharp focus and edges; low values (< 100.0) indicate blur.
    Implemented using standard NumPy convolution without requiring OpenCV.
    """
    # Convert to grayscale
    gray = pil_img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0

    # Standard discrete Laplacian kernel:
    # [ 0,  1,  0 ]
    # [ 1, -4,  1 ]
    # [ 0,  1,  0 ]
    laplacian = (
        arr[:-2, 1:-1]   # Top
        + arr[2:, 1:-1]   # Bottom
        + arr[1:-1, :-2]  # Left
        + arr[1:-1, 2:]   # Right
        - 4.0 * arr[1:-1, 1:-1] # Center
    )

    variance = float(np.var(laplacian))
    return round(variance, 2)


def detect_watermarks_and_text(pil_img: Image.Image) -> Tuple[bool, float, str]:
    """Inspects corner regions and borders for high-frequency text overlays or static logo watermarks.
    
    Returns:
        (detected: bool, confidence: float, details: str)
    """
    w, h = pil_img.size
    if w < 100 or h < 100:
        return False, 0.0, "clean"

    # Analyze the 4 corners (typically where stock logos / copyright text reside)
    corner_w = int(w * 0.18)
    corner_h = int(h * 0.08)

    corners = [
        pil_img.crop((0, 0, corner_w, corner_h)),                      # Top-Left
        pil_img.crop((w - corner_w, 0, w, corner_h)),                  # Top-Right
        pil_img.crop((0, h - corner_h, corner_w, h)),                  # Bottom-Left
        pil_img.crop((w - corner_w, h - corner_h, w, h)),              # Bottom-Right
    ]

    # Check for text/watermark signatures: very high local edge contrast with high standard deviation
    for idx, c in enumerate(corners):
        c_gray = c.convert("L")
        c_arr = np.asarray(c_gray, dtype=np.float32)
        std_dev = float(np.std(c_arr))

        # Check for pure bright white / high contrast watermark stamps
        bright_ratio = float(np.mean(c_arr > 240))
        if bright_ratio > 0.35 and std_dev > 45.0:
            return True, round(bright_ratio, 2), f"high_contrast_corner_{idx}"

    return False, 0.0, "clean"
