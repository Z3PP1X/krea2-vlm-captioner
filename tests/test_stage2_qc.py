"""Unit tests for Stage 2 Quality Control filters, EXIF strip, and pHash deduplication."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from pipeline.qc.filters import (
    strip_exif_and_to_srgb,
    compute_laplacian_variance,
    detect_watermarks_and_text,
)
from pipeline.qc.dedup import (
    compute_phash,
    hamming_distance,
    cluster_by_phash,
)


def test_exif_strip_and_srgb_normalization():
    # Create an RGBA image with artificial metadata
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 255))
    clean = strip_exif_and_to_srgb(img)
    assert clean.mode == "RGB"
    assert clean.size == (200, 200)
    assert not clean.info.get("exif")


def test_laplacian_sharpness_calculation():
    # 1. Create a sharp image with high-frequency check pattern
    sharp_arr = np.zeros((300, 300), dtype=np.uint8)
    sharp_arr[::10, :] = 255
    sharp_arr[:, ::10] = 255
    sharp_img = Image.fromarray(sharp_arr)
    sharp_var = compute_laplacian_variance(sharp_img)

    # 2. Heavily blurred image
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=8))
    blurry_var = compute_laplacian_variance(blurry_img)

    assert sharp_var > 100.0
    assert blurry_var < sharp_var
    assert blurry_var < 50.0


def test_phash_computation_and_clustering():
    # Identical images (img1 and img2)
    arr1 = np.zeros((400, 400), dtype=np.uint8)
    arr1[::20, :] = 255
    img1 = Image.fromarray(arr1).convert("RGB")
    img2 = Image.fromarray(arr1).convert("RGB")

    # Distinct pattern image (diagonal bands + opposite grid)
    arr3 = np.zeros((400, 400), dtype=np.uint8)
    for i in range(400):
        arr3[i, i] = 255
        arr3[i, 399 - i] = 255
    arr3[:, ::5] = 255
    img3 = Image.fromarray(arr3).convert("RGB")

    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    h3 = compute_phash(img3)

    assert hamming_distance(h1, h2) == 0
    assert hamming_distance(h1, h3) > 10

    items = [
        {"image_id": "1", "phash": h1},
        {"image_id": "2", "phash": h2},
        {"image_id": "3", "phash": h3},
    ]

    clusters = cluster_by_phash(items, threshold=6)
    # img1 and img2 should be clustered together
    assert len(clusters) == 2
    sizes = sorted([len(c) for c in clusters])
    assert sizes == [1, 2]
