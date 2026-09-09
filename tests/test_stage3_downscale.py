"""Unit tests for Stage 3 Downscaling and Multiples-of-16 calculations."""

import pytest
from pipeline.stages.stage3_downscale import calculate_downscale_dimensions


def test_downscale_large_image():
    # 4000 x 3000 -> long edge 4000 scaled down to max 2048
    # aspect ratio 4:3 -> 2048 x 1536
    w, h = calculate_downscale_dimensions(4000, 3000, max_long_edge=2048, multiple_of=16)
    assert max(w, h) == 2048
    assert w % 16 == 0
    assert h % 16 == 0
    # 1536 is exactly 96 * 16
    assert h == 1536


def test_no_upscale_smaller_image():
    # 1024 x 768 is smaller than 2048 -> do NOT upscale, only ensure mod 16
    w, h = calculate_downscale_dimensions(1024, 768, max_long_edge=2048, multiple_of=16)
    assert w == 1024
    assert h == 768
    assert w % 16 == 0
    assert h % 16 == 0


def test_non_multiple_of_16_rounding():
    # 1500 x 999 -> long edge 1500 <= 2048, both rounded to multiples of 16
    w, h = calculate_downscale_dimensions(1500, 999, max_long_edge=2048, multiple_of=16)
    assert w % 16 == 0
    assert h % 16 == 0
    assert abs(w - 1500) < 16
    assert abs(h - 999) < 16
