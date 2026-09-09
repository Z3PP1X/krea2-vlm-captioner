"""Unit tests for Stage 1 pre-filtering, resolution sniffing, and copyright compliance."""

import io
import struct
import pytest
from PIL import Image

from pipeline.crawler.prefilter import (
    sniff_image_dimensions,
    is_resolution_acceptable,
)
from pipeline.crawler.compliance import ComplianceChecker
from pipeline.crawler.rate_limiter import DomainRateLimiter


def create_mock_png_bytes(width: int, height: int) -> bytes:
    """Generates valid PNG header bytes."""
    header = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: 4 bytes length (13), 4 bytes "IHDR", 4 bytes width, 4 bytes height, etc.
    ihdr = b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return header + ihdr


def create_mock_jpeg_bytes(width: int, height: int) -> bytes:
    """Generates synthetic JPEG bytes with a SOF0 marker."""
    buffer = io.BytesIO()
    # Create simple 1x1 image with PIL and save to JPEG
    img = Image.new("RGB", (width, height), color="red")
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_sniff_png_resolution():
    data = create_mock_png_bytes(1920, 1080)
    dims = sniff_image_dimensions(data)
    assert dims == (1920, 1080)


def test_sniff_jpeg_resolution():
    data = create_mock_jpeg_bytes(1600, 1200)
    dims = sniff_image_dimensions(data)
    assert dims == (1600, 1200)


def test_resolution_threshold_check():
    # 800 x 600 = 480,000 px = 0.48 MP -> should be rejected under 0.7 MP
    ok, mp = is_resolution_acceptable(800, 600, min_megapixels=0.7)
    assert not ok
    assert round(mp, 2) == 0.48

    # 1200 x 800 = 960,000 px = 0.96 MP -> should be accepted
    ok, mp = is_resolution_acceptable(1200, 800, min_megapixels=0.7)
    assert ok
    assert round(mp, 2) == 0.96

    # 1600 x 1600 = 2.56 MP -> accepted
    ok, mp = is_resolution_acceptable(1600, 1600, min_megapixels=0.7)
    assert ok
    assert round(mp, 2) == 2.56


def test_tdm_reservation_header_detection():
    compliance = ComplianceChecker(user_agent="TestBot", filter_tdm=True)

    # 1. Negative case
    is_res, reason = compliance.check_tdm_reservation(headers={"Content-Type": "image/jpeg"})
    assert not is_res

    # 2. Positive case via W3C TDM Reservation header
    is_res, reason = compliance.check_tdm_reservation(headers={"tdm-reservation": "1"})
    assert is_res
    assert "§ 44b UrhG" in reason

    # 3. Positive case via X-Robots-Tag: noai
    is_res, reason = compliance.check_tdm_reservation(headers={"X-Robots-Tag": "noai, noimageai"})
    assert is_res
    assert "noai" in reason


def test_tdm_reservation_html_meta_detection():
    compliance = ComplianceChecker(user_agent="TestBot", filter_tdm=True)

    # HTML with TDM meta tag
    html = """
    <html>
      <head>
        <meta name="tdm-reservation" content="1">
      </head>
      <body>Page content</body>
    </html>
    """
    is_res, reason = compliance.check_tdm_reservation(html_content=html)
    assert is_res
    assert "tdm-reservation=1" in reason

    # HTML with robots noai tag
    html_noai = """
    <html>
      <head>
        <meta name="robots" content="index, follow, noai">
      </head>
      <body>Page content</body>
    </html>
    """
    is_res, reason = compliance.check_tdm_reservation(html_content=html_noai)
    assert is_res
    assert "noai" in reason


def test_domain_rate_limiter():
    limiter = DomainRateLimiter(default_delay=0.05)
    import time
    start = time.time()
    limiter.wait("https://example.com/page1")
    limiter.wait("https://example.com/page2")
    duration = time.time() - start
    assert duration >= 0.04
