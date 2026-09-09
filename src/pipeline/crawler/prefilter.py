"""Image resolution pre-filter without downloading full image payloads.

Inspects HTML metadata, HTTP headers, or reads the first few bytes (via HTTP Range requests)
to determine dimensions (width, height) and filter out images below the minimum megapixel threshold (e.g. 0.7 MP).
"""

from __future__ import annotations

import io
import struct
import logging
from typing import Optional, Tuple
import requests

logger = logging.getLogger("pipeline.crawler.prefilter")


def sniff_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Determines (width, height) from the initial bytes of an image stream.
    
    Supports JPEG, PNG, GIF, and WebP.
    """
    if len(data) < 16:
        return None

    # 1. PNG check: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return width, height

    # 2. GIF check: GIF87a or GIF89a
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        if len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return width, height

    # 3. WebP check: RIFF....WEBP
    if data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP":
        format_fourcc = data[12:16]
        if format_fourcc == b"VP8 ":  # Lossy simple
            if len(data) >= 30:
                # Keyframe check
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
        elif format_fourcc == b"VP8L":  # Lossless simple
            if len(data) >= 25:
                b0, b1, b2, b3 = data[21:25]
                width = 1 + (((b1 & 0x3F) << 8) | b0)
                height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                return width, height
        elif format_fourcc == b"VP8X":  # Extended format
            if len(data) >= 30:
                # 24-bit width and height at offset 24
                w = 1 + struct.unpack("<I", data[24:27] + b"\x00")[0]
                h = 1 + struct.unpack("<I", data[27:30] + b"\x00")[0]
                return w, h

    # 4. JPEG check: \xFF\xD8
    if data.startswith(b"\xff\xd8"):
        try:
            stream = io.BytesIO(data)
            stream.read(2)  # Skip SOI
            while True:
                marker_bytes = stream.read(2)
                if len(marker_bytes) < 2:
                    break
                if marker_bytes[0] != 0xFF:
                    break
                marker = marker_bytes[1]
                # Ignore padding
                while marker == 0xFF:
                    b = stream.read(1)
                    if not b:
                        break
                    marker = b[0]

                # SOF markers containing dimensions:
                # SOF0 (Baseline), SOF1 (Extended), SOF2 (Progressive), SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15
                if marker in [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]:
                    length = struct.unpack(">H", stream.read(2))[0]
                    precision = stream.read(1)[0]
                    h, w = struct.unpack(">HH", stream.read(4))
                    return w, h
                elif marker in [0xD8, 0xD9]:  # SOI, EOI
                    continue
                elif marker == 0xDA:  # SOS: Start of scan, header parsing ends
                    break
                else:
                    # Skip other chunks
                    chunk_len_bytes = stream.read(2)
                    if len(chunk_len_bytes) < 2:
                        break
                    chunk_len = struct.unpack(">H", chunk_len_bytes)[0]
                    if chunk_len < 2:
                        break
                    stream.seek(chunk_len - 2, io.SEEK_CUR)
        except Exception:
            pass

    return None


def fetch_remote_resolution(
    url: str,
    session: requests.Session,
    headers: Optional[dict] = None,
    timeout: float = 15.0,
    range_bytes: int = 32768,
) -> Optional[Tuple[int, int]]:
    """Attempts to determine remote image resolution using an HTTP Range request without downloading the full image."""
    req_headers = {"Range": f"bytes=0-{range_bytes - 1}"}
    if headers:
        req_headers.update(headers)

    try:
        # First attempt Range request
        resp = session.get(url, headers=req_headers, timeout=timeout, stream=True)
        if resp.status_code in [200, 206]:
            chunk = resp.raw.read(range_bytes)
            dims = sniff_image_dimensions(chunk)
            if dims:
                return dims
    except Exception as exc:
        logger.debug(f"Range check failed for {url}: {exc}")

    return None


def is_resolution_acceptable(
    width: Optional[int],
    height: Optional[int],
    min_megapixels: float = 0.7,
) -> Tuple[bool, float]:
    """Returns whether dimensions meet the minimum megapixel threshold (e.g. 0.7 MP)."""
    if width is None or height is None or width <= 0 or height <= 0:
        return False, 0.0
    mp = (width * height) / 1_000_000.0
    return mp >= min_megapixels, mp
