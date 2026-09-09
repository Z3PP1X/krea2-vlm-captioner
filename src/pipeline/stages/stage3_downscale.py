"""Stufe 3 – Downscaling: Lange Kante max 2048px, Kantenlängen mod 16, Lanczos-Resampling."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image

from pipeline.manifest import Manifest, ManifestEntry

logger = logging.getLogger("pipeline.stage3_downscale")


def calculate_downscale_dimensions(
    orig_w: int,
    orig_h: int,
    max_long_edge: int = 2048,
    multiple_of: int = 16,
) -> Tuple[int, int]:
    """Calculates downscaled dimensions capped at max_long_edge and rounded to multiples of 16.
    
    Rules:
    1. Downscale only; never upscale.
    2. Preserve aspect ratio as closely as possible.
    3. Both dimensions must be positive multiples of `multiple_of` (e.g. 16).
    """
    long_edge = max(orig_w, orig_h)

    if long_edge > max_long_edge:
        scale = max_long_edge / float(long_edge)
        target_w = orig_w * scale
        target_h = orig_h * scale
    else:
        target_w = float(orig_w)
        target_h = float(orig_h)

    # Round to nearest multiple of 16
    new_w = max(multiple_of, int(round(target_w / multiple_of) * multiple_of))
    new_h = max(multiple_of, int(round(target_h / multiple_of) * multiple_of))

    return new_w, new_h


def run_stage3(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 3: Downscaling & Pre-Processing."""
    general_cfg = config.get("general", {})
    downscale_cfg = config.get("stage3_downscale", {})

    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)

    processed_dir = Path(general_cfg.get("processed_dir", "data/processed")) / "images"
    processed_dir.mkdir(parents=True, exist_ok=True)

    max_long_edge = int(downscale_cfg.get("max_long_edge", 2048))
    multiple_of = int(downscale_cfg.get("multiple_of", 16))
    out_format = downscale_cfg.get("format", "jpeg").lower()
    jpeg_quality = int(downscale_cfg.get("jpeg_quality", 95))

    logger.info("=" * 60)
    logger.info("  STUFE 3: DOWNSCALING & STANDARDIZATION")
    logger.info("=" * 60)
    logger.info(f"Max Long Edge         : {max_long_edge} px (No upscaling)")
    logger.info(f"Multiple of (DiT/VAE) : {multiple_of} px")
    logger.info(f"Output Format         : {out_format.upper()} (Quality {jpeg_quality})")
    logger.info(f"Destination Dir       : {processed_dir.resolve()}")
    logger.info("=" * 60)

    # Eligible entries: passed QC and pending downscaling
    eligible = [
        e for e in manifest
        if e.stages_status.get("stage2_qc") == "passed"
        and e.stages_status.get("stage3_downscale") in ["pending", None]
    ]

    logger.info(f"Found {len(eligible)} images eligible for downscaling.")
    if not eligible:
        logger.info("No pending items for Stage 3 Downscale.")
        return 0

    success_count = 0

    for idx, entry in enumerate(eligible, 1):
        raw_path = Path(entry.raw_path) if entry.raw_path else None
        if not raw_path or not raw_path.exists():
            logger.warning(f"Raw file missing for {entry.image_id}: {raw_path}")
            entry.update_stage("stage3_downscale", "failed", reasons=["missing_raw_file"])
            continue

        try:
            with Image.open(raw_path) as img:
                orig_w, orig_h = img.size
                new_w, new_h = calculate_downscale_dimensions(
                    orig_w,
                    orig_h,
                    max_long_edge=max_long_edge,
                    multiple_of=multiple_of,
                )

                if (new_w, new_h) != (orig_w, orig_h):
                    resampled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    resampled = img.copy()

                # Ensure RGB mode for JPEG
                if out_format == "jpeg" and resampled.mode != "RGB":
                    resampled = resampled.convert("RGB")

                # Destination file naming
                ext = ".jpg" if out_format == "jpeg" else ".png"
                dest_filename = f"{entry.image_id[:16]}{ext}"
                dest_file_path = processed_dir / dest_filename

                if out_format == "jpeg":
                    resampled.save(dest_file_path, format="JPEG", quality=jpeg_quality, optimize=True)
                else:
                    resampled.save(dest_file_path, format="PNG", optimize=True)

                rel_processed_path = os.path.relpath(dest_file_path, Path(general_cfg.get("manifest_path", "data")).parent).replace("\\", "/")

                # Update Manifest
                entry.processed_path = rel_processed_path
                entry.width = new_w
                entry.height = new_h
                entry.aspect_ratio = round(new_w / new_h, 3)
                entry.megapixels = round((new_w * new_h) / 1_000_000.0, 3)
                entry.update_stage("stage3_downscale", "done")
                entry.stages_status["stage4_caption"] = "pending"

                success_count += 1
                logger.info(f"[{idx}/{len(eligible)}] Downscaled {raw_path.name}: {orig_w}x{orig_h} -> {new_w}x{new_h}")

        except Exception as exc:
            logger.error(f"Error downscaling {raw_path}: {exc}")
            entry.update_stage("stage3_downscale", "failed", reasons=[f"downscale_error_{type(exc).__name__}"])

    manifest.save()

    logger.info("=" * 60)
    logger.info(f"  STUFE 3 ERGEBNIS: {success_count} Bilder erfolgreich aufbereitet")
    logger.info("=" * 60)

    return 0
