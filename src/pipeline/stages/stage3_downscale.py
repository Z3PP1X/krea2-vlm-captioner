"""Stufe 3 – Downscaling: Lange Kante max 2048px, Kantenlängen mod 16, Lanczos-Resampling."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    workers = getattr(args, "max_workers", None) or 8
    max_long_edge = getattr(args, "max_dim", None) or int(downscale_cfg.get("max_long_edge", 2048))
    multiple_of = int(downscale_cfg.get("multiple_of", 16))
    out_format = downscale_cfg.get("format", "jpeg").lower()
    jpeg_quality = int(downscale_cfg.get("jpeg_quality", 95))

    logger.info("=" * 60)
    logger.info("  STUFE 3: DOWNSCALING & STANDARDIZATION")
    logger.info("=" * 60)
    logger.info(f"Workers               : {workers}")
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
    failed_count = 0

    def process_downscale(entry: ManifestEntry) -> tuple[str, Any]:
        raw_path = entry.get_raw_file()
        if not raw_path or not raw_path.exists():
            entry.update_stage("stage3_downscale", "failed", reasons=["missing_raw_file"])
            return ("failed", entry.image_id, "missing_raw_file")

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

                if out_format == "jpeg" and resampled.mode != "RGB":
                    resampled = resampled.convert("RGB")

                ext = ".jpg" if out_format == "jpeg" else ".png"
                dest_filename = f"{entry.image_id[:16]}{ext}"
                dest_file_path = processed_dir / dest_filename

                if out_format == "jpeg":
                    resampled.save(dest_file_path, format="JPEG", quality=jpeg_quality, optimize=True)
                else:
                    resampled.save(dest_file_path, format="PNG", optimize=True)

                rel_processed_path = os.path.relpath(dest_file_path, Path(general_cfg.get("manifest_path", "data")).parent).replace("\\", "/")

                entry.processed_path = rel_processed_path
                entry.width = new_w
                entry.height = new_h
                entry.aspect_ratio = round(new_w / new_h, 3)
                entry.megapixels = round((new_w * new_h) / 1_000_000.0, 3)
                entry.update_stage("stage3_downscale", "done")
                entry.stages_status["stage4_caption"] = "pending"

                return ("success", raw_path.name, orig_w, orig_h, new_w, new_h)

        except Exception as exc:
            entry.update_stage("stage3_downscale", "failed", reasons=[f"downscale_error_{type(exc).__name__}"])
            return ("failed", entry.image_id, str(exc))

    if workers <= 1:
        for idx, entry in enumerate(eligible, 1):
            res = process_downscale(entry)
            if res[0] == "success":
                success_count += 1
                logger.info(f"[{idx}/{len(eligible)}] Downscaled {res[1]}: {res[2]}x{res[3]} -> {res[4]}x{res[5]}")
            else:
                failed_count += 1
    else:
        logger.info(f"Downscaling concurrently with {workers} CPU workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_entry = {executor.submit(process_downscale, entry): entry for entry in eligible}
            done_idx = 0
            for future in as_completed(future_to_entry):
                done_idx += 1
                try:
                    res = future.result()
                    if res[0] == "success":
                        success_count += 1
                        if done_idx % 25 == 0 or done_idx == len(eligible):
                            logger.info(f"[{done_idx}/{len(eligible)}] Processed downscaling ({success_count} succeeded)")
                    else:
                        failed_count += 1
                except Exception as exc:
                    failed_count += 1
                    logger.error(f"Worker exception during downscale: {exc}")

    manifest.save()

    logger.info("=" * 60)
    logger.info(f"  STUFE 3 ERGEBNIS: {success_count} Bilder erfolgreich aufbereitet (Fehlgeschlagen: {failed_count})")
    logger.info("=" * 60)

    return 0
