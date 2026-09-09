"""Stufe 2 – Qualitätskontrolle: sRGB, EXIF-Strip, Mindestauflösung, pHash-Deduplizierung und Schärfefilter."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image

from pipeline.manifest import Manifest, ManifestEntry
from pipeline.qc.filters import (
    strip_exif_and_to_srgb,
    compute_laplacian_variance,
    detect_watermarks_and_text,
)
from pipeline.qc.dedup import (
    compute_phash,
    cluster_by_phash,
)

logger = logging.getLogger("pipeline.stage2_qc")


def run_stage2(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 2: Quality Control."""
    general_cfg = config.get("general", {})
    qc_cfg = config.get("stage2_qc", {})

    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)

    min_edge = int(qc_cfg.get("min_edge_px", 1024))
    laplacian_min = float(qc_cfg.get("laplacian_variance_min", 100.0))
    phash_threshold = int(qc_cfg.get("phash_hamming_threshold", 6))
    detect_watermarks = qc_cfg.get("detect_watermarks", True)
    strip_exif = qc_cfg.get("strip_exif", True)
    convert_srgb = qc_cfg.get("convert_srgb", True)

    logger.info("=" * 60)
    logger.info("  STUFE 2: QUALITÄTSKONTROLLE & DEDUPLIZIERUNG")
    logger.info("=" * 60)
    logger.info(f"Min Edge Length       : {min_edge} px")
    logger.info(f"Laplacian Var Min     : {laplacian_min}")
    logger.info(f"pHash Hamming Thresh  : <= {phash_threshold}")
    logger.info(f"Detect Watermarks     : {detect_watermarks}")
    logger.info(f"sRGB Normalization    : {convert_srgb}")
    logger.info(f"EXIF Stripping        : {strip_exif}")
    logger.info("=" * 60)

    # Process all entries ready for QC (including recovering any falsely marked missing_raw_file)
    eligible_entries = [
        e for e in manifest
        if e.stages_status.get("stage1_crawl") == "downloaded"
        and (e.stages_status.get("stage2_qc") in ["pending", None] or "missing_raw_file" in e.rejection_reasons)
    ]

    logger.info(f"Found {len(eligible_entries)} items eligible for QC.")
    if not eligible_entries:
        logger.info("No pending items for Stage 2 QC.")
        return 0

    rejection_counts: Dict[str, int] = {}
    valid_qc_items: List[Dict[str, Any]] = []

    for idx, entry in enumerate(eligible_entries, 1):
        raw_path = entry.get_raw_file()
        if not raw_path or not raw_path.exists():
            entry.update_stage("stage2_qc", "failed", reasons=["missing_raw_file"])
            continue

        if "missing_raw_file" in entry.rejection_reasons:
            entry.rejection_reasons.remove("missing_raw_file")
        entry.raw_path = str(raw_path).replace("\\", "/")

        try:
            with Image.open(raw_path) as raw_img:
                # 1. Strip EXIF and convert to sRGB
                clean_img = strip_exif_and_to_srgb(raw_img) if strip_exif or convert_srgb else raw_img
                w, h = clean_img.size
                entry.width = w
                entry.height = h
                entry.aspect_ratio = round(w / h, 3)
                entry.megapixels = round((w * h) / 1_000_000.0, 3)

                # Overwrite raw file with cleaned sRGB version (EXIF stripped)
                if strip_exif or convert_srgb:
                    clean_img.save(raw_path, format=clean_img.format or "JPEG", quality=100)

                reasons: List[str] = []

                # 2. Resolution check (Minimum edge >= 1024px)
                if min(w, h) < min_edge:
                    reasons.append(f"min_edge_below_{min_edge}")

                # 3. Sharpness calculation (Laplacian variance)
                lap_var = compute_laplacian_variance(clean_img)
                entry.qc_metrics["laplacian_variance"] = lap_var
                if lap_var < laplacian_min:
                    reasons.append("blurry_laplacian_low")

                # 4. Watermark & Text Detection
                if detect_watermarks:
                    has_wm, wm_conf, wm_detail = detect_watermarks_and_text(clean_img)
                    entry.qc_metrics["has_watermark"] = has_wm
                    entry.qc_metrics["watermark_confidence"] = wm_conf
                    if has_wm:
                        reasons.append(f"watermark_detected_{wm_detail}")

                # 5. Compute pHash
                ph = compute_phash(clean_img)
                entry.phash = ph

                if reasons:
                    entry.update_stage("stage2_qc", "rejected", reasons=reasons)
                    for r in reasons:
                        rejection_counts[r] = rejection_counts.get(r, 0) + 1
                    logger.debug(f"[QC REJECT] {raw_path.name}: {reasons}")
                else:
                    # Item passed initial QC; eligible for pHash deduplication
                    valid_qc_items.append({
                        "image_id": entry.image_id,
                        "phash": ph,
                        "score": entry.megapixels * (lap_var / 100.0),
                        "entry": entry,
                    })

        except Exception as exc:
            logger.error(f"Error processing QC for {raw_path}: {exc}")
            entry.update_stage("stage2_qc", "failed", reasons=[f"qc_exception_{type(exc).__name__}"])

    # 6. pHash Deduplication Clustering
    logger.info(f"Clustering {len(valid_qc_items)} items for pHash deduplication...")
    clusters = cluster_by_phash(valid_qc_items, threshold=phash_threshold)
    duplicate_count = 0

    for cluster_idx, cluster in enumerate(clusters, 1):
        cluster_id = f"cluster_{cluster_idx:04d}"
        if len(cluster) == 1:
            # Single item: passes cleanly
            item = cluster[0]
            item["entry"].phash_cluster_id = cluster_id
            item["entry"].update_stage("stage2_qc", "passed")
            item["entry"].stages_status["stage3_downscale"] = "pending"
        else:
            # Multi-item duplicate cluster: sort by quality score descending
            cluster.sort(key=lambda x: x["score"], reverse=True)
            best_item = cluster[0]
            best_item["entry"].phash_cluster_id = cluster_id
            best_item["entry"].update_stage("stage2_qc", "passed")
            best_item["entry"].stages_status["stage3_downscale"] = "pending"

            # Mark all others as duplicate rejects
            for dup in cluster[1:]:
                dup_entry = dup["entry"]
                dup_entry.phash_cluster_id = cluster_id
                dup_entry.update_stage(
                    "stage2_qc",
                    "rejected",
                    reasons=[f"duplicate_phash_{cluster_id}"],
                )
                rejection_counts["duplicate_phash"] = rejection_counts.get("duplicate_phash", 0) + 1
                duplicate_count += 1
                logger.debug(f"[QC DUP REJECT] {dup_entry.raw_path} -> dup of {best_item['entry'].raw_path}")

    # Save manifest atomically
    manifest.save()

    passed_count = len(manifest.filter_by_stage("stage2_qc", "passed"))
    rejected_total = sum(rejection_counts.values())

    logger.info("=" * 60)
    logger.info("  STUFE 2 ERGEBNIS")
    logger.info(f"  Bestanden (Passed)       : {passed_count}")
    logger.info(f"  Abgelehnt Gesamt         : {rejected_total}")
    for reason, count in rejection_counts.items():
        logger.info(f"    - {reason:<28}: {count}")
    logger.info("=" * 60)

    return 0
