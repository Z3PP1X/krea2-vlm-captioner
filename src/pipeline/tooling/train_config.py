"""Stufe 6 – Trainings-Konfigurationsgenerator für AI-Toolkit (Krea 2 RAW & Curated)."""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml

from pipeline.manifest import Manifest

logger = logging.getLogger("pipeline.train_config")


def select_curated_subset(
    manifest: Manifest,
    target_count: int = 2500,
) -> List[Any]:
    """Selects top N images based on sharpness, resolution, and compliant screening."""
    eligible = [
        e for e in manifest
        if e.stages_status.get("stage4_caption") == "captioned"
        and not e.is_rejected()
        and e.processed_path
    ]

    # Score by megapixels * laplacian variance
    def score(e):
        lap = e.qc_metrics.get("laplacian_variance", 100.0)
        mp = e.megapixels or 1.0
        return mp * (lap / 100.0)

    eligible.sort(key=score, reverse=True)
    return eligible[:target_count]


def run_train_config(args: Any, config: Dict[str, Any]) -> int:
    """Generates Krea 2 RAW and Curated AI-Toolkit training configurations."""
    general_cfg = config.get("general", {})
    train_cfg = config.get("stage6_training", {})

    export_dir = Path(general_cfg.get("export_dir", "data/export"))
    processed_dir = Path(general_cfg.get("processed_dir", "data/processed")) / "images"
    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)

    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Full Set Config for Krea 2 RAW
    template_raw_path = Path("templates/ai_toolkit_krea2_raw.yaml")
    if template_raw_path.exists():
        with open(template_raw_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    else:
        raw_cfg = {"config": {"name": "krea2_style_lora_raw"}}

    out_raw_file = export_dir / "ai_toolkit_krea2_raw.yaml"
    with open(out_raw_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=False)
    logger.info(f"Generated Krea 2 RAW training config: {out_raw_file.resolve()}")

    # 2. Build Curated 2.5k Subset
    curated_items = select_curated_subset(manifest, target_count=2500)
    curated_dir = export_dir / "curated_subset"
    curated_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Populating curated subset ({len(curated_items)} items) in {curated_dir.resolve()}...")
    for item in curated_items:
        src_img = Path(item.processed_path) if item.processed_path else None
        if src_img and src_img.exists():
            dest_img = curated_dir / src_img.name
            if not dest_img.exists():
                shutil.copy2(src_img, dest_img)

            src_txt = src_img.with_suffix(".txt")
            if src_txt.exists():
                dest_txt = curated_dir / src_txt.name
                if not dest_txt.exists():
                    shutil.copy2(src_txt, dest_txt)

    # 3. Generate Curated Set Config
    template_curated_path = Path("templates/ai_toolkit_krea2_curated.yaml")
    if template_curated_path.exists():
        with open(template_curated_path, "r", encoding="utf-8") as f:
            curated_cfg = yaml.safe_load(f)
    else:
        curated_cfg = {"config": {"name": "krea2_style_lora_curated_2k"}}

    out_curated_file = export_dir / "ai_toolkit_krea2_curated.yaml"
    with open(out_curated_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(curated_cfg, f, sort_keys=False)
    logger.info(f"Generated Curated 2.5k training config: {out_curated_file.resolve()}")

    return 0
