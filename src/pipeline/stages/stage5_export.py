"""Stufe 5 – Datensatz strukturieren und Export für AI-Toolkit & musubi-tuner."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any

from pipeline.manifest import Manifest
from pipeline.export.distribution import analyze_dataset_distribution
from pipeline.export.balancer import calculate_subconcept_repeats
from pipeline.export.ai_toolkit_builder import (
    generate_ai_toolkit_dataset_config,
    write_ai_toolkit_dataset_yaml,
)
from pipeline.export.ai_toolkit_integrator import integrate_with_ai_toolkit
from pipeline.export.musubi_builder import generate_musubi_dataset_toml

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAVE_RICH = True
    console = Console()
except ImportError:
    HAVE_RICH = False
    console = None

logger = logging.getLogger("pipeline.stage5_export")


def write_distribution_markdown_report(
    analysis: Dict[str, Any],
    repeats: Dict[str, int],
    output_path: Path,
) -> None:
    """Writes human-readable distribution analysis markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = analysis.get("total_eligible_items", 0)

    lines = [
        "# Krea 2 LoRA Dataset — Distribution & Balancing Report",
        "",
        f"**Total Compliant Images Exported:** {total}",
        "",
        "## 1. Style Distribution",
        "| Style | Image Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]
    for style, count in analysis.get("by_style", {}).items():
        pct = (count / total * 100) if total else 0
        lines.append(f"| `{style}` | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## 2. Location Distribution",
        "| Location | Image Count | Percentage |",
        "| :--- | :--- | :--- |",
    ])
    for loc, count in analysis.get("by_location", {}).items():
        pct = (count / total * 100) if total else 0
        lines.append(f"| `{loc}` | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## 3. Pose / Subconcept Distribution & Balanced Repeats",
        "| Pose / Practice | Image Count | Balanced `num_repeats` | Effective Images |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for pose, count in analysis.get("by_pose", {}).items():
        rep = repeats.get(pose, 1)
        eff = count * rep
        lines.append(f"| `{pose}` | {count} | **{rep}x** | {eff} |")

    lines.extend([
        "",
        "## 4. Aspect Ratio Buckets",
        "| Ratio Category | Count |",
        "| :--- | :--- |",
    ])
    for ar_cat, count in analysis.get("aspect_ratios", {}).items():
        lines.append(f"| {ar_cat} | {count} |")

    warnings = analysis.get("dominance_warnings", [])
    if warnings:
        lines.extend([
            "",
            "## ⚠️ Dominance Warnings (> Threshold)",
        ])
        for w in warnings:
            lines.append(f"- **Warning**: Scene `{w['scene']}` represents **{w['ratio']*100:.1f}%** of the entire dataset ({w['count']} images, threshold: {w['threshold']*100:.0f}%). Consider downsampling or diversifying to prevent identity overfit.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_stage5(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 5: Dataset Structuring and Export."""
    general_cfg = config.get("general", {})
    export_cfg = config.get("stage5_export", {})
    cap_cfg = config.get("stage4_caption", {})

    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)

    processed_images_dir = Path(general_cfg.get("processed_dir", "data/processed")) / "images"
    export_dir = Path(general_cfg.get("export_dir", "data/export"))
    reports_dir = Path(general_cfg.get("reports_dir", "data/reports"))

    max_balance = float(export_cfg.get("max_balance_factor", 4.0))
    max_dominance = float(export_cfg.get("max_person_dominance_ratio", 0.35))
    trigger_word = cap_cfg.get("trigger_word", "restrained_elegance")

    logger.info("=" * 60)
    logger.info("  STUFE 5: DATENSATZ STRUKTURIEREN & EXPORT")
    logger.info("=" * 60)
    logger.info(f"Target Framework       : {export_cfg.get('target_framework', 'ai_toolkit')}")
    logger.info(f"Max Balance Factor     : {max_balance}x")
    logger.info(f"Max Dominance Ratio    : {max_dominance*100:.0f}%")
    logger.info(f"Export Directory       : {export_dir.resolve()}")
    logger.info("=" * 60)

    # 1. Analyze distribution across valid captioned images
    analysis = analyze_dataset_distribution(manifest, max_person_dominance_ratio=max_dominance)
    total_eligible = analysis.get("total_eligible_items", 0)
    logger.info(f"Total captioned items eligible for export: {total_eligible}")

    # 2. Calculate balanced repeats
    repeats_by_pose = calculate_subconcept_repeats(analysis.get("by_pose", {}), max_factor=max_balance)

    # 3. Build and write AI-Toolkit Dataset YAML
    ai_toolkit_cfg = generate_ai_toolkit_dataset_config(
        images_dir=processed_images_dir,
        trigger_word=trigger_word,
        repeats_by_pose=repeats_by_pose,
        aspect_ratio_bucketing=bool(export_cfg.get("aspect_ratio_bucketing", True)),
    )
    ai_yaml_path = export_dir / "ai_toolkit_dataset.yaml"
    write_ai_toolkit_dataset_yaml(ai_yaml_path, ai_toolkit_cfg)
    logger.info(f"Generated AI-Toolkit config: {ai_yaml_path.resolve()}")

    # 4. Generate musubi-tuner dataset.toml as companion
    musubi_toml_path = export_dir / "dataset.toml"
    generate_musubi_dataset_toml(images_dir=processed_images_dir, output_path=musubi_toml_path)
    logger.info(f"Generated musubi-tuner dataset.toml: {musubi_toml_path.resolve()}")

    # 5. Write distribution report
    dist_report_path = reports_dir / "distribution_report.md"
    write_distribution_markdown_report(analysis, repeats_by_pose, dist_report_path)
    logger.info(f"Generated Distribution Report: {dist_report_path.resolve()}")

    # 6. Update Manifest status for exported items
    for entry in manifest:
        if entry.stages_status.get("stage4_caption") == "captioned" and not entry.is_rejected():
            entry.update_stage("stage5_export", "exported")
    manifest.save()

    # 7. Automated AI-Toolkit Integration (symlink, SQLite DB, training config)
    auto_link = bool(export_cfg.get("auto_link_ai_toolkit", True))
    custom_ai_dir = getattr(args, "ai_toolkit_dir", None) or export_cfg.get("ai_toolkit_dir")
    dataset_name = getattr(args, "dataset_name", None) or export_cfg.get("dataset_name", trigger_word)

    if auto_link:
        integration = integrate_with_ai_toolkit(
            images_dir=processed_images_dir,
            dataset_name=dataset_name,
            trigger_word=trigger_word,
            custom_ai_toolkit_dir=custom_ai_dir,
        )
        if integration.get("ai_toolkit_found"):
            logger.info(f"AI-Toolkit Directory detected: {integration['ai_toolkit_dir']}")
            if integration.get("linked"):
                logger.info(f"  ✓ Symlink created: {integration['link_path']}")
            if integration.get("db_registered"):
                logger.info(f"  ✓ Dataset registered in SQLite DB: {integration['db_path']}")
            if integration.get("config_installed"):
                logger.info(f"  ✓ Training config installed: {integration['config_path']}")
        else:
            logger.info("AI-Toolkit directory not detected. Automatic integration skipped.")

    # Terminal summary display
    if HAVE_RICH and console:
        table = Table(title="Subconcept Distribution & Calculated Repeats", header_style="bold cyan")
        table.add_column("Pose / Practice", style="white")
        table.add_column("Raw Count", justify="right", style="yellow")
        table.add_column("Multiplier", justify="right", style="bold green")
        table.add_column("Effective Count", justify="right", style="bold green")

        for pose, count in analysis.get("by_pose", {}).items():
            rep = repeats_by_pose.get(pose, 1)
            table.add_row(pose, str(count), f"{rep}x", str(count * rep))

        console.print(table)

    warnings = analysis.get("dominance_warnings", [])
    if warnings:
        for w in warnings:
            logger.warning(
                f"[DOMINANCE WARNING] Scene '{w['scene']}' constitutes {w['ratio']*100:.1f}% of the dataset! "
                f"Threshold: {w['threshold']*100:.0f}%."
            )

    logger.info("=" * 60)
    logger.info("  STUFE 5 ERGEBNIS: EXPORT ERFOLGREICH")
    logger.info("=" * 60)

    return 0
