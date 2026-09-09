"""Main CLI entrypoint for Krea 2 LoRA Data Pipeline."""

from __future__ import annotations

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.manifest import Manifest
from pipeline.logging_utils import setup_logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAVE_RICH = True
    console = Console()
except ImportError:
    HAVE_RICH = False
    console = None


def load_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """Loads YAML pipeline configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def print_status_table(summary: Dict[str, Any]) -> None:
    """Displays rich terminal status summary of the pipeline."""
    total = summary.get("total_items", 0)
    stages = summary.get("stages", {})
    rejections = summary.get("rejections", {})

    if HAVE_RICH:
        console.print(Panel(f"[bold cyan]Krea 2 LoRA Pipeline Status[/bold cyan] — Total Items: [bold]{total}[/bold]"))

        table = Table(title="Stage Processing Breakdown", header_style="bold magenta")
        table.add_column("Stage Name", style="cyan", width=22)
        table.add_column("Pending", justify="right", style="yellow")
        table.add_column("Completed / Passed", justify="right", style="green")
        table.add_column("Rejected / Failed", justify="right", style="red")
        table.add_column("Other / Skipped", justify="right", style="dim")

        for stage_name, counts in stages.items():
            pending = str(counts.get("pending", 0))
            passed = str(
                counts.get("passed", 0)
                or counts.get("downloaded", 0)
                or counts.get("done", 0)
                or counts.get("captioned", 0)
                or counts.get("exported", 0)
            )
            rejected = str(
                counts.get("rejected", 0)
                or counts.get("rejected_screening", 0)
                or counts.get("failed", 0)
            )
            skipped = str(counts.get("skipped", 0))

            table.add_row(stage_name, pending, passed, rejected, skipped)

        console.print(table)

        if rejections:
            rej_table = Table(title="Rejection Reasons Summary", header_style="bold red")
            rej_table.add_column("Reason", style="white")
            rej_table.add_column("Count", justify="right", style="bold red")
            for reason, count in rejections.items():
                rej_table.add_row(reason, str(count))
            console.print(rej_table)
    else:
        print(f"=== Krea 2 Pipeline Status (Total: {total}) ===")
        for stage_name, counts in stages.items():
            print(f"  [{stage_name}]: {counts}")
        if rejections:
            print("  Rejections:", rejections)


def handle_status(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Handles 'status' subcommand."""
    manifest_path = config.get("general", {}).get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)
    summary = manifest.summary()
    print_status_table(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs CLI argument parser with subcommands for each stage."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Krea 2 LoRA End-to-End Data Pipeline CLI",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to pipeline configuration file (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline Stage Command")

    # Status
    status_parser = subparsers.add_parser("status", help="Show manifest processing status and metrics")
    status_parser.set_defaults(func=handle_status)

    # Stage 1: Crawl
    crawl_parser = subparsers.add_parser("crawl", help="Stufe 1: Crawl images with pre-download resolution filter")
    crawl_parser.add_argument("--url", type=str, help="Target URL (XenForo thread or dbNaked channel)")
    crawl_parser.add_argument("--pages", type=str, help="Pages to crawl (e.g. '1-3', '27', 'all')")
    crawl_parser.add_argument("-w", "--max-workers", type=int, default=None, help="Concurrent download workers (e.g. 4, 8, 16)")
    crawl_parser.add_argument("--delay", type=float, default=None, help="Rate limit delay in seconds per domain")

    # Stage 2: QC
    qc_parser = subparsers.add_parser("qc", help="Stufe 2: Quality control, sRGB, EXIF strip, and pHash deduplication")
    qc_parser.add_argument("-w", "--max-workers", type=int, default=None, help="Concurrent CPU QC workers (e.g. 8, 16)")
    qc_parser.add_argument("-e", "--min-edge", type=int, default=None, help="Minimum width and height for image file in pixels (default: 512)")
    qc_parser.add_argument("--force", action="store_true", help="Re-run QC on all downloaded images, resetting previous rejections")

    # Stage 3: Downscale
    downscale_parser = subparsers.add_parser("downscale", help="Stufe 3: Downscale to max 2048px (multiples of 16)")
    downscale_parser.add_argument("-w", "--max-workers", type=int, default=None, help="Concurrent CPU resizing workers (e.g. 8)")
    downscale_parser.add_argument("--max-dim", type=int, default=None, help="Max bounding box edge in pixels (default: 2048)")

    # Stage 4: Caption
    caption_parser = subparsers.add_parser("caption", help="Stufe 4: Qwen-VL Offline-Batch captioning & screening")
    caption_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Parallel GPU batch size (e.g. 16, 32, 64)")
    caption_parser.add_argument("-m", "--model", type=str, default=None, help="Hugging Face model ID (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    caption_parser.add_argument("--mode", type=str, choices=["style", "subject"], default=None, help="Caption mode ('style' omits style keywords; 'subject' includes all)")
    caption_parser.add_argument("-t", "--trigger", type=str, default=None, help="Trigger token prepended to captions")
    caption_parser.add_argument("--sample", type=int, default=None, help="Process random sample (e.g. --sample 200)")
    caption_parser.add_argument("--max-model-len", type=int, default=None, help="Maximum context length in tokens (default: 12288)")
    caption_parser.add_argument("--ignore-watermarks", action="store_true", help="Do not reject images due to watermarks or text overlays")
    caption_parser.add_argument("--no-screening", action="store_true", help="Bypass all screening gates and keep all generated captions")
    caption_parser.add_argument("--force", action="store_true", help="Force recaptioning of all valid images, ignoring previous status")
    caption_parser.add_argument("--retry-failed", action="store_true", default=True, help="Retry images that previously failed captioning (default: True)")

    # Stage 5: Export
    export_parser = subparsers.add_parser("export", help="Stufe 5: Export dataset for AI-Toolkit with balanced repeats")
    export_parser.add_argument("--ai-toolkit-dir", type=str, default=None, help="Path to AI-Toolkit installation (e.g. /app/ai-toolkit) for automatic symlink and config integration")
    export_parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name inside AI-Toolkit (default: restrained_elegance)")

    # Stage 6: Tooling
    train_parser = subparsers.add_parser("train-config", help="Stufe 6: Generate AI-Toolkit Krea 2 RAW training config")

    # Stage 7: Validation
    validate_parser = subparsers.add_parser("validate", help="Stufe 7: Generate prompt grid & validation contact sheet")

    # Stage 8: Character Stack
    char_parser = subparsers.add_parser("character-config", help="Stufe 8: Generate sequential character LoRA configs")

    # Run All
    run_all_parser = subparsers.add_parser("run-all", help="Execute Stages 1 to 5 sequentially")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {}

    log_level = "DEBUG" if args.verbose else config.get("general", {}).get("log_level", "INFO")
    logger = setup_logging(level=log_level)

    if hasattr(args, "func"):
        return args.func(args, config)

    # Dispatches to stage handlers as they are implemented
    if args.command == "crawl":
        from pipeline.stages.stage1_crawl import run_stage1
        return run_stage1(args, config)
    elif args.command == "qc":
        from pipeline.stages.stage2_qc import run_stage2
        return run_stage2(args, config)
    elif args.command == "downscale":
        from pipeline.stages.stage3_downscale import run_stage3
        return run_stage3(args, config)
    elif args.command == "caption":
        from pipeline.stages.stage4_caption import run_stage4
        return run_stage4(args, config)
    elif args.command == "export":
        from pipeline.stages.stage5_export import run_stage5
        return run_stage5(args, config)
    elif args.command == "train-config":
        from pipeline.tooling.train_config import run_train_config
        return run_train_config(args, config)
    elif args.command == "validate":
        from pipeline.tooling.validation_grid import run_validation_grid
        return run_validation_grid(args, config)
    elif args.command == "character-config":
        from pipeline.tooling.character_stack import run_character_stack
        return run_character_stack(args, config)
    elif args.command == "run-all":
        logger.info("Executing full data pipeline (Stages 1 through 5)...")
        from pipeline.stages.stage1_crawl import run_stage1
        from pipeline.stages.stage2_qc import run_stage2
        from pipeline.stages.stage3_downscale import run_stage3
        from pipeline.stages.stage4_caption import run_stage4
        from pipeline.stages.stage5_export import run_stage5

        rc = run_stage1(args, config)
        if rc != 0:
            return rc
        rc = run_stage2(args, config)
        if rc != 0:
            return rc
        rc = run_stage3(args, config)
        if rc != 0:
            return rc
        rc = run_stage4(args, config)
        if rc != 0:
            return rc
        return run_stage5(args, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
