#!/usr/bin/env python3
"""Dataset Caption Refinement Tool (Krea 2 & Shibari/Bondage Skills).

Iterates over an already tagged image dataset (a directory of images + .txt files
or a manifest.jsonl), feeds both the image and the existing caption to Google Gemma 4 12B,
audits visual accuracy, corrects inaccuracies, and expands the caption into an exhaustive
7-layer Krea 2 visual narrative of at least 400 tokens (~280-350+ words).
"""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.logging_utils import setup_logging
from pipeline.captioning.schema import (
    load_vocabulary,
    get_vllm_json_schema,
    build_system_prompt,
    determine_caption_tier,
    build_tier_prompt,
)
from pipeline.captioning.assembly import assemble_caption
from pipeline.captioning.hf_gemma4_engine import Gemma4HfEngine

logger = logging.getLogger("dataset.caption_refiner")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and elevate existing dataset captions into concise, dense Krea 2 visual narratives (max 340 tokens)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to folder containing images and .txt caption pairs.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of dataset in AI-Toolkit (e.g. 'restrained_elegance' -> /app/ai-toolkit/datasets/restrained_elegance).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.jsonl (alternative to --dataset-dir).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-12B-it",
        help="VLM model name (default: google/gemma-4-12B-it).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sampling temperature (default: 0.75).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=340,
        help="Maximum generation token budget (default: 340, 1/3 of previous limit).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=150,
        help="Target minimum caption tokens (default: 150).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch processing chunk size (default: 32).",
    )
    parser.add_argument(
        "--parallel-sub-batch",
        type=int,
        default=8,
        help="Parallel GPU tensor batch size (default: 8, up to 16 for 96GB VRAM).",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Optional override trigger token to prepend as token 0 (e.g. 'kink, sexandsubmission').",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .txt.bak backups of original captions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-refine all captions even if a .txt.bak backup exists.",
    )
    parser.add_argument(
        "--show-captions",
        dest="show_captions",
        action="store_true",
        default=True,
        help="Display full generated captions in the terminal as they are processed (default: True).",
    )
    parser.add_argument(
        "--no-show-captions",
        dest="show_captions",
        action="store_false",
        help="Disable full caption terminal printing (only show summary lines).",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="30,40,30",
        help="Percentage distribution for [tags, short, dense] captions (default: '30,40,30').",
    )
    parser.add_argument(
        "--tier-mode",
        type=str,
        default="auto",
        choices=["auto", "dense-only", "short-only", "tags-only"],
        help="Enforce a specific tier or use 'auto' to apply the --tiers distribution (default: auto).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def discover_dataset_items(dataset_dir: Path, force: bool = False) -> List[Dict[str, Any]]:
    """Discovers image and .txt pairs in a dataset directory."""
    items = []
    for file_path in sorted(dataset_dir.rglob("*")):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            txt_path = file_path.with_suffix(".txt")
            bak_path = file_path.with_suffix(".txt.bak")

            existing_caption = ""
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        existing_caption = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read {txt_path.name}: {e}")

            # Check if already refined
            is_already_refined = bak_path.exists() and len(existing_caption.split()) >= 220
            if is_already_refined and not force:
                continue

            items.append({
                "image_path": file_path,
                "txt_path": txt_path,
                "bak_path": bak_path,
                "existing_caption": existing_caption,
            })
    return items


def main():
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    logger.info("=" * 70)
    logger.info("  KREA 2 DATASET CAPTION REFINEMENT TOOL")
    logger.info("=" * 70)
    logger.info(f"Target VLM Model       : {args.model}")
    logger.info(f"Sampling Temperature   : {args.temperature}")
    logger.info(f"Generation Token Budget: {args.max_tokens} max tokens")
    logger.info(f"Target Minimum Tokens  : {args.min_tokens} tokens (~280+ words)")
    logger.info(f"Parallel GPU Sub-Batch : {args.parallel_sub_batch}")
    logger.info(f"Batch Chunk Size       : {args.batch_size}")
    logger.info("=" * 70)

    # 1. Discover items to refine
    items: List[Dict[str, Any]] = []

    if args.dataset_name and not args.dataset_dir:
        candidates = [
            Path(f"/app/ai-toolkit/datasets/{args.dataset_name}"),
            PROJECT_ROOT / "data" / "datasets" / args.dataset_name,
            PROJECT_ROOT / "data" / args.dataset_name,
            PROJECT_ROOT / args.dataset_name,
        ]
        for c in candidates:
            if c.exists():
                args.dataset_dir = str(c)
                break
        if not args.dataset_dir:
            args.dataset_dir = f"/app/ai-toolkit/datasets/{args.dataset_name}"

    if not args.trigger:
        if args.dataset_name == "restrained_elegance" or (args.dataset_dir and "restrained_elegance" in str(args.dataset_dir)):
            args.trigger = "restrained_elegance"
            logger.info("Auto-assigned trigger token: 'restrained_elegance'")

    if args.dataset_dir:
        d_path = Path(args.dataset_dir).resolve()
        if not d_path.exists():
            logger.error(f"Dataset directory not found: {d_path}")
            sys.exit(1)
        items = discover_dataset_items(d_path, force=args.force)
        logger.info(f"Discovered {len(items)} image items in {d_path}")
    elif args.manifest:
        m_path = Path(args.manifest).resolve()
        if not m_path.exists():
            logger.error(f"Manifest file not found: {m_path}")
            sys.exit(1)
        from pipeline.manifest import Manifest
        manifest = Manifest(str(m_path))
        for entry in manifest:
            img_p = entry.get_processed_file() or entry.get_raw_file() or Path(entry.processed_path or entry.raw_path or "")
            if img_p and img_p.exists():
                txt_p = img_p.with_suffix(".txt")
                bak_p = img_p.with_suffix(".txt.bak")
                existing_cap = entry.caption_text or ""
                if txt_p.exists() and not existing_cap:
                    try:
                        with open(txt_p, "r", encoding="utf-8") as f:
                            existing_cap = f.read().strip()
                    except Exception:
                        pass
                items.append({
                    "image_path": img_p,
                    "txt_path": txt_p,
                    "bak_p": bak_p,
                    "existing_caption": existing_cap,
                    "manifest_entry": entry,
                })
        logger.info(f"Discovered {len(items)} items from manifest {m_path}")
    else:
        # Default fallback: check common dataset locations
        candidates = [
            Path("/app/ai-toolkit/datasets/restrained_elegance"),
            Path("/app/ai-toolkit/datasets/kink_collection"),
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "data" / "raw",
        ]
        found = None
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                found = c
                break
        if found:
            items = discover_dataset_items(found, force=args.force)
            logger.info(f"Auto-selected dataset directory: {found} ({len(items)} items)")
            if not args.trigger and "restrained_elegance" in str(found):
                args.trigger = "restrained_elegance"
                logger.info("Auto-assigned trigger token: 'restrained_elegance'")
        else:
            logger.error("Please specify --dataset-name <name> or --dataset-dir <path> or --manifest <path>.")
            sys.exit(1)

    if not items:
        logger.info("No items require refinement. Use --force to re-refine all items.")
        return

    # 2. Prepare System Prompt for Pure Caption Output
    vocab = load_vocabulary()
    system_prompt = build_system_prompt(vocab, raw_caption_mode=True)

    # 3. Initialize Engine
    engine = Gemma4HfEngine(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_sub_batch_size=args.parallel_sub_batch,
    )
    engine._ensure_model_loaded()

    # Parse distribution
    try:
        t_parts = [int(p.strip()) for p in args.tiers.split(",")]
        tier_dist = (t_parts[0], t_parts[1], t_parts[2])
    except Exception:
        tier_dist = (30, 40, 30)

    tier_stats = {
        "tags": {"count": 0, "words": 0},
        "short": {"count": 0, "words": 0},
        "dense": {"count": 0, "words": 0},
    }

    total_items = len(items)
    batch_size = args.batch_size
    total_batches = (total_items + batch_size - 1) // batch_size
    logger.info(f"\nStarting refinement across {total_batches} batches...")
    logger.info(f"Tier Distribution : {tier_dist[0]}% Tags | {tier_dist[1]}% Short (max 150 tok) | {tier_dist[2]}% Dense (max 340 tok)")

    refined_count = 0
    failed_count = 0
    total_words_before = 0
    total_words_after = 0

    for b_idx, b_start in enumerate(range(0, total_items, batch_size), start=1):
        batch_items = items[b_start : b_start + batch_size]
        b_end = min(b_start + batch_size, total_items)
        logger.info(f"\n--- Batch {b_idx}/{total_batches} [Items {b_start + 1}-{b_end} of {total_items}] ---")

        image_paths = [it["image_path"] for it in batch_items]
        user_prompts = []

        for idx_in_batch, it in enumerate(batch_items):
            global_idx = b_start + idx_in_batch
            img_name = it["image_path"].name
            existing = it["existing_caption"]
            w_old = len(existing.split()) if existing else 0
            total_words_before += w_old

            # Determine tier
            if args.tier_mode == "tags-only":
                item_tier = "tags"
            elif args.tier_mode == "short-only":
                item_tier = "short"
            elif args.tier_mode == "dense-only":
                item_tier = "dense"
            else:
                item_tier = determine_caption_tier(global_idx, distribution=tier_dist)
            it["assigned_tier"] = item_tier

            # Determine trigger token
            item_trigger = args.trigger
            if not item_trigger:
                if it["existing_caption"]:
                    m_trig = re.match(r"^(kink,\s*[a-zA-Z0-9_\-\.]+)", it["existing_caption"])
                    if m_trig:
                        item_trigger = m_trig.group(1)
                    elif it["existing_caption"].startswith("restrained_elegance"):
                        item_trigger = "restrained_elegance"
                if not item_trigger:
                    item_trigger = "restrained_elegance"
            it["item_trigger"] = item_trigger

            prompt_text = build_tier_prompt(
                tier=item_tier,
                existing_caption=existing,
                trigger_word=item_trigger,
            )
            user_prompts.append(prompt_text)

        # Generate outputs via Gemma 4 parallel engine in raw text mode
        try:
            results = engine.generate_batch(
                image_paths=image_paths,
                user_prompts=user_prompts,
                system_prompt=system_prompt,
                temperature=args.temperature,
                raw_text_mode=True,
            )
        except Exception as exc:
            logger.error(f"Batch {b_idx} generation failed: {exc}")
            results = [None] * len(batch_items)

        # Assemble and write updated captions
        for it, res_data in zip(batch_items, results):
            if res_data is None:
                failed_count += 1
                continue

            # Determine trigger token
            item_trigger = it.get("item_trigger", args.trigger or "restrained_elegance")

            caption_text = assemble_caption(
                res_data,
                trigger_word=item_trigger,
                caption_mode="style",
            )

            # Backup original .txt if requested
            txt_path = it["txt_path"]
            bak_path = it["bak_path"] if "bak_path" in it else it["image_path"].with_suffix(".txt.bak")

            if not args.no_backup and txt_path.exists() and not bak_path.exists():
                try:
                    shutil.copy2(txt_path, bak_path)
                except Exception as e:
                    logger.warning(f"Could not backup {txt_path.name}: {e}")

            # Write updated caption
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption_text + "\n")
            except Exception as e:
                logger.error(f"Could not write {txt_path}: {e}")
                failed_count += 1
                continue

            # Update manifest if entry present
            if "manifest_entry" in it:
                entry = it["manifest_entry"]
                entry.caption_data = res_data
                entry.caption_text = caption_text
                entry.update_stage("stage4_caption", "captioned")

            w_old = len(it["existing_caption"].split()) if it["existing_caption"] else 0
            w_new = len(caption_text.split())
            approx_tokens = int(w_new * 1.35)
            total_words_after += w_new
            refined_count += 1

            tier_name = it.get("assigned_tier", "dense")
            tier_stats[tier_name]["count"] += 1
            tier_stats[tier_name]["words"] += w_new
            tier_label = f"[{tier_name.upper()}]"

            # Append to live monitor log file
            try:
                live_log_path = txt_path.parent / "live_captions.log"
                with open(live_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {it['image_path'].name} {tier_label} | {w_new}w (~{approx_tokens} tokens)\n")
                    lf.write(caption_text + "\n\n" + ("=" * 80) + "\n\n")
            except Exception:
                pass

            if args.show_captions:
                print("\n" + "─" * 78)
                print(f"  ✓ {tier_label} {it['image_path'].name}")
                print(f"    Tier    : {tier_name.capitalize()} (Target: {tier_dist[0]}% tags / {tier_dist[1]}% short / {tier_dist[2]}% dense)")
                print(f"    Metrics : {w_old}w -> {w_new}w (~{approx_tokens} tokens)")
                print(f"    Saved To: {txt_path}")
                print("─" * 78)
                print(caption_text)
                print("─" * 78 + "\n")
            else:
                logger.info(
                    f"  ✓ {tier_label} {it['image_path'].name}: {w_old}w -> {w_new}w (~{approx_tokens} tokens) | {caption_text[:65]}..."
                )

    logger.info("\n" + "=" * 70)
    logger.info("  DATASET CAPTION REFINEMENT COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total Items Processed : {total_items}")
    logger.info(f"Successfully Refined  : {refined_count}")
    logger.info(f"Failed / Skipped      : {failed_count}")
    logger.info("-" * 70)
    logger.info("  STRATIFIED TIER BREAKDOWN:")
    for t_name, s_data in tier_stats.items():
        c = s_data["count"]
        pct = (c / max(1, refined_count)) * 100
        avg_w = s_data["words"] / max(1, c)
        avg_tok = int(avg_w * 1.35)
        logger.info(f"  • {t_name.capitalize():<8} : {c:>5} images ({pct:>5.1f}%) | Avg: {avg_w:>5.1f} words (~{avg_tok:>3} tokens)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
