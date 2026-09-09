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
)
from pipeline.captioning.assembly import assemble_caption
from pipeline.captioning.hf_gemma4_engine import Gemma4HfEngine

logger = logging.getLogger("dataset.caption_refiner")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and elevate existing dataset captions to 400+ token Krea 2 visual narratives."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to folder containing images and .txt caption pairs.",
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
        default=750,
        help="Maximum generation token budget (default: 750).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=400,
        help="Target minimum caption tokens (default: 400).",
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
        # Default fallback: check data/processed or data/raw
        default_dir = PROJECT_ROOT / "data" / "processed"
        if not default_dir.exists() or not any(default_dir.iterdir()):
            default_dir = PROJECT_ROOT / "data" / "raw"
        if default_dir.exists():
            items = discover_dataset_items(default_dir, force=args.force)
            logger.info(f"Auto-selected default dataset directory: {default_dir} ({len(items)} items)")
        else:
            logger.error("Please specify --dataset-dir <path> or --manifest <path>.")
            sys.exit(1)

    if not items:
        logger.info("No items require refinement. Use --force to re-refine all items.")
        return

    # 2. Prepare Schema and System Prompt
    vocab = load_vocabulary()
    json_schema = get_vllm_json_schema(vocab)
    system_prompt = build_system_prompt(vocab)

    # 3. Initialize Engine
    engine = Gemma4HfEngine(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_sub_batch_size=args.parallel_sub_batch,
    )
    engine._ensure_model_loaded()

    total_items = len(items)
    batch_size = args.batch_size
    total_batches = (total_items + batch_size - 1) // batch_size
    logger.info(f"\nStarting refinement across {total_batches} batches...")

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

        for it in batch_items:
            img_name = it["image_path"].name
            existing = it["existing_caption"]
            w_old = len(existing.split()) if existing else 0
            total_words_before += w_old

            if existing:
                prompt_text = (
                    f"Existing Draft Caption:\n\"{existing}\"\n\n"
                    f"Refinement Task:\n"
                    f"1. Audit the draft caption against this image. Fix any hallucinations or inaccurate rope/hardware terms.\n"
                    f"2. EXPAND the caption into an exhaustive 7-layer Krea 2 visual narrative of AT LEAST 400 TOKENS (~280-350+ words in fluent English).\n"
                    f"3. Fully describe: (1) Medium/framing, (2) Subject anatomy/skin texture, (3) Pose mechanics/tension, "
                    f"(4) Exact shibari knots (takate-kote, hishime, kikko) or metallic hardware (chrome handcuffs, O-rings, carabiners), "
                    f"(5) Studio environment/flooring, (6) Chiaroscuro lighting/specular highlights, and (7) Camera optics/DoF.\n"
                    f"Output strictly conforming to the JSON schema."
                )
            else:
                prompt_text = (
                    f"Task:\n"
                    f"Inspect this image and generate an exhaustive 7-layer Krea 2 visual narrative of AT LEAST 400 TOKENS (~280-350+ words).\n"
                    f"Meticulously detail subject anatomy, body tension, exact shibari knots or bondage hardware, tactile textures, "
                    f"studio environment, lighting gradients, and camera optics.\n"
                    f"Output strictly conforming to the JSON schema."
                )
            user_prompts.append(prompt_text)

        # Generate outputs via Gemma 4 parallel engine
        try:
            results = engine.generate_batch(
                image_paths=image_paths,
                user_prompts=user_prompts,
                system_prompt=system_prompt,
                json_schema=json_schema,
                temperature=args.temperature,
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
            item_trigger = args.trigger
            if not item_trigger:
                # Try to extract trigger token from existing caption (e.g. 'kink, category')
                if it["existing_caption"]:
                    m_trig = re.match(r"^(kink,\s*[a-zA-Z0-9_\-\.]+)", it["existing_caption"])
                    if m_trig:
                        item_trigger = m_trig.group(1)
                    elif it["existing_caption"].startswith("restrained_elegance"):
                        item_trigger = "restrained_elegance"

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

            logger.info(
                f"  ✓ [REFINED] {it['image_path'].name}: {w_old}w -> {w_new}w (~{approx_tokens} tokens) | {caption_text[:65]}..."
            )

    logger.info("\n" + "=" * 70)
    logger.info("  DATASET CAPTION REFINEMENT COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total Items Processed : {total_items}")
    logger.info(f"Successfully Refined  : {refined_count}")
    logger.info(f"Failed / Skipped      : {failed_count}")
    if refined_count > 0:
        avg_before = total_words_before / max(1, refined_count)
        avg_after = total_words_after / max(1, refined_count)
        logger.info(f"Average Words Before  : {avg_before:.1f} words (~{int(avg_before * 1.35)} tokens)")
        logger.info(f"Average Words After   : {avg_after:.1f} words (~{int(avg_after * 1.35)} tokens)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
