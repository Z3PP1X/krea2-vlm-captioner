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
from typing import List, Dict, Any, Optional, Tuple

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
        help="Re-refine all captions even if a .txt.bak backup exists or token count > threshold.",
    )
    parser.add_argument(
        "--skip-token-threshold",
        type=int,
        default=400,
        help="Token length threshold to detect already-updated captions (default: 400). Captions with tokens > threshold will be excluded from the pipeline.",
    )
    parser.add_argument(
        "--skip-updated",
        dest="skip_updated",
        action="store_true",
        default=True,
        help="Exclude already-updated captions from refinement (default: True).",
    )
    parser.add_argument(
        "--no-skip-updated",
        dest="skip_updated",
        action="store_false",
        help="Do not skip already-updated captions.",
    )
    parser.add_argument(
        "--dedup-by-hash",
        dest="dedup_by_hash",
        action="store_true",
        default=True,
        help="Deduplicate images sharing the same hash before .jpg, send only primary to VLM, and sync duplicates in parallel (default: True).",
    )
    parser.add_argument(
        "--no-dedup-by-hash",
        dest="dedup_by_hash",
        action="store_false",
        help="Disable hash-based deduplication.",
    )
    parser.add_argument(
        "--sync-duplicates",
        dest="sync_duplicates",
        action="store_true",
        default=True,
        help="Synchronize captions to duplicate images in parallel (default: True).",
    )
    parser.add_argument(
        "--no-sync-duplicates",
        dest="sync_duplicates",
        action="store_false",
        help="Disable automatic caption synchronization to duplicates.",
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


def estimate_token_count(text: str) -> int:
    """Estimates token count for a caption text string."""
    if not text:
        return 0
    words = text.split()
    word_tokens = int(len(words) * 1.35)
    regex_tokens = len(re.findall(r"\w+|[^\w\s]", text))
    return max(word_tokens, regex_tokens)


def is_caption_already_updated(
    caption: str,
    bak_path: Optional[Path] = None,
    token_threshold: int = 400,
) -> bool:
    """
    Checks whether a caption has already been updated in a previous run.
    Criteria:
      1. .txt.bak exists (created when previous script updated caption).
      2. Token count > token_threshold (default 400; old Qwen used max 300 tokens).
      3. Word count >= 280 words (~400+ tokens).
    """
    if not caption or not caption.strip():
        return False
    if bak_path and bak_path.exists():
        return True

    words = caption.split()
    tokens = estimate_token_count(caption)
    if tokens > token_threshold or len(words) >= 280:
        return True
    return False


def extract_hash_key(file_path: Path) -> str:
    """
    Extracts the image hash / base identifier before the image extension.
    Handles:
      - 068e12987d7b5ee9.jpg -> '068e12987d7b5ee9'
      - 068e12987d7b5ee9.jpg.jpg -> '068e12987d7b5ee9'
      - .thumbs/068e12987d7b5ee9.jpg.jpg -> '068e12987d7b5ee9'
      - 068e12987d7b5ee9_1.jpg -> '068e12987d7b5ee9'
      - 068e12987d7b5ee9 (1).jpg -> '068e12987d7b5ee9'
      - image_001.jpg -> 'image_001'
    """
    name = file_path.name
    m = re.match(r"^([0-9a-fA-F]{8,64})", name)
    if m:
        return m.group(1).lower()
    parts = re.split(r"\.(jpe?g|png|webp|bmp)", name, flags=re.IGNORECASE)
    if parts and parts[0]:
        return parts[0].strip().lower()
    return file_path.stem.lower()


def rank_primary_candidate(item: Dict[str, Any]) -> tuple:
    """
    Ranks candidates sharing the same hash so the true primary image is chosen.
    Priority:
      1. Not in a hidden directory (e.g. not in .thumbs, not starting with '.')
      2. Not double-extended (e.g. .jpg rather than .jpg.jpg)
      3. Has longer existing caption
      4. Shorter filename
      5. Shorter full path
    """
    p = item["image_path"]
    is_hidden_dir = any(part.startswith(".") for part in p.parent.parts)
    is_double_ext = bool(re.search(r"\.(jpe?g|png|webp|bmp)\.(jpe?g|png|webp|bmp)$", p.name, re.I))
    cap = item.get("existing_caption", "").strip()
    cap_len = len(cap.split())

    return (
        1 if is_hidden_dir else 0,
        1 if is_double_ext else 0,
        -cap_len,
        len(p.name),
        len(str(p)),
    )


def sync_caption_to_duplicates(
    primary_item: Dict[str, Any],
    caption_text: str,
    backup: bool = True,
) -> List[str]:
    """
    Synchronizes caption_text in parallel to all duplicate image files associated with primary_item.
    Returns list of duplicate file names that were updated.
    """
    synced_names = []
    duplicates = primary_item.get("duplicates", [])
    if not duplicates:
        return synced_names

    for dup in duplicates:
        dup_img = dup["image_path"]
        txt_path = dup["txt_path"]
        bak_path = dup.get("bak_path") or dup_img.with_suffix(".txt.bak")

        target_txts = [txt_path]
        hash_key = dup.get("hash_key")
        if hash_key:
            alt_txt = dup_img.parent / f"{hash_key}.txt"
            if alt_txt not in target_txts:
                target_txts.append(alt_txt)

        for target_txt in target_txts:
            try:
                target_txt.parent.mkdir(parents=True, exist_ok=True)
                if backup and target_txt.exists() and not bak_path.exists():
                    try:
                        shutil.copy2(target_txt, bak_path)
                    except Exception:
                        pass
                with open(target_txt, "w", encoding="utf-8") as f:
                    f.write(caption_text.strip() + "\n")
            except Exception as e:
                logger.warning(f"Could not sync duplicate caption to {target_txt}: {e}")

        dup["existing_caption"] = caption_text.strip()
        synced_names.append(dup_img.name)

    return synced_names


def discover_dataset_items(
    dataset_dir: Path,
    force: bool = False,
    skip_updated: bool = True,
    skip_token_threshold: int = 400,
    dedup_by_hash: bool = True,
    sync_duplicates: bool = True,
    no_backup: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Discovers image and .txt pairs in a dataset directory with hash deduplication
    and updated-caption exclusion.
    """
    discovery_stats = {
        "total_files": 0,
        "unique_hashes": 0,
        "duplicates_detected": 0,
        "already_updated_skipped": 0,
        "duplicates_synced_existing": 0,
        "queued_for_vlm": 0,
    }

    raw_items = []
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

            hash_key = extract_hash_key(file_path) if dedup_by_hash else file_path.name
            raw_items.append({
                "image_path": file_path,
                "txt_path": txt_path,
                "bak_path": bak_path,
                "existing_caption": existing_caption,
                "hash_key": hash_key,
                "duplicates": [],
            })

    discovery_stats["total_files"] = len(raw_items)

    if not dedup_by_hash:
        items_to_process = []
        for it in raw_items:
            is_upd = skip_updated and is_caption_already_updated(
                it["existing_caption"], it["bak_path"], token_threshold=skip_token_threshold
            )
            if is_upd and not force:
                discovery_stats["already_updated_skipped"] += 1
                continue
            items_to_process.append(it)
        discovery_stats["queued_for_vlm"] = len(items_to_process)
        return items_to_process, discovery_stats

    # Group by hash key
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for it in raw_items:
        groups.setdefault(it["hash_key"], []).append(it)

    discovery_stats["unique_hashes"] = len(groups)
    items_to_process = []

    for hash_key, group_items in groups.items():
        # Rank so the true original image is primary
        group_items.sort(key=rank_primary_candidate)
        primary = group_items[0]
        duplicates = group_items[1:]
        primary["duplicates"] = duplicates
        discovery_stats["duplicates_detected"] += len(duplicates)

        # Check if ANY candidate in this group has an already-updated caption
        best_updated_caption = None
        for it in group_items:
            cap = it["existing_caption"]
            bak = it["bak_path"]
            if is_caption_already_updated(cap, bak, token_threshold=skip_token_threshold):
                if best_updated_caption is None or len(cap.split()) > len(best_updated_caption.split()):
                    best_updated_caption = cap

        group_is_updated = (best_updated_caption is not None)

        if group_is_updated and skip_updated and not force:
            discovery_stats["already_updated_skipped"] += 1
            primary["existing_caption"] = best_updated_caption

            # Ensure primary has this updated caption in its .txt
            if sync_duplicates:
                p_cap = ""
                if primary["txt_path"].exists():
                    try:
                        with open(primary["txt_path"], "r", encoding="utf-8") as pf:
                            p_cap = pf.read().strip()
                    except Exception:
                        pass

                if p_cap != best_updated_caption:
                    try:
                        primary["txt_path"].parent.mkdir(parents=True, exist_ok=True)
                        with open(primary["txt_path"], "w", encoding="utf-8") as pf:
                            pf.write(best_updated_caption + "\n")
                    except Exception as e:
                        logger.warning(f"Could not sync caption to primary {primary['txt_path']}: {e}")

                # Sync to all duplicates for this already-updated item
                synced = sync_caption_to_duplicates(
                    primary, best_updated_caption, backup=not no_backup
                )
                discovery_stats["duplicates_synced_existing"] += len(synced)

            continue  # Exclude from VLM queue!

        items_to_process.append(primary)

    discovery_stats["queued_for_vlm"] = len(items_to_process)
    return items_to_process, discovery_stats


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
    discovery_stats: Dict[str, int] = {}

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
        items, discovery_stats = discover_dataset_items(
            d_path,
            force=args.force,
            skip_updated=args.skip_updated,
            skip_token_threshold=args.skip_token_threshold,
            dedup_by_hash=args.dedup_by_hash,
            sync_duplicates=args.sync_duplicates,
            no_backup=args.no_backup,
        )
    elif args.manifest:
        m_path = Path(args.manifest).resolve()
        if not m_path.exists():
            logger.error(f"Manifest file not found: {m_path}")
            sys.exit(1)
        from pipeline.manifest import Manifest
        manifest = Manifest(str(m_path))
        raw_manifest_items = []
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
                hash_k = extract_hash_key(img_p) if args.dedup_by_hash else img_p.name
                raw_manifest_items.append({
                    "image_path": img_p,
                    "txt_path": txt_p,
                    "bak_path": bak_p,
                    "existing_caption": existing_cap,
                    "manifest_entry": entry,
                    "hash_key": hash_k,
                    "duplicates": [],
                })

        discovery_stats = {
            "total_files": len(raw_manifest_items),
            "unique_hashes": len(raw_manifest_items),
            "duplicates_detected": 0,
            "already_updated_skipped": 0,
            "duplicates_synced_existing": 0,
            "queued_for_vlm": 0,
        }

        if args.dedup_by_hash:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in raw_manifest_items:
                groups.setdefault(it["hash_key"], []).append(it)
            discovery_stats["unique_hashes"] = len(groups)

            for hash_k, g_items in groups.items():
                g_items.sort(key=rank_primary_candidate)
                primary = g_items[0]
                dups = g_items[1:]
                primary["duplicates"] = dups
                discovery_stats["duplicates_detected"] += len(dups)

                best_upd = None
                for it in g_items:
                    cap = it["existing_caption"]
                    bak = it["bak_path"]
                    if is_caption_already_updated(cap, bak, token_threshold=args.skip_token_threshold):
                        if best_upd is None or len(cap.split()) > len(best_upd.split()):
                            best_upd = cap

                if best_upd and args.skip_updated and not args.force:
                    discovery_stats["already_updated_skipped"] += 1
                    primary["existing_caption"] = best_upd
                    if args.sync_duplicates:
                        synced = sync_caption_to_duplicates(primary, best_upd, backup=not args.no_backup)
                        discovery_stats["duplicates_synced_existing"] += len(synced)
                    continue
                items.append(primary)
        else:
            for it in raw_manifest_items:
                if args.skip_updated and not args.force and is_caption_already_updated(it["existing_caption"], it["bak_path"], token_threshold=args.skip_token_threshold):
                    discovery_stats["already_updated_skipped"] += 1
                    continue
                items.append(it)
        discovery_stats["queued_for_vlm"] = len(items)
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
            items, discovery_stats = discover_dataset_items(
                found,
                force=args.force,
                skip_updated=args.skip_updated,
                skip_token_threshold=args.skip_token_threshold,
                dedup_by_hash=args.dedup_by_hash,
                sync_duplicates=args.sync_duplicates,
                no_backup=args.no_backup,
            )
            logger.info(f"Auto-selected dataset directory: {found}")
            if not args.trigger and "restrained_elegance" in str(found):
                args.trigger = "restrained_elegance"
                logger.info("Auto-assigned trigger token: 'restrained_elegance'")
        else:
            logger.error("Please specify --dataset-name <name> or --dataset-dir <path> or --manifest <path>.")
            sys.exit(1)

    if discovery_stats:
        logger.info("=" * 70)
        logger.info("  DISCOVERY & DEDUPLICATION REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Image Files Discovered  : {discovery_stats['total_files']}")
        logger.info(f"Unique Image Hashes           : {discovery_stats['unique_hashes']}")
        logger.info(f"Duplicate Images Detected     : {discovery_stats['duplicates_detected']} (excluded from VLM queue)")
        logger.info(f"Already-Updated Captions      : {discovery_stats['already_updated_skipped']} (> {args.skip_token_threshold} tokens or .bak) -> SKIPPED")
        if discovery_stats['duplicates_synced_existing'] > 0:
            logger.info(f"Duplicates Synced from Prev   : {discovery_stats['duplicates_synced_existing']} updated in parallel")
        logger.info(f"Primary Images Queued for VLM : {len(items)}")
        logger.info("=" * 70 + "\n")

    if not items:
        logger.info("No items require refinement. All captions are up to date! Use --force to re-refine all items.")
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
    parallel_synced_duplicates_count = 0
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

            # Synchronize duplicates in parallel
            synced_dups = []
            if args.sync_duplicates and it.get("duplicates"):
                synced_dups = sync_caption_to_duplicates(
                    it, caption_text, backup=not args.no_backup
                )
                parallel_synced_duplicates_count += len(synced_dups)

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
                    if synced_dups:
                        lf.write(f"  ↳ Synced {len(synced_dups)} duplicate(s) in parallel: {', '.join(synced_dups)}\n")
                    lf.write(caption_text + "\n\n" + ("=" * 80) + "\n\n")
            except Exception:
                pass

            if args.show_captions:
                print("\n" + "─" * 78)
                print(f"  ✓ {tier_label} {it['image_path'].name}")
                print(f"    Tier    : {tier_name.capitalize()} (Target: {tier_dist[0]}% tags / {tier_dist[1]}% short / {tier_dist[2]}% dense)")
                print(f"    Metrics : {w_old}w -> {w_new}w (~{approx_tokens} tokens)")
                print(f"    Saved To: {txt_path}")
                if synced_dups:
                    print(f"    ↳ Synced {len(synced_dups)} duplicate(s) in parallel: {', '.join(synced_dups[:4])}{'...' if len(synced_dups) > 4 else ''}")
                print("─" * 78)
                print(caption_text)
                print("─" * 78 + "\n")
            else:
                dup_str = f" (+{len(synced_dups)} dups synced)" if synced_dups else ""
                logger.info(
                    f"  ✓ {tier_label} {it['image_path'].name}{dup_str}: {w_old}w -> {w_new}w (~{approx_tokens} tokens) | {caption_text[:65]}..."
                )

    logger.info("\n" + "=" * 70)
    logger.info("  DATASET CAPTION REFINEMENT COMPLETED")
    logger.info("=" * 70)
    if discovery_stats:
        logger.info(f"Total Discovered Files        : {discovery_stats.get('total_files', 0)}")
        logger.info(f"Unique Image Hashes           : {discovery_stats.get('unique_hashes', 0)}")
        logger.info(f"Duplicates Detected           : {discovery_stats.get('duplicates_detected', 0)}")
        logger.info(f"Already-Updated (Skipped)     : {discovery_stats.get('already_updated_skipped', 0)}")
        if discovery_stats.get('duplicates_synced_existing', 0) > 0:
            logger.info(f"Duplicates Synced from Prev   : {discovery_stats.get('duplicates_synced_existing', 0)}")
    logger.info(f"Primary Images Refined (VLM)  : {refined_count}")
    logger.info(f"Duplicates Synced in Parallel : {parallel_synced_duplicates_count}")
    logger.info(f"Failed / Errors               : {failed_count}")
    logger.info("-" * 70)
    logger.info("  STRATIFIED TIER BREAKDOWN (Refined Primaries):")
    for t_name, s_data in tier_stats.items():
        c = s_data["count"]
        pct = (c / max(1, refined_count)) * 100
        avg_w = s_data["words"] / max(1, c)
        avg_tok = int(avg_w * 1.35)
        logger.info(f"  • {t_name.capitalize():<8} : {c:>5} images ({pct:>5.1f}%) | Avg: {avg_w:>5.1f} words (~{avg_tok:>3} tokens)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
