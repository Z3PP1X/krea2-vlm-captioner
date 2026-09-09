"""Kink Collection End-to-End Orchestrator (Stages 1-5)

Ultra-fast automated pipeline execution for 9 curated dbNaked channels:
1. Fast direct-card candidate extraction (never hangs on slow scene HTML).
2. Crawls up to 400 images per channel (skipping first & last image per scene, max 7 per scene).
3. Sets per-item trigger: 'kink, <category>' (e.g. 'kink, sexandsubmission').
4. Enforces minimum resolution >= 256px.
5. Executes Stage 2 (QC & deduplication with 16 parallel CPU workers).
6. Executes Stage 3 (Downscale to max 2048px).
7. Executes Stage 4 (Offline batch captioning with Gemma 4 12B on RTX PRO 6000 96GB VRAM, screening relaxed).
8. Executes Stage 5 (Exports into single unified dataset 'kink_collection' for AI-Toolkit).
"""

from __future__ import annotations

import os
import re
import io
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from bs4 import BeautifulSoup

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.manifest import Manifest, ManifestEntry
from pipeline.crawler.models import CandidateImage
from pipeline.crawler.dbnaked import (
    DBNAKED_HEADERS,
    build_dbnaked_page_url,
)
from pipeline.logging_utils import setup_logging
from pipeline.stages.stage2_qc import run_stage2
from pipeline.stages.stage3_downscale import run_stage3
from pipeline.stages.stage4_caption import run_stage4
from pipeline.stages.stage5_export import run_stage5
from pipeline.cli import load_config, print_status_table

logger = logging.getLogger("pipeline.kink_orchestrator")

TARGET_CHANNELS = [
    {
        "category": "sexandsubmission",
        "url": "https://dbnaked.com/bdsm/channels/sexandsubmission.com?media=pictures",
        "trigger": "kink, sexandsubmission",
    },
    {
        "category": "hogtied",
        "url": "https://dbnaked.com/bdsm/channels/hogtied.com?media=pictures",
        "trigger": "kink, hogtied",
    },
    {
        "category": "chantasbitches",
        "url": "https://dbnaked.com/bdsm/channels/chantasbitches.com?media=pictures",
        "trigger": "kink, chantasbitches",
    },
    {
        "category": "devicebondage",
        "url": "https://dbnaked.com/bdsm/channels/devicebondage.com?media=pictures",
        "trigger": "kink, devicebondage",
    },
    {
        "category": "electrosluts",
        "url": "https://dbnaked.com/bdsm/channels/electrosluts.com?media=pictures",
        "trigger": "kink, electrosluts",
    },
    {
        "category": "hardtied",
        "url": "https://dbnaked.com/bdsm/channels/hardtied.com?media=pictures",
        "trigger": "kink, hardtied",
    },
    {
        "category": "infernalrestraints",
        "url": "https://dbnaked.com/bdsm/channels/infernalrestraints.com?media=pictures",
        "trigger": "kink, infernalrestraints",
    },
    {
        "category": "sadisticrope",
        "url": "https://dbnaked.com/bdsm/channels/sadisticrope.com?media=pictures",
        "trigger": "kink, sadisticrope",
    },
    {
        "category": "thetrainingofo",
        "url": "https://dbnaked.com/bdsm/channels/thetrainingofo.com?media=pictures",
        "trigger": "kink, thetrainingofo",
    },
]


def extract_candidates_from_channel_page(
    html: str,
    channel_url: str,
    category: str,
    trigger: str,
    max_per_scene: int = 7,
) -> List[CandidateImage]:
    """Extracts candidate images directly from scene cards on the channel page in milliseconds.
    
    Avoids slow scene HTML round-trips by reading scene metadata, image counts, and constructing
    high-res CDN image URLs directly.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_=lambda c: c and "tmb" in c and "card" in c)
    candidates: List[CandidateImage] = []

    for card in cards:
        scene_id = card.get("data-app-modal-data")
        if not scene_id or not scene_id.isdigit():
            continue

        a_link = card.find("a", href=True)
        title = a_link.get("title", f"Scene {scene_id}").strip() if a_link else f"Scene {scene_id}"
        scene_url = urljoin(channel_url, a_link["href"]) if a_link else ""

        img_div = card.find("div", class_="images")
        total_imgs = 0
        if img_div:
            num_txt = img_div.get_text(strip=True)
            if num_txt.isdigit():
                total_imgs = int(num_txt)

        # Rule 1: Skip first (1) and last (total_imgs) image
        if total_imgs <= 2:
            continue

        pool = list(range(2, total_imgs))
        # Rule 2: Max 7 images per scene, evenly spaced across progression
        if len(pool) <= max_per_scene:
            chosen_nums = pool
        else:
            indices = [int(round(i * (len(pool) - 1) / float(max_per_scene - 1))) for i in range(max_per_scene)]
            indices = sorted(list(dict.fromkeys(indices)))
            chosen_nums = [pool[i] for i in indices]

        for num in chosen_nums:
            img_url = f"https://i.dbnaked.com/scene/{scene_id}/t1600x1600/{num}.jpg"
            candidates.append(
                CandidateImage(
                    source="dbnaked",
                    source_url=img_url,
                    page_url=scene_url,
                    context_title=title,
                    category=category,
                    trigger_word=trigger,
                    http_headers=DBNAKED_HEADERS,
                )
            )

    return candidates


def crawl_and_download_channel(
    channel_info: Dict[str, str],
    manifest: Manifest,
    raw_dir: Path,
    session: requests.Session,
    max_channel_images: int = 400,
    max_per_scene: int = 7,
    min_res: int = 256,
    workers: int = 16,
    timeout: float = 15.0,
) -> int:
    """Crawls scenes for a channel until max_channel_images are acquired."""
    category = channel_info["category"]
    channel_url = channel_info["url"]
    trigger = channel_info["trigger"]

    logger.info(f"\n{'='*65}")
    logger.info(f" CRAWLING CHANNEL: {category.upper()} (Target: {max_channel_images} images)")
    logger.info(f" Trigger Token   : '{trigger}'")
    logger.info(f" Channel URL     : {channel_url}")
    logger.info(f"{'='*65}")

    # Check how many images we already have for this category
    existing_count = sum(
        1 for e in manifest
        if e.category == category and e.stages_status.get("stage1_crawl") == "downloaded"
    )
    if existing_count >= max_channel_images:
        logger.info(f"Category '{category}' already has {existing_count} downloaded images (>= {max_channel_images}). Skipping crawl.")
        return existing_count

    selected_candidates: List[CandidateImage] = []
    seen_urls = set(e.source_url for e in manifest)
    needed = max_channel_images - existing_count

    page = 1
    max_channel_pages = 25

    while len(selected_candidates) < needed and page <= max_channel_pages:
        p_url = build_dbnaked_page_url(channel_url, page)
        logger.info(f"[{category}] Inspecting channel page {page}: {p_url}")
        try:
            r = session.get(p_url, headers=DBNAKED_HEADERS, timeout=timeout)
            if r.status_code != 200:
                logger.warning(f"Channel page {page} returned {r.status_code}")
                break
        except Exception as e:
            logger.warning(f"Error fetching channel page {page}: {e}")
            break

        page_cands = extract_candidates_from_channel_page(
            html=r.text,
            channel_url=p_url,
            category=category,
            trigger=trigger,
            max_per_scene=max_per_scene,
        )

        if not page_cands:
            logger.info(f"[{category}] No more candidates found on page {page}.")
            break

        added_this_page = 0
        for c in page_cands:
            if len(selected_candidates) >= needed:
                break
            if c.source_url not in seen_urls:
                seen_urls.add(c.source_url)
                selected_candidates.append(c)
                added_this_page += 1

        logger.info(f"[{category}] Page {page}: added {added_this_page} images. Progress: {len(selected_candidates)}/{needed}")
        page += 1

    logger.info(f"[{category}] Extracted {len(selected_candidates)} total candidates. Starting fast parallel download ({workers} workers)...")

    downloaded_now = 0

    def _download_task(cand: CandidateImage) -> Optional[ManifestEntry]:
        try:
            # Fast check: already in manifest?
            existing = manifest.get_by_url(cand.source_url)
            if existing and existing.stages_status.get("stage1_crawl") == "downloaded":
                return existing

            img_r = session.get(cand.source_url, headers=cand.http_headers, timeout=timeout)
            if img_r.status_code != 200:
                logger.warning(f"Download failed ({img_r.status_code}): {cand.source_url}")
                return None

            img_bytes = img_r.content
            # Verify resolution >= min_res
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                if min(w, h) < min_res:
                    return None

            sha = hashlib.sha256(img_bytes).hexdigest()
            image_id = sha[:16]
            ext = Path(urlparse(cand.source_url).path).suffix.lower() or ".jpg"
            filename = f"{image_id}{ext}"
            file_path = raw_dir / filename

            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(img_bytes)

            rel_raw_path = f"data/raw/{filename}"
            return ManifestEntry(
                image_id=image_id,
                source="dbnaked",
                source_url=cand.source_url,
                page_url=cand.page_url,
                context_title=cand.context_title,
                context_tags=cand.context_tags,
                category=cand.category,
                trigger_word=cand.trigger_word,
                raw_path=rel_raw_path,
                sha256=sha,
                width=w,
                height=h,
                megapixels=round((w * h) / 1_000_000.0, 3),
                aspect_ratio=round(float(w) / max(h, 1), 3),
                stages_status={
                    "stage1_crawl": "downloaded",
                    "stage2_qc": "pending",
                    "stage3_downscale": "pending",
                    "stage4_caption": "pending",
                    "stage5_export": "pending",
                },
            )
        except Exception as err:
            logger.warning(f"Error processing {cand.source_url}: {err}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_task, c) for c in selected_candidates]
        for f in as_completed(futures):
            entry = f.result()
            if entry:
                manifest.add_or_update(entry)
                downloaded_now += 1
                if downloaded_now % 50 == 0 or downloaded_now == len(selected_candidates):
                    logger.info(f"[{category}] Processed {downloaded_now}/{len(selected_candidates)} images ({downloaded_now / len(selected_candidates) * 100:.1f}%)...")

    manifest.save()
    total_channel_now = existing_count + downloaded_now
    logger.info(f"[{category}] Finished: {downloaded_now} newly downloaded, total {total_channel_now} for this channel.")
    return total_channel_now


def main():
    parser = argparse.ArgumentParser(description="Kink Collection Complete Pipeline (Stages 1-5)")
    parser.add_argument("-c", "--config", default="config/pipeline.yaml", help="Path to pipeline configuration")
    parser.add_argument("--max-channel-images", type=int, default=400, help="Max images per channel (default: 400)")
    parser.add_argument("--max-per-scene", type=int, default=7, help="Max images per scene (default: 7)")
    parser.add_argument("--min-res", type=int, default=256, help="Minimum edge resolution (default: 256)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel CPU workers (default: 16)")
    parser.add_argument("--batch-size", type=int, default=32, help="Pipeline batch size (default: 32)")
    parser.add_argument("--parallel-sub-batch", type=int, default=8, help="Parallel GPU tensor batch size for Gemma 4 (default: 8, up to 16-24 for 96GB VRAM)")
    parser.add_argument("--model", type=str, default="google/gemma-4-12B-it", help="Model name (default: google/gemma-4-12B-it)")
    parser.add_argument("--dataset-name", type=str, default="kink_collection", help="Unified dataset name in AI-Toolkit (default: kink_collection)")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip Stage 1 crawl and jump to processing")
    parser.add_argument("--skip-qc", action="store_true", help="Skip Stage 2 QC")
    parser.add_argument("--skip-downscale", action="store_true", help="Skip Stage 3 Downscale")
    parser.add_argument("--skip-caption", action="store_true", help="Skip Stage 4 Caption")
    parser.add_argument("--skip-export", action="store_true", help="Skip Stage 5 Export")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging")

    args = parser.parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    config = load_config(args.config)
    manifest_path = config.get("general", {}).get("manifest_path", "data/manifest.jsonl")
    manifest_file = (PROJECT_ROOT / manifest_path).resolve() if not Path(manifest_path).is_absolute() else Path(manifest_path)
    raw_dir = (PROJECT_ROOT / config.get("general", {}).get("raw_dir", "data/raw")).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(str(manifest_file))

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers * 2, pool_maxsize=args.workers * 4, max_retries=2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    logger.info("=" * 70)
    logger.info("  KINK COLLECTION: 9 CHANNELS AUTOMATED PIPELINE (STAGES 1-5)")
    logger.info("=" * 70)
    logger.info(f"Target GPU Architecture: RTX PRO 6000 (96GB VRAM, 16 vCPUs)")
    logger.info(f"Multimodal Captioner   : {args.model}")
    logger.info(f"Batch Size             : {args.batch_size}")
    logger.info(f"Min Resolution         : {args.min_res} px")
    logger.info(f"Max Per Channel        : {args.max_channel_images} images")
    logger.info(f"Max Per Scene          : {args.max_per_scene} images (excluding first & last)")
    logger.info(f"Output Dataset Name    : {args.dataset_name}")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STAGE 1: CRAWL 9 CHANNELS
    # --------------------------------------------------------------------------
    if not args.skip_crawl:
        logger.info("\n>>> STARTING STAGE 1: CRAWLING 9 CHANNELS <<<")
        for ch in TARGET_CHANNELS:
            crawl_and_download_channel(
                channel_info=ch,
                manifest=manifest,
                raw_dir=raw_dir,
                session=session,
                max_channel_images=args.max_channel_images,
                max_per_scene=args.max_per_scene,
                min_res=args.min_res,
                workers=args.workers,
            )
        manifest.save()
        logger.info("Stage 1 Crawl Finished. Current Status:")
        print_status_table(manifest.summary())
    else:
        logger.info("Skipping Stage 1 (--skip-crawl set).")

    # --------------------------------------------------------------------------
    # STAGE 2: QUALITY CONTROL & DEDUPLICATION
    # --------------------------------------------------------------------------
    if not args.skip_qc:
        logger.info("\n>>> STARTING STAGE 2: QUALITY CONTROL & DEDUPLICATION <<<")
        args_qc = argparse.Namespace(
            max_workers=args.workers,
            min_edge=args.min_res,
            force=False,
        )
        # Configure lenient QC so low-res down to 256px passes
        config.setdefault("stage2_qc", {})
        config["stage2_qc"]["min_edge"] = args.min_res
        config["stage2_qc"]["laplacian_variance_min"] = 15.0
        config["stage2_qc"]["detect_watermarks"] = False
        config["stage2_qc"]["detect_text"] = False
        run_stage2(args_qc, config)
    else:
        logger.info("Skipping Stage 2 (--skip-qc set).")

    # --------------------------------------------------------------------------
    # STAGE 3: DOWNSCALING
    # --------------------------------------------------------------------------
    if not args.skip_downscale:
        logger.info("\n>>> STARTING STAGE 3: DOWNSCALING (MAX 2048px) <<<")
        args_downscale = argparse.Namespace(
            max_workers=args.workers,
            max_dim=2048,
        )
        run_stage3(args_downscale, config)
    else:
        logger.info("Skipping Stage 3 (--skip-downscale set).")

    # --------------------------------------------------------------------------
    # STAGE 4: CAPTIONING WITH GEMMA 4 12B
    # --------------------------------------------------------------------------
    if not args.skip_caption:
        logger.info(f"\n>>> STARTING STAGE 4: GEMMA 4 12B MULTIMODAL CAPTIONING <<<")
        args_caption = argparse.Namespace(
            model=args.model,
            batch_size=args.batch_size,
            parallel_sub_batch=args.parallel_sub_batch,
            trigger=None,          # Uses per-item trigger_word: 'kink, <category>'!
            mode="style",
            max_model_len=8192,
            ignore_watermarks=True,
            no_screening=True,     # Do not discard on screening gates
            force=False,
            sample=None,
            retry_failed=True,
        )
        config.setdefault("stage4_caption", {})
        config["stage4_caption"]["vllm_gpu_memory_utilization"] = 0.90
        config["stage4_caption"]["max_model_len"] = 8192
        run_stage4(args_caption, config)
    else:
        logger.info("Skipping Stage 4 (--skip-caption set).")

    # --------------------------------------------------------------------------
    # STAGE 5: EXPORT TO AI-TOOLKIT
    # --------------------------------------------------------------------------
    if not args.skip_export:
        logger.info(f"\n>>> STARTING STAGE 5: EXPORTING UNIFIED DATASET '{args.dataset_name}' <<<")
        args_export = argparse.Namespace(
            dataset_name=args.dataset_name,
            ai_toolkit_dir="/app/ai-toolkit",
        )
        run_stage5(args_export, config)
    else:
        logger.info("Skipping Stage 5 (--skip-export set).")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("  KINK COLLECTION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    print_status_table(manifest.summary())


if __name__ == "__main__":
    main()
