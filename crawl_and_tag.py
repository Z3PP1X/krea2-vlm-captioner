#!/usr/bin/env python3
"""
Unified Crawl & Tag Pipeline for Krea 2
========================================
Automates the full end-to-end workflow:
  1. Crawls image sets from XenForo threads.
  2. Organizes each image set into its own directory with 'genres.txt'.
  3. Feeds each image + its specific 'genres.txt' context + your trigger word into Gemma / VLM.
  4. Generates a paired '<image>.txt' containing the pure 7-layer Krea 2 caption.

Usage Examples:
  python crawl_and_tag.py --pages 27 -t "restrained_elegance"
  python crawl_and_tag.py --pages 11-13 -t "shibori" -m "gemma4:32b" -o ./downloads
  python crawl_and_tag.py --output-dir ./downloads -t "caning" --skip-crawl  # Tag existing sets
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

# Import crawler and captioner modules
from crawler import create_crawler, parse_page_range, DEFAULT_URL
from caption_images import caption_directory, setup_logging

logger = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Crawl & Tag Pipeline: Scrape XenForo/dbNaked image sets and auto-caption them for Krea 2."
    )

    # Crawl Arguments
    crawl_group = parser.add_argument_group("Crawler Options")
    crawl_group.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"Target URL to crawl (XenForo thread or dbNaked channel/scene, default: {DEFAULT_URL})",
    )
    crawl_group.add_argument(
        "-p", "--pages",
        type=str,
        default="27",
        help="Pages to crawl: e.g. '27', '25-27', '11-15', or 'all'",
    )
    crawl_group.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./downloads",
        help="Base directory to store downloaded image sets (default: ./downloads)",
    )
    crawl_group.add_argument(
        "--crawl-workers",
        type=int,
        default=4,
        help="Concurrent download worker threads for images (default: 4)",
    )
    crawl_group.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite delay in seconds between page crawls (default: 1.0)",
    )
    crawl_group.add_argument(
        "--by-page",
        action="store_true",
        help="Group set folders inside page subdirectories (e.g. downloads/page_027/Set/)",
    )
    crawl_group.add_argument(
        "--min-size",
        type=int,
        default=15360,
        help="Minimum file size in bytes to filter out banners/icons (default: 15KB)",
    )
    crawl_group.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop crawling after downloading N images",
    )
    crawl_group.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip the crawling stage and only run captioning on existing sets in --output-dir",
    )

    # Captioner Arguments
    caption_group = parser.add_argument_group("Captioner Options")
    caption_group.add_argument(
        "-t", "--trigger",
        type=str,
        default=None,
        help="Trigger token to prepend (e.g. 'restrained_elegance', 'shibori')",
    )
    caption_group.add_argument(
        "--tier",
        type=str,
        choices=["dense", "mid", "short"],
        default="dense",
        help="Which caption tier to save into .txt: 'dense' (7-layer narrative, default), 'mid', or 'short'",
    )
    caption_group.add_argument(
        "-m", "--model",
        type=str,
        default=os.environ.get("VISION_MODEL", "gemma4:32b"),
        help="VLM model identifier (default: 'gemma4:32b' or env $VISION_MODEL)",
    )
    caption_group.add_argument(
        "-u", "--url-vlm",
        type=str,
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Inference base URL (default: 'http://localhost:11434' or env $OLLAMA_URL)",
    )
    caption_group.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help="API backend protocol: 'ollama' (default) or 'openai' (for vLLM / SGLang on RunPod)",
    )
    caption_group.add_argument(
        "--caption-workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads for VLM captioning (default: 1 for local Ollama, 2-8 for vLLM)",
    )
    caption_group.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum image edge size in pixels before downscaling for VLM (default: 1024)",
    )
    caption_group.add_argument(
        "--save-json",
        action="store_true",
        help="Also write full JSON metadata (.json) alongside the .txt file",
    )
    caption_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files instead of skipping them",
    )
    caption_group.add_argument(
        "--skip-caption",
        action="store_true",
        help="Skip the captioning stage and only perform crawling",
    )
    caption_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir).resolve()

    logger.info("============================================================")
    logger.info("          UNIFIED CRAWL & TAG PIPELINE (KREA 2)             ")
    logger.info("============================================================")
    logger.info(f"Target Output Dir : {output_dir}")
    logger.info(f"Trigger Token     : {args.trigger or '(None)'}")
    logger.info(f"VLM Model         : {args.model}")
    logger.info(f"VLM Endpoint      : {args.backend} @ {args.url_vlm}")
    logger.info("============================================================")

    # -------------------------------------------------------------------------
    # STAGE 1: CRAWL
    # -------------------------------------------------------------------------
    if not args.skip_crawl:
        logger.info("\n>>> STAGE 1: Starting Image Crawler...")
        try:
            if "dbnaked.com" in args.url.lower():
                from dbnaked_crawler import DbNakedCrawler, parse_page_range as parse_db_pages
                crawler = DbNakedCrawler(
                    channel_url=args.url,
                    output_dir=str(output_dir),
                    workers=args.crawl_workers,
                    delay=args.delay,
                    max_images=args.max_images,
                    overwrite=args.overwrite,
                    dry_run=False,
                )
                page_numbers = parse_db_pages(args.pages)
                logger.info(f"Detected dbNaked Channel URL: {args.url}")
                logger.info(f"Pages to crawl: {page_numbers}")
                total_downloaded = crawler.crawl(page_numbers)
            else:
                crawler = XenForoImageCrawler(
                    thread_url=args.url,
                    output_dir=str(output_dir),
                    by_set=True,
                    by_page=args.by_page,
                    naming="original",
                    workers=args.crawl_workers,
                    delay=args.delay,
                    min_size=args.min_size,
                    max_images=args.max_images,
                    overwrite=False,
                    dry_run=False,
                )
                total_pages = crawler.fetch_thread_info()
                page_numbers = parse_page_range(args.pages, max_available=total_pages)
                logger.info(f"Thread Title Detected: {crawler.thread_tags or 'N/A'}")
                logger.info(f"Pages to crawl: {page_numbers}")

                total_downloaded = 0
                for p in page_numbers:
                    items = crawler.crawl_page(p)
                    if not items:
                        continue
                    cnt = crawler.download_items(items)
                    total_downloaded += cnt

            logger.info(f"\n[+] Crawling completed. Total images saved: {total_downloaded}")
        except Exception as exc:
            logger.error(f"Crawler encountered an error: {exc}", exc_info=args.verbose)
            sys.exit(1)
    else:
        logger.info("\n[*] Skipping Stage 1 (--skip-crawl requested). Using existing files.")

    # -------------------------------------------------------------------------
    # STAGE 2: VLM CAPTIONING
    # -------------------------------------------------------------------------
    if not args.skip_caption:
        logger.info("\n>>> STAGE 2: Starting Krea 2 VLM Captioner...")
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            sys.exit(1)

        # Build captioner arguments namespace
        caption_args = argparse.Namespace(
            input_dir=str(output_dir),
            trigger=args.trigger,
            tier=args.tier,
            model=args.model,
            url=args.url_vlm,
            backend=args.backend,
            workers=args.caption_workers,
            max_size=args.max_size,
            timeout=180,
            recursive=True,
            save_json=args.save_json,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        success, skipped, errors = caption_directory(output_dir, caption_args)

        logger.info("\n============================================================")
        logger.info("              PIPELINE EXECUTION SUMMARY                    ")
        logger.info("============================================================")
        logger.info(f"Target Directory     : {output_dir}")
        logger.info(f"Successfully Tagged  : {success}")
        logger.info(f"Skipped (Pre-existing): {skipped}")
        logger.info(f"Errors               : {errors}")
        logger.info("============================================================")
    else:
        logger.info("\n[*] Skipping Stage 2 (--skip-caption requested).")


if __name__ == "__main__":
    main()
