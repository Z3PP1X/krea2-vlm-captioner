"""Stufe 1 – Crawling mit Vorfilter, Compliance-Prüfung und Manifest-Registrierung."""

from __future__ import annotations

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, unquote
import requests
from PIL import Image

from pipeline.manifest import Manifest, ManifestEntry
from pipeline.crawler.models import CandidateImage
from pipeline.crawler.prefilter import (
    fetch_remote_resolution,
    is_resolution_acceptable,
)
from pipeline.crawler.compliance import ComplianceChecker
from pipeline.crawler.rate_limiter import DomainRateLimiter
from pipeline.crawler.xenforo import (
    parse_page_range,
    build_page_url,
    extract_xenforo_candidates,
)
from pipeline.crawler.dbnaked import (
    is_dbnaked_channel,
    extract_scene_links_from_channel,
    extract_dbnaked_gallery_candidates,
    DBNAKED_HEADERS,
)

logger = logging.getLogger("pipeline.stage1_crawl")


def sanitize_name(name: str, max_length: int = 80) -> str:
    """Sanitizes folder/file name for safe cross-platform saving."""
    name = unquote(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(". ")
    return name[:max_length] if name else "item"


def run_stage1(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 1: Crawling with Pre-filter."""
    general_cfg = config.get("general", {})
    crawl_cfg = config.get("stage1_crawl", {})

    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    raw_dir = Path(general_cfg.get("raw_dir", "data/raw"))
    min_mp = float(crawl_cfg.get("min_resolution_mp", 0.7))
    user_agent = crawl_cfg.get("user_agent", "Krea2LoRAPipelineBot/1.0")
    respect_robots = crawl_cfg.get("respect_robots_txt", True)
    filter_tdm = crawl_cfg.get("filter_tdm_opt_out", True)
    rate_limit_delay = float(crawl_cfg.get("default_rate_limit_delay", 1.0))
    timeout = float(crawl_cfg.get("request_timeout", 25))

    target_url = getattr(args, "url", None) or "https://xxx-files.org/threads/hard-tied-photocollection.20611/page-27"
    pages_arg = getattr(args, "pages", None) or "27"

    logger.info("=" * 60)
    logger.info("  STUFE 1: CRAWLING MIT VORFILTER & COMPLIANCE")
    logger.info("=" * 60)
    logger.info(f"Target URL         : {target_url}")
    logger.info(f"Pages              : {pages_arg}")
    logger.info(f"Min Resolution MP  : {min_mp} MP")
    logger.info(f"Robots.txt Check   : {respect_robots}")
    logger.info(f"TDM Opt-out Filter : {filter_tdm} (§ 44b UrhG)")
    logger.info(f"Rate Limit Delay   : {rate_limit_delay}s")
    logger.info(f"Output Raw Dir     : {raw_dir.resolve()}")
    logger.info("=" * 60)

    manifest = Manifest(manifest_path)
    compliance = ComplianceChecker(user_agent=user_agent, respect_robots_txt=respect_robots, filter_tdm=filter_tdm)
    limiter = DomainRateLimiter(default_delay=rate_limit_delay)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    # Determine site type and fetch candidate images
    candidates: List[CandidateImage] = []
    domain = compliance.get_domain(target_url)

    # 1. Robots.txt check for the target page
    if not compliance.is_url_allowed_by_robots(target_url, session=session):
        logger.warning(f"Target URL disallowed by robots.txt: {target_url}")
        return 1

    if "dbnaked.com" in domain:
        limiter.wait(target_url)
        resp = session.get(target_url, headers=DBNAKED_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch {target_url}: HTTP {resp.status_code}")
            return 1

        # TDM check on channel page
        is_reserved, reason = compliance.check_tdm_reservation(resp.headers, resp.text)
        if is_reserved:
            logger.warning(f"Crawling halted due to TDM reservation: {reason}")
            return 1

        if is_dbnaked_channel(target_url):
            scene_urls = extract_scene_links_from_channel(resp.text, target_url)
            logger.info(f"Found {len(scene_urls)} scenes in channel.")
            for s_url in scene_urls[:10]:  # batch of scenes
                limiter.wait(s_url)
                s_resp = session.get(s_url, headers=DBNAKED_HEADERS, timeout=timeout)
                if s_resp.status_code == 200:
                    candidates.extend(extract_dbnaked_gallery_candidates(s_url, s_resp.text))
        else:
            candidates.extend(extract_dbnaked_gallery_candidates(target_url, resp.text))

    else:
        # XenForo or generic forum
        pages = parse_page_range(pages_arg)
        for p in pages:
            page_url = build_page_url(target_url, p)
            limiter.wait(page_url)
            logger.info(f"Fetching forum page {p}: {page_url}")
            resp = session.get(page_url, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {page_url}: HTTP {resp.status_code}")
                continue

            # TDM check on forum page
            is_reserved, reason = compliance.check_tdm_reservation(resp.headers, resp.text)
            if is_reserved:
                logger.warning(f"Page {page_url} excluded due to TDM reservation: {reason}")
                continue

            page_candidates = extract_xenforo_candidates(page_url, resp.text)
            candidates.extend(page_candidates)

    logger.info(f"Found {len(candidates)} total candidate images.")

    # 2. Process Candidates with Pre-Filter & Download
    downloaded_count = 0
    skipped_existing = 0
    skipped_low_res = 0
    skipped_tdm = 0

    for idx, cand in enumerate(candidates, 1):
        # A. Idempotency Check via Manifest
        existing = manifest.get_by_url(cand.source_url)
        if existing and existing.stages_status.get("stage1_crawl") == "downloaded":
            if existing.raw_path and (Path(existing.raw_path).exists() or (raw_dir.parent / existing.raw_path).exists()):
                skipped_existing += 1
                continue

        # B. Pre-download Resolution Check
        width = cand.estimated_width
        height = cand.estimated_height

        # If not known from HTML attributes, fetch image header
        if width is None or height is None:
            limiter.wait(cand.source_url, custom_delay=0.2)
            header_dims = fetch_remote_resolution(
                cand.source_url,
                session=session,
                headers=cand.http_headers,
                timeout=timeout,
            )
            if header_dims:
                width, height = header_dims

        # Check Megapixel threshold
        if width and height:
            ok, mp = is_resolution_acceptable(width, height, min_megapixels=min_mp)
            if not ok:
                logger.debug(f"[PRE-FILTER SKIP] {width}x{height} ({mp:.2f} MP < {min_mp} MP): {cand.source_url}")
                skipped_low_res += 1
                continue

        # C. Download Image Payload
        limiter.wait(cand.source_url)
        try:
            img_resp = session.get(cand.source_url, headers=cand.http_headers, timeout=timeout)
            if img_resp.status_code != 200:
                logger.warning(f"Download failed ({img_resp.status_code}): {cand.source_url}")
                continue

            # Check TDM headers on image response
            is_reserved, reason = compliance.check_tdm_reservation(img_resp.headers)
            if is_reserved:
                logger.info(f"[TDM EXCLUDE] {reason}: {cand.source_url}")
                skipped_tdm += 1
                continue

            img_bytes = img_resp.content
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()

            # Determine actual dimensions
            try:
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    actual_w, actual_h = pil_img.size
            except Exception as e:
                logger.warning(f"Corrupt image data from {cand.source_url}: {e}")
                continue

            # Secondary post-download MP verification
            actual_mp = (actual_w * actual_h) / 1_000_000.0
            if actual_mp < min_mp:
                logger.debug(f"[POST-FETCH SKIP] Actual {actual_w}x{actual_h} ({actual_mp:.2f} MP < {min_mp} MP)")
                skipped_low_res += 1
                continue

            # Save File
            set_folder = sanitize_name(cand.context_title)
            dest_dir = raw_dir / cand.source / set_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            ext = os.path.splitext(urlparse(cand.source_url).path)[1].lower() or ".jpg"
            dest_filename = f"{sha256_hash[:16]}{ext}"
            dest_path = dest_dir / dest_filename

            with open(dest_path, "wb") as f:
                f.write(img_bytes)

            rel_raw_path = os.path.relpath(dest_path, raw_dir.parent).replace("\\", "/")

            # Register in Manifest
            entry = ManifestEntry(
                image_id=sha256_hash,
                source=cand.source,
                source_url=cand.source_url,
                page_url=cand.page_url,
                context_title=cand.context_title,
                context_tags=cand.context_tags,
                raw_path=rel_raw_path,
                sha256=sha256_hash,
                width=actual_w,
                height=actual_h,
                megapixels=round(actual_mp, 3),
                aspect_ratio=round(actual_w / actual_h, 3),
                stages_status={
                    "stage1_crawl": "downloaded",
                    "stage2_qc": "pending",
                    "stage3_downscale": "pending",
                    "stage4_caption": "pending",
                    "stage5_export": "pending",
                },
            )
            manifest.add_or_update(entry)
            downloaded_count += 1
            logger.info(f"[{idx}/{len(candidates)}] Downloaded: {dest_filename} ({actual_w}x{actual_h}, {actual_mp:.2f} MP)")

            # Save periodically every 10 images
            if downloaded_count % 10 == 0:
                manifest.save()

        except Exception as exc:
            logger.error(f"Error downloading {cand.source_url}: {exc}")

    # Final manifest save
    manifest.save()

    logger.info("=" * 60)
    logger.info("  STUFE 1 ERGEBNIS")
    logger.info(f"  Heruntergeladen        : {downloaded_count}")
    logger.info(f"  Übersprungen (Existiert): {skipped_existing}")
    logger.info(f"  Gefiltert (< {min_mp} MP): {skipped_low_res}")
    logger.info(f"  Ausgeschlossen (§ 44b)  : {skipped_tdm}")
    logger.info("=" * 60)

    return 0
