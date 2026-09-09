"""Stufe 1 – Crawling mit Vorfilter, Compliance-Prüfung und Manifest-Registrierung."""

from __future__ import annotations

import os
import io
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, unquote
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    build_dbnaked_page_url,
    extract_scene_links_from_channel,
    extract_dbnaked_gallery_candidates,
    DBNAKED_HEADERS,
)
from pipeline.crawler.scrolller import (
    is_scrolller_url,
    fetch_scrolller_candidates,
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
    workers = getattr(args, "max_workers", None) or int(crawl_cfg.get("max_workers", 8))
    user_delay = getattr(args, "delay", None)
    rate_limit_delay = float(user_delay) if user_delay is not None else float(crawl_cfg.get("default_rate_limit_delay", 1.0))
    timeout = float(crawl_cfg.get("request_timeout", 25))

    target_url = getattr(args, "url", None) or "https://xxx-files.org/threads/hard-tied-photocollection.20611/page-27"
    pages_arg = getattr(args, "pages", None) or "27"

    logger.info("=" * 60)
    logger.info("  STUFE 1: CRAWLING MIT VORFILTER & COMPLIANCE")
    logger.info("=" * 60)
    logger.info(f"Target URL         : {target_url}")
    logger.info(f"Pages              : {pages_arg}")
    logger.info(f"Workers            : {workers}")
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
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max(10, workers * 2),
        pool_maxsize=max(20, workers * 4),
        max_retries=2,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Determine site type and fetch candidate images
    candidates: List[CandidateImage] = []
    domain = compliance.get_domain(target_url)

    # 1. Robots.txt check for the target page
    if not compliance.is_url_allowed_by_robots(target_url, session=session):
        logger.warning(f"Target URL disallowed by robots.txt: {target_url}")
        return 1

    if is_scrolller_url(target_url) or "scrolller.com" in domain:
        user_pages = getattr(args, "pages", None)
        if user_pages:
            pages = parse_page_range(user_pages)
            max_p = max(pages) if pages else 10
        else:
            max_p = 25  # fetch all/up to 25 pages by default for scrolller

        candidates.extend(
            fetch_scrolller_candidates(
                url=target_url,
                session=session,
                max_pages=max_p,
                timeout=timeout,
            )
        )

    elif "dbnaked.com" in domain:
        if is_dbnaked_channel(target_url):
            pages = parse_page_range(pages_arg)
            all_scene_urls = set()

            for p in pages:
                p_url = build_dbnaked_page_url(target_url, p)
                limiter.wait(p_url)
                logger.info(f"Fetching dbNaked channel page {p}: {p_url}")
                resp = session.get(p_url, headers=DBNAKED_HEADERS, timeout=timeout)
                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch {p_url}: HTTP {resp.status_code}")
                    continue

                is_reserved, reason = compliance.check_tdm_reservation(resp.headers, resp.text)
                if is_reserved:
                    logger.warning(f"Crawling halted on {p_url} due to TDM reservation: {reason}")
                    continue

                scenes = extract_scene_links_from_channel(resp.text, p_url)
                logger.info(f"Found {len(scenes)} scenes on page {p}.")
                all_scene_urls.update(scenes)

            logger.info(f"Discovered {len(all_scene_urls)} total unique scenes in channel across pages {pages_arg}.")
            for s_url in sorted(list(all_scene_urls)):
                limiter.wait(s_url)
                s_resp = session.get(s_url, headers=DBNAKED_HEADERS, timeout=timeout)
                if s_resp.status_code == 200:
                    candidates.extend(extract_dbnaked_gallery_candidates(s_url, s_resp.text))
        else:
            limiter.wait(target_url)
            resp = session.get(target_url, headers=DBNAKED_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                candidates.extend(extract_dbnaked_gallery_candidates(target_url, resp.text))
            else:
                logger.error(f"Failed to fetch {target_url}: HTTP {resp.status_code}")
                return 1

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

    # 2. Process Candidates with Pre-Filter & Download (Concurrent Workers)
    downloaded_count = 0
    skipped_existing = 0
    skipped_low_res = 0
    skipped_tdm = 0
    processed_count = 0

    def download_candidate(cand_idx: int, cand: CandidateImage) -> tuple[str, Any]:
        """Downloads, checks resolution, and registers a single candidate image."""
        # A. Idempotency Check via Manifest
        existing = manifest.get_by_url(cand.source_url)
        if existing and existing.stages_status.get("stage1_crawl") == "downloaded":
            if existing.raw_path and (Path(existing.raw_path).exists() or (raw_dir.parent / existing.raw_path).exists()):
                return ("skipped_existing", cand.source_url)

        # B. Pre-download Resolution Check
        width = cand.estimated_width
        height = cand.estimated_height

        if width is None or height is None:
            limiter.wait(cand.source_url, custom_delay=min(0.2, rate_limit_delay))
            header_dims = fetch_remote_resolution(
                cand.source_url,
                session=session,
                headers=cand.http_headers,
                timeout=timeout,
            )
            if header_dims:
                width, height = header_dims

        if width and height:
            ok, mp = is_resolution_acceptable(width, height, min_megapixels=min_mp)
            if not ok:
                logger.debug(f"[PRE-FILTER SKIP] {width}x{height} ({mp:.2f} MP < {min_mp} MP): {cand.source_url}")
                return ("skipped_low_res", cand.source_url)

        # C. Download Image Payload
        limiter.wait(cand.source_url)
        try:
            img_resp = session.get(cand.source_url, headers=cand.http_headers, timeout=timeout)
            if img_resp.status_code != 200:
                logger.warning(f"Download failed ({img_resp.status_code}): {cand.source_url}")
                return ("failed", cand.source_url)

            # Check TDM headers on image response
            is_reserved, reason = compliance.check_tdm_reservation(img_resp.headers)
            if is_reserved:
                logger.info(f"[TDM EXCLUDE] {reason}: {cand.source_url}")
                return ("skipped_tdm", cand.source_url)

            img_bytes = img_resp.content
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()

            # Determine actual dimensions
            try:
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    actual_w, actual_h = pil_img.size
            except Exception as e:
                logger.warning(f"Corrupt image data from {cand.source_url}: {e}")
                return ("corrupt", cand.source_url)

            # Secondary post-download MP verification
            actual_mp = (actual_w * actual_h) / 1_000_000.0
            if actual_mp < min_mp:
                logger.debug(f"[POST-FETCH SKIP] Actual {actual_w}x{actual_h} ({actual_mp:.2f} MP < {min_mp} MP)")
                return ("skipped_low_res", cand.source_url)

            # Save File
            set_folder = sanitize_name(cand.context_title)
            dest_dir = raw_dir / cand.source / set_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            ext = os.path.splitext(urlparse(cand.source_url).path)[1].lower() or ".jpg"
            dest_filename = f"{sha256_hash[:16]}{ext}"
            dest_path = dest_dir / dest_filename

            with open(dest_path, "wb") as f:
                f.write(img_bytes)

            rel_raw_path = str(dest_path).replace("\\", "/")

            # Register in Manifest (thread-safe)
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
            return ("downloaded", dest_filename, actual_w, actual_h, actual_mp)

        except Exception as exc:
            logger.error(f"Error downloading {cand.source_url}: {exc}")
            return ("error", cand.source_url)

    def _handle_result(res: tuple[str, Any]) -> None:
        nonlocal downloaded_count, skipped_existing, skipped_low_res, skipped_tdm, processed_count
        processed_count += 1
        status = res[0]
        if status == "downloaded":
            downloaded_count += 1
            _, dest_filename, actual_w, actual_h, actual_mp = res
            logger.info(f"[{processed_count}/{len(candidates)}] Downloaded: {dest_filename} ({actual_w}x{actual_h}, {actual_mp:.2f} MP)")
            if downloaded_count % 10 == 0:
                manifest.save()
        elif status == "skipped_existing":
            skipped_existing += 1
        elif status == "skipped_low_res":
            skipped_low_res += 1
        elif status == "skipped_tdm":
            skipped_tdm += 1

    if workers <= 1:
        logger.info("Executing downloads sequentially (1 worker)...")
        for idx, cand in enumerate(candidates, 1):
            res = download_candidate(idx, cand)
            _handle_result(res)
    else:
        logger.info(f"Executing downloads concurrently with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_cand = {
                executor.submit(download_candidate, idx, cand): (idx, cand)
                for idx, cand in enumerate(candidates, 1)
            }
            for future in as_completed(future_to_cand):
                idx, cand = future_to_cand[future]
                try:
                    res = future.result()
                    _handle_result(res)
                except Exception as exc:
                    logger.error(f"Worker exception on image {cand.source_url}: {exc}")

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
