#!/usr/bin/env python3
"""
dbNaked Image Gallery & Channel Crawler
========================================
Extracts high-resolution (1600x1600) image sets and metadata from dbnaked.com channels,
studios, and picture galleries. Automatically generates set folders and 'genres.txt'
metadata compatible with the Krea 2 VLM Captioner.

Usage:
  python dbnaked_crawler.py --url "https://dbnaked.com/bdsm/channels/infernalrestraints.com" --pages 1-3
  python dbnaked_crawler.py --url "https://dbnaked.com/bdsm/channels/infernalrestraints.com" -p 1 -o ./downloads
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib.request
from bs4 import BeautifulSoup
from PIL import Image
import io

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        DownloadColumn,
        TimeRemainingColumn,
    )
    HAVE_RICH = True
    console = Console()
except ImportError:
    HAVE_RICH = False
    console = None

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

logger = logging.getLogger("dbnaked_crawler")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def sanitize_folder_name(name: str, max_length: int = 120) -> str:
    """Sanitizes folder name for safe OS directory creation."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(". ")
    if not name:
        name = "gallery_set"
    return name[:max_length].strip()


def parse_page_range(pages_str: str, max_available: Optional[int] = None) -> List[int]:
    """Parses ranges like '1', '1-5', '1,3,5', or 'all'."""
    pages_str = pages_str.strip().lower()
    if pages_str == "all":
        limit = max_available or 25
        return list(range(1, limit + 1))

    pages: Set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s_str, e_str = part.split("-", 1)
            start, end = int(s_str.strip()), int(e_str.strip())
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                if p >= 1:
                    pages.add(p)
        else:
            p = int(part)
            if p >= 1:
                pages.add(p)
    return sorted(list(pages))


class DbNakedCrawler:
    def __init__(
        self,
        channel_url: str,
        output_dir: str = "./downloads",
        workers: int = 4,
        delay: float = 1.0,
        max_sets: Optional[int] = None,
        max_images: Optional[int] = None,
        overwrite: bool = False,
        dry_run: bool = False,
    ):
        self.channel_url = channel_url.strip()
        self.output_dir = Path(output_dir).resolve()
        self.workers = workers
        self.delay = delay
        self.max_sets = max_sets
        self.max_images = max_images
        self.overwrite = overwrite
        self.dry_run = dry_run

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

        self.total_pages: Optional[int] = None

    def build_page_url(self, page_num: int) -> str:
        """Constructs channel page URL with media=pictures parameter."""
        parsed = urlparse(self.channel_url)
        params = parse_qs(parsed.query)
        params["media"] = ["pictures"]
        params["sort"] = ["latest"]
        if page_num > 1:
            params["page"] = [str(page_num)]
        elif "page" in params:
            del params["page"]

        new_query = urlencode(params, doseq=True)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

    def detect_total_pages(self, soup: BeautifulSoup) -> int:
        """Extracts total pages from pagination links."""
        max_p = 1
        for a in soup.select("ul.pagination a, .pagination a"):
            href = a.get("href", "")
            m = re.search(r"[?&]page=(\d+)", href)
            if m:
                max_p = max(max_p, int(m.group(1)))
            text = a.get_text(strip=True)
            if text.isdigit():
                max_p = max(max_p, int(text))
        return max_p

    def fetch_channel_page(self, page_num: int) -> Tuple[List[str], int]:
        """Fetches channel page and returns list of gallery URLs."""
        url = self.build_page_url(page_num)
        logger.info(f"Fetching channel page {page_num}: {url}")
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        total_p = self.detect_total_pages(soup)

        gallery_urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Match gallery paths like /pictures/content/bdsm/sites/.../185248_ransom
            if "/pictures/content/" in href and re.search(r"/\d+_[^/?#]+$", href):
                full = urljoin(url, href)
                if full not in gallery_urls:
                    gallery_urls.append(full)

        return gallery_urls, total_p

    def parse_gallery_page(self, gallery_url: str) -> Optional[Dict]:
        """Extracts full gallery metadata, models, description, and high-res image URLs."""
        try:
            resp = self.session.get(gallery_url, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch gallery {gallery_url}: status {resp.status_code}")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract scene ID & slug from URL
            m_id = re.search(r"/(\d+)_([^/?#]+)$", gallery_url)
            scene_id = m_id.group(1) if m_id else "unknown_id"
            slug = m_id.group(2) if m_id else "gallery"

            # Title
            h1 = soup.find("h1")
            raw_title = h1.get_text(strip=True) if h1 else slug.replace("-", " ").title()
            # Clean title
            title = re.sub(r'["\'-].*$', '', raw_title).strip() or slug.replace("-", " ").title()

            # Studio / Channel name
            channel_el = soup.select_one("a[href*='/pictures/content/'][href*='/sites/'], a[href*='/channel/']")
            studio = channel_el.get_text(strip=True).replace("@", "").strip() if channel_el else "Unknown Studio"

            # Models
            models = []
            for a in soup.select("a[href*='/models/'], a[href*='/pornstars/']"):
                t = a.get_text(strip=True)
                if t and t not in models and len(t) < 40 and not t.isdigit():
                    models.append(t)

            # Categories / Genres
            genres = []
            for a in soup.select("a[href*='/categories/'], a[href*='/tags/'], a[href*='/tag/']"):
                g = a.get_text(strip=True)
                if g and g not in genres and len(g) < 30:
                    genres.append(g)

            # Scene description
            desc = ""
            desc_meta = soup.find("meta", attrs={"name": "description"})
            if desc_meta and desc_meta.get("content"):
                desc = desc_meta["content"].strip()
                # Remove common boilerplate prefix
                desc = re.sub(r'^"[^"]+"\s*#\d+:\s*', '', desc)
                desc = re.sub(r'\s*@\s*dbNaked$', '', desc).strip()

            # High-res images: look for links to /t1600x1600/ or direct scene images
            image_urls = []
            seen_imgs = set()

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if "/t1600x1600/" in href or (f"/scene/{scene_id}/" in href and href.endswith(".jpg")):
                    if href.startswith("//"):
                        href = "https:" + href
                    elif not href.startswith("http"):
                        href = urljoin("https://dbnaked.com", href)
                    if href not in seen_imgs:
                        seen_imgs.add(href)
                        image_urls.append(href)

            # Fallback: inspect img tags if no direct 1600x1600 links found
            if not image_urls:
                for img in soup.find_all("img"):
                    src = (img.get("src") or img.get("data-src") or "").strip()
                    if f"/scene/{scene_id}/" in src and src.endswith(".jpg"):
                        highres = src.replace("/t300x300/", "/t1600x1600/").replace("/t800x800/", "/t1600x1600/")
                        if highres.startswith("//"):
                            highres = "https:" + highres
                        elif not highres.startswith("http"):
                            highres = urljoin("https://dbnaked.com", highres)
                        if highres not in seen_imgs:
                            seen_imgs.add(highres)
                            image_urls.append(highres)

            return {
                "scene_id": scene_id,
                "title": title,
                "slug": slug,
                "studio": studio,
                "models": models,
                "genres": genres,
                "description": desc,
                "gallery_url": gallery_url,
                "images": image_urls,
            }

        except Exception as exc:
            logger.error(f"Error parsing gallery {gallery_url}: {exc}")
            return None

    def write_set_genres(self, set_dir: Path, meta: Dict):
        """Creates genres.txt in the set directory."""
        if self.dry_run:
            return

        set_dir.mkdir(parents=True, exist_ok=True)
        genre_file = set_dir / "genres.txt"
        if genre_file.exists() and not self.overwrite:
            return

        genres_list = meta.get("genres", [])
        models_list = meta.get("models", [])
        genres_str = ", ".join(genres_list) if genres_list else "Bondage, Fetish"
        models_str = ", ".join(models_list) if models_list else "N/A"

        lines = [
            f"Genres: {genres_str}",
            "",
            f"Set Title: {meta.get('title', 'Unknown Set')}",
            f"Studio: {meta.get('studio', 'N/A')}",
            f"Models: {models_str}",
            f"Scene ID: {meta.get('scene_id', 'N/A')}",
            f"Source URL: {meta.get('gallery_url', '')}",
        ]

        if meta.get("description"):
            lines.append("")
            lines.append(f"Description: {meta['description']}")

        if genres_list or models_list:
            lines.append("")
            lines.append("Tags:")
            for m in models_list:
                lines.append(f"- Model: {m}")
            for g in genres_list:
                lines.append(f"- {g}")

        lines.append("")

        with open(genre_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def download_image(self, img_url: str, dest_path: Path) -> bool:
        """Downloads single image with required Referer header via urllib."""
        if dest_path.exists() and not self.overwrite:
            return True

        if self.dry_run:
            return True

        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif not img_url.startswith("http"):
            img_url = urljoin("https://dbnaked.com", img_url)

        req = urllib.request.Request(
            img_url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Referer": "https://dbnaked.com/",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Connection": "close",
            }
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        content = resp.read()
                        if len(content) > 1024:
                            with open(dest_path, "wb") as f:
                                f.write(content)
                            return True
                time.sleep(0.5)
            except Exception:
                time.sleep(1.0)

        return False

    def crawl(self, pages: List[int]) -> int:
        """Executes crawl over the specified page range."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("       DBNAKED HIGH-RES GALLERY CRAWLER")
        logger.info("=" * 60)
        logger.info(f"Target Channel : {self.channel_url}")
        logger.info(f"Output Base Dir: {self.output_dir}")
        logger.info(f"Pages to Crawl : {pages}")
        logger.info(f"Workers        : {self.workers}")
        logger.info("=" * 60)

        total_downloaded = 0
        total_sets_processed = 0

        for page in pages:
            try:
                gallery_urls, max_p = self.fetch_channel_page(page)
                self.total_pages = max_p

                if not gallery_urls:
                    logger.warning(f"No galleries found on page {page}.")
                    continue

                logger.info(f"Page {page}: Found {len(gallery_urls)} galleries.")

                for gal_url in gallery_urls:
                    if self.max_sets and total_sets_processed >= self.max_sets:
                        logger.info(f"Reached maximum sets limit ({self.max_sets}). Stopping.")
                        return total_downloaded

                    logger.info(f"[*] Processing gallery: {gal_url}")
                    meta = self.parse_gallery_page(gal_url)
                    if not meta or not meta["images"]:
                        logger.warning(f"No images found for {gal_url}")
                        continue

                    # Create set directory name: e.g. "Infernal Restraints - Ransom 185248"
                    clean_studio = sanitize_folder_name(meta["studio"])
                    clean_title = sanitize_folder_name(meta["title"])
                    folder_name = f"{clean_studio} - {clean_title} ({meta['scene_id']})"
                    set_dir = self.output_dir / folder_name

                    # Write genres.txt
                    self.write_set_genres(set_dir, meta)

                    # Download all images
                    images = meta["images"]
                    logger.info(f"    Downloading {len(images)} high-res images into: {folder_name}")

                    download_tasks = []
                    for idx, img_url in enumerate(images, 1):
                        dest_name = f"{meta['slug']}_{idx:03d}.jpg"
                        dest_path = set_dir / dest_name
                        download_tasks.append((img_url, dest_path))

                    set_saved = 0
                    if self.workers > 1:
                        with ThreadPoolExecutor(max_workers=self.workers) as pool:
                            futures = [pool.submit(self.download_image, u, p) for u, p in download_tasks]
                            for fut in as_completed(futures):
                                if fut.result():
                                    set_saved += 1
                                    total_downloaded += 1
                    else:
                        for u, p in download_tasks:
                            if self.download_image(u, p):
                                set_saved += 1
                                total_downloaded += 1

                    logger.info(f"    [+] Saved {set_saved}/{len(images)} images + genres.txt")
                    total_sets_processed += 1

                    if self.delay > 0:
                        time.sleep(self.delay)

            except Exception as exc:
                logger.error(f"Error on page {page}: {exc}")

        logger.info("=" * 60)
        logger.info(f"CRAWL COMPLETE: Processed {total_sets_processed} sets, downloaded {total_downloaded} images.")
        logger.info("=" * 60)
        return total_downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Scrape 1600x1600 image sets and genres.txt from dbnaked.com channels."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://dbnaked.com/bdsm/channels/infernalrestraints.com",
        help="Channel or studio URL to crawl.",
    )
    parser.add_argument(
        "-p", "--pages",
        type=str,
        default="1",
        help="Page or page range to crawl (e.g. '1', '1-3', '1,2,5', or 'all'). Default: 1",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./downloads",
        help="Output directory for image sets (default: ./downloads).",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of concurrent download worker threads (default: 4).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between galleries (default: 1.0).",
    )
    parser.add_argument(
        "--max-sets",
        type=int,
        default=None,
        help="Stop after N image sets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images instead of skipping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List galleries and images without downloading.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    crawler = DbNakedCrawler(
        channel_url=args.url,
        output_dir=args.output_dir,
        workers=args.workers,
        delay=args.delay,
        max_sets=args.max_sets,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    pages = parse_page_range(args.pages)
    crawler.crawl(pages)


if __name__ == "__main__":
    main()
