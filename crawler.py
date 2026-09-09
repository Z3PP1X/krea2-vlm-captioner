#!/usr/bin/env python3
"""
Multi-Site Forum & Gallery Image Crawler
Supported Sites:
  - xxx-files.org (and generic XenForo 1.x / 2.x boards)
  - dbnaked.com (channels, galleries, categories, and direct scenes)

Features:
  - Extracts full-resolution images
  - Organizes downloads by image set / scene folders
  - Generates genres.txt for each set containing tags & metadata
  - Multi-threaded downloads with resume support & polite delays
"""

import os
import re
import sys
import time
import json
import logging
import argparse
import functools
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure unbuffered printing so logs appear in real-time
print = functools.partial(print, flush=True)

import requests
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
        TransferSpeedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    HAVE_RICH = True
    console = Console()
except ImportError:
    HAVE_RICH = False
    console = None

DEFAULT_URL = "https://xxx-files.org/threads/hard-tied-photocollection.20611/page-27"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_MIN_SIZE_BYTES = 15 * 1024  # 15 KB (filters out avatars, banners, icons)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def sanitize_filename(filename: str, max_length: int = 150) -> str:
    """Sanitizes a string to be a safe, valid filename across operating systems."""
    filename = unquote(filename)
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    filename = re.sub(r"\s+", " ", filename).strip(". ")
    if not filename:
        filename = "image"
    name, ext = os.path.splitext(filename)
    if len(filename) > max_length:
        name = name[: max_length - len(ext) - 5].strip()
        filename = f"{name}{ext}"
    return filename


def sanitize_folder_name(name: str, max_length: int = 120) -> str:
    """Sanitizes a string to be a safe folder name."""
    name = unquote(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(". ")
    if not name:
        name = "image_set"
    if len(name) > max_length:
        name = name[:max_length].strip()
    return name


def unique_list(seq: List[str]) -> List[str]:
    """Preserves order while removing case-insensitive duplicates."""
    seen = set()
    res = []
    for x in seq:
        x_clean = x.strip()
        if x_clean and x_clean.lower() not in seen:
            seen.add(x_clean.lower())
            res.append(x_clean)
    return res


def parse_page_range(pages_str: str, max_available: Optional[int] = None) -> List[int]:
    """Parses a page range string such as '1', '1-5', '11-123', or 'all'."""
    pages_str = pages_str.strip().lower()
    if pages_str == "all":
        if max_available is None or max_available < 1:
            raise ValueError("Cannot determine total pages for 'all'. Please specify a numeric range.")
        return list(range(1, max_available + 1))

    pages: Set[int] = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
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


# ==============================================================================
# Base Crawler
# ==============================================================================
class BaseCrawler:
    def __init__(
        self,
        output_dir: str = "./downloads",
        by_set: bool = True,
        by_page: bool = False,
        workers: int = 4,
        delay: float = 1.0,
        min_size: int = DEFAULT_MIN_SIZE_BYTES,
        max_images: Optional[int] = None,
        overwrite: bool = False,
        dry_run: bool = False,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.by_set = by_set
        self.by_page = by_page
        self.workers = workers
        self.delay = delay
        self.min_size = min_size
        self.max_images = max_images
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.user_agent = user_agent

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        self.manifest: Dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, dict]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        if not self.dry_run:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def download_image(self, item: Dict, target_path: str, referer: Optional[str] = None) -> Dict:
        """Downloads a single image, validating size and format."""
        url = item["image_url"]
        headers = {}
        if referer:
            headers["Referer"] = referer

        if not self.overwrite and os.path.exists(target_path):
            existing_size = os.path.getsize(target_path)
            if existing_size >= self.min_size:
                return {
                    "status": "skipped",
                    "reason": "exists",
                    "path": target_path,
                    "bytes": existing_size,
                    "item": item,
                }

        retries = 3
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=headers, timeout=20, stream=True)
                if resp.status_code == 200:
                    content = resp.content
                    if len(content) < self.min_size:
                        return {
                            "status": "filtered",
                            "reason": f"too small ({len(content)} bytes < {self.min_size})",
                            "item": item,
                        }

                    try:
                        im = Image.open(io.BytesIO(content))
                        im.verify()
                    except Exception as img_err:
                        return {
                            "status": "failed",
                            "reason": f"invalid image file: {img_err}",
                            "item": item,
                        }

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "wb") as f:
                        f.write(content)

                    return {
                        "status": "success",
                        "path": target_path,
                        "bytes": len(content),
                        "item": item,
                    }
                else:
                    last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5 * (attempt + 1))

        return {"status": "failed", "reason": last_err, "item": item}


# ==============================================================================
# XenForo Crawler (xxx-files.org)
# ==============================================================================
class XenForoImageCrawler(BaseCrawler):
    def __init__(self, thread_url: str = DEFAULT_URL, naming: str = "original", **kwargs):
        super().__init__(**kwargs)
        self.raw_url = thread_url
        self.naming = naming
        self.base_thread_url, self.initial_page = self._parse_thread_base(thread_url)
        self.total_thread_pages: Optional[int] = None
        self.thread_tags: List[str] = []
        self.existing_names: Set[str] = set()
        self.created_sets: Set[str] = set()

        self.signature_patterns = [
            re.compile(r"picstate\.com/files/.*/BEL\.png", re.IGNORECASE),
            re.compile(r"data/avatars/", re.IGNORECASE),
            re.compile(r"styles/.*logo\.png", re.IGNORECASE),
            re.compile(r"smilies/", re.IGNORECASE),
        ]

    def _parse_thread_base(self, url: str) -> Tuple[str, int]:
        parsed = urlparse(url)
        path = parsed.path
        page_match = re.search(r"/page-(\d+)/?$", path)
        if page_match:
            page_num = int(page_match.group(1))
            base_path = path[: page_match.start()]
            if not base_path.endswith("/"):
                base_path += "/"
            base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
            return base_url, page_num
        else:
            clean_url = url.split("?")[0].split("#")[0]
            if not clean_url.endswith("/"):
                clean_url += "/"
            return clean_url, 1

    def get_page_url(self, page_num: int) -> str:
        if page_num == 1:
            return self.base_thread_url
        return f"{self.base_thread_url}page-{page_num}"

    def fetch_thread_info(self) -> int:
        target_url = self.get_page_url(self.initial_page)
        res = self.session.get(target_url, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")

        nav = soup.select_one(".PageNav, .pageNav")
        if nav and nav.get("data-last"):
            try:
                self.total_thread_pages = int(nav.get("data-last"))
            except ValueError:
                self.total_thread_pages = 1
        else:
            page_links = soup.select(".PageNav a, .pageNav a, .pageNav-page a")
            max_p = 1
            for a in page_links:
                m = re.search(r"page-(\d+)", a.get("href", ""))
                if m:
                    max_p = max(max_p, int(m.group(1)))
            self.total_thread_pages = max_p

        for t_tag in soup.select(".tagList a.tag, .tagBlock a.tag, a.tag"):
            t_text = t_tag.text.strip()
            if t_text and t_text not in self.thread_tags:
                self.thread_tags.append(t_text)

        return self.total_thread_pages or 1

    def is_signature_or_banner(self, url: str) -> bool:
        for pat in self.signature_patterns:
            if pat.search(url):
                return True
        return False

    def parse_post_metadata(self, post_element) -> Dict:
        msg_text = post_element.select_one(".messageText, .messageContent, .message-body")
        if not msg_text:
            return {"title": None, "genres": [], "metadata": {}}

        lines = [l.strip() for l in msg_text.stripped_strings if l.strip()]
        title = None
        genres: List[str] = []
        metadata: Dict[str, str] = {}

        for line in lines:
            m_genre = re.match(r"^(?:Genres?|Tags?)\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_genre:
                raw_genres = m_genre.group(1).strip().rstrip(",")
                genres = [g.strip() for g in raw_genres.split(",") if g.strip()]
                metadata["genres_raw"] = raw_genres
                continue

            m_format = re.match(r"^Format\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_format:
                metadata["format"] = m_format.group(1).strip()
                continue

            m_photos = re.match(r"^(?:Number of photos|Photos)\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_photos:
                metadata["photos_count"] = m_photos.group(1).strip()
                continue

            m_res = re.match(r"^Resolution\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_res:
                metadata["resolution"] = m_res.group(1).strip()
                continue

            m_size = re.match(r"^Filesize\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_size:
                metadata["filesize"] = m_size.group(1).strip()
                continue

            if not title and not line.startswith("http") and not line.startswith("[") and len(line) < 120:
                if not any(line.lower().startswith(k) for k in ["download", "format", "resolution", "genre", "filesize", "number"]):
                    title = line

        if not genres and self.thread_tags:
            genres = list(self.thread_tags)
            metadata["genres_raw"] = ", ".join(genres)

        return {"title": title, "genres": genres, "metadata": metadata}

    def write_set_genre_file(self, set_dir: str, post_meta: Dict, page_num: int, post_id: str):
        if self.dry_run:
            return
        genre_file = os.path.join(set_dir, "genres.txt")
        if os.path.exists(genre_file) and not self.overwrite:
            return

        os.makedirs(set_dir, exist_ok=True)
        title = post_meta.get("title") or "Unknown Set"
        genres = post_meta.get("genres") or []
        genres_raw = post_meta.get("metadata", {}).get("genres_raw") or ", ".join(genres) or "N/A"
        meta = post_meta.get("metadata", {})

        content_lines = [
            f"Genres: {genres_raw}",
            "",
            f"Set Title: {title}",
            f"Forum Page: {page_num}",
            f"Post ID: {post_id}",
        ]
        for k, label in [("format", "Format"), ("photos_count", "Number of photos"), ("resolution", "Resolution"), ("filesize", "Filesize")]:
            if k in meta:
                content_lines.append(f"{label}: {meta[k]}")

        if genres:
            content_lines.append("")
            content_lines.append("Tags:")
            for g in genres:
                content_lines.append(f"- {g}")

        content_lines.append("")
        with open(genre_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content_lines))

    def extract_image_items(self, page_num: int, soup: BeautifulSoup) -> List[Dict]:
        items: List[Dict] = []
        seen_urls: Set[str] = set()

        messages = soup.select("li.message, .messageList .message, article.message")
        for msg_idx, msg in enumerate(messages):
            post_id = msg.get("id") or f"post_{msg_idx}"
            body = msg.select_one(".messageText, .messageContent, .message-body")
            if not body:
                continue

            post_meta = self.parse_post_metadata(msg)
            set_title = post_meta.get("title")
            set_folder_name = sanitize_folder_name(set_title) if set_title else f"set_{post_id}"

            # Linked images (<a href="..."><img src="..."></a>)
            for a_tag in body.find_all("a"):
                href = a_tag.get("href", "").strip()
                img = a_tag.find("img")
                if not img:
                    continue

                img_src = (img.get("src") or img.get("data-url") or img.get("data-src") or "").strip()
                if not img_src or self.is_signature_or_banner(img_src):
                    continue

                img_url = urljoin(self.base_thread_url, img_src)
                if img_url in seen_urls:
                    continue

                filename = None
                if href and any(ext in href.lower() for ext in [".jpg", ".png", ".webp", ".jpeg"]):
                    clean_href = re.sub(r"\.html?$", "", href, flags=re.IGNORECASE)
                    path_parts = clean_href.rstrip("/").split("/")
                    candidate = path_parts[-1]
                    if any(candidate.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        filename = candidate

                if not filename:
                    img_path = urlparse(img_url).path
                    filename = os.path.basename(img_path)

                filename = sanitize_filename(filename)
                seen_urls.add(img_url)
                items.append({
                    "page": page_num,
                    "post_id": post_id,
                    "set_name": set_folder_name,
                    "post_meta": post_meta,
                    "image_url": img_url,
                    "landing_url": href or None,
                    "filename": filename,
                })

            # Standalone images
            for img in body.find_all("img"):
                if img.parent and img.parent.name == "a":
                    continue
                img_src = (img.get("src") or img.get("data-url") or img.get("data-src") or "").strip()
                if not img_src or self.is_signature_or_banner(img_src):
                    continue

                img_url = urljoin(self.base_thread_url, img_src)
                if img_url in seen_urls:
                    continue

                img_path = urlparse(img_url).path
                filename = sanitize_filename(os.path.basename(img_path))
                seen_urls.add(img_url)
                items.append({
                    "page": page_num,
                    "post_id": post_id,
                    "set_name": set_folder_name,
                    "post_meta": post_meta,
                    "image_url": img_url,
                    "landing_url": None,
                    "filename": filename,
                })

        return items

    def _determine_target_path(self, item: Dict, existing_names: Set[str]) -> Tuple[str, str]:
        if self.by_set:
            folder = os.path.join(self.output_dir, f"page_{item['page']:03d}", item["set_name"]) if self.by_page else os.path.join(self.output_dir, item["set_name"])
        elif self.by_page:
            folder = os.path.join(self.output_dir, f"page_{item['page']:03d}")
        else:
            folder = self.output_dir

        base_name = item["filename"]
        name, ext = os.path.splitext(base_name)
        if not ext:
            ext = ".jpg"

        folder_lower = folder.lower()
        candidate = f"{name}{ext}"
        counter = 1
        full_key = f"{folder_lower}::{candidate.lower()}"
        while full_key in existing_names:
            candidate = f"{name}_{counter}{ext}"
            counter += 1
            full_key = f"{folder_lower}::{candidate.lower()}"

        existing_names.add(full_key)
        return folder, os.path.join(folder, candidate)

    def crawl_page(self, page_num: int) -> List[Dict]:
        """Scrapes a single page, creates set folders and genres.txt, returns image items."""
        page_url = self.get_page_url(page_num)
        try:
            res = self.session.get(page_url, timeout=20)
            if res.status_code != 200:
                return []
            soup = BeautifulSoup(res.text, "html.parser")
            items = self.extract_image_items(page_num, soup)
            prepared_items = []
            for it in items:
                folder, target_path = self._determine_target_path(it, self.existing_names)
                if self.by_set and folder not in self.created_sets:
                    self.write_set_genre_file(folder, it["post_meta"], page_num, it["post_id"])
                    self.created_sets.add(folder)
                it["target_path"] = target_path
                prepared_items.append(it)
            return prepared_items
        except Exception as e:
            print(f"Error crawling page {page_num}: {e}")
            return []

    def download_items(self, items: List[Dict]) -> int:
        """Downloads a list of image items in parallel, returns count of downloaded images."""
        if not items:
            return 0
        success_count = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.download_image, item, item["target_path"], self.base_thread_url): item
                for item in items
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res["status"] in ("success", "skipped"):
                    success_count += 1
                    rel_path = os.path.relpath(res["path"], self.output_dir)
                    self.manifest[res["item"]["image_url"]] = {
                        "filename": rel_path,
                        "set_name": res["item"]["set_name"],
                        "bytes": res.get("bytes", 0),
                        "page": res["item"]["page"],
                        "post_id": res["item"]["post_id"],
                        "timestamp": time.time(),
                    }
        self._save_manifest()
        return success_count

    def run(self, pages: List[int]) -> Dict:
        if HAVE_RICH:
            console.print(Panel.fit(
                f"[bold cyan]XenForo Forum Image Crawler[/bold cyan]\n"
                f"[dim]Target Thread:[/dim] {self.base_thread_url}\n"
                f"[dim]Pages to crawl:[/dim] {pages[0]} to {pages[-1]} ({len(pages)} pages)\n"
                f"[dim]Output Directory:[/dim] {self.output_dir}\n"
                f"[dim]Folder per Set:[/dim] {self.by_set} (with genres.txt)\n"
                f"[dim]Threads:[/dim] {self.workers} | [dim]Min Size:[/dim] {self.min_size // 1024} KB | [dim]Mode:[/dim] {'DRY-RUN' if self.dry_run else 'DOWNLOAD'}",
                title="[bold green]Crawler Configuration[/bold green]",
            ))

        results = {"success": 0, "skipped": 0, "filtered": 0, "failed": 0, "total_bytes": 0, "sets_created": 0}
        total_images_processed = 0

        for page_idx, page_num in enumerate(pages):
            page_url = self.get_page_url(page_num)
            msg = f"[{page_idx+1}/{len(pages)}] Scraping Page {page_num} ({page_url})..."
            if HAVE_RICH:
                console.print(f"[bold cyan]{msg}[/bold cyan]")
            else:
                print(msg)

            page_items = self.crawl_page(page_num)
            print(f"  -> Found {len(page_items)} images across post sets.")

            if self.max_images and (total_images_processed + len(page_items)) > self.max_images:
                page_items = page_items[: self.max_images - total_images_processed]

            if self.dry_run:
                for it in page_items[:10]:
                    print(f"  [DRY-RUN] -> Set: {it['set_name']} | File: {it['filename']}")
                total_images_processed += len(page_items)
                if self.max_images and total_images_processed >= self.max_images:
                    break
                continue

            cnt = self.download_items(page_items)
            results["success"] += cnt
            total_images_processed += len(page_items)
            print(f"  Page {page_num} done: {len(page_items)} images (Downloaded: {cnt})\n")

            if self.max_images and total_images_processed >= self.max_images:
                break

            if page_idx < len(pages) - 1 and self.delay > 0:
                time.sleep(self.delay)

        self._save_manifest()
        return results


# ==============================================================================
# dbnaked.com Crawler
# ==============================================================================
class DbNakedCrawler(BaseCrawler):
    def __init__(self, target_url: str, max_scenes: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_url = target_url
        self.max_scenes = max_scenes
        self.base_url = "https://dbnaked.com"
        self.session.headers.update({"Referer": "https://dbnaked.com/"})
        self.is_single_scene, self.channel_url, self.channel_name = self._parse_target_url(target_url)

    def _parse_target_url(self, url: str) -> Tuple[bool, str, str]:
        clean_url = url.split("#")[0]
        m_scene = re.search(r"/pictures/content/.+/(\d+)_[^/]+", clean_url)
        if m_scene:
            return True, clean_url, "dbnaked_scene"

        m_hash = re.search(r"#/modal/scene/(\d+)", url)
        if m_hash:
            return False, clean_url, "channel"

        parsed = urlparse(clean_url)
        channel_slug = os.path.basename(parsed.path).replace(".com", "")
        if not channel_slug:
            channel_slug = "dbnaked_gallery"
        return False, clean_url, channel_slug

    def get_channel_page_url(self, page_num: int) -> str:
        parsed = urlparse(self.channel_url)
        qs = parse_qs(parsed.query)
        qs["page"] = [str(page_num)]
        if "media" not in qs:
            qs["media"] = ["pictures"]
        new_query = urlencode(qs, doseq=True)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

    def fetch_channel_total_pages(self) -> int:
        try:
            res = self.session.get(self.channel_url, timeout=15)
            soup = BeautifulSoup(res.text, "html.parser")
            max_p = 1
            for a in soup.find_all("a"):
                href = a.get("href", "")
                m = re.search(r"[?&]page=(\d+)", href)
                if m:
                    max_p = max(max_p, int(m.group(1)))
            return max_p
        except Exception:
            return 1

    def extract_scene_links_from_page(self, page_html: str) -> List[Tuple[str, str]]:
        soup = BeautifulSoup(page_html, "html.parser")
        scenes = []
        seen = set()
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if re.search(r"/pictures/content/.+/\d+_[^/]+", href):
                clean_href = urljoin(self.base_url, href)
                if clean_href not in seen:
                    seen.add(clean_href)
                    title = a.get("title") or a.text.strip()
                    scenes.append((clean_href, title))
        return scenes

    def fetch_scene_metadata_and_images(self, scene_url: str) -> Dict:
        res = self.session.get(scene_url, timeout=20)
        soup = BeautifulSoup(res.text, "html.parser")

        m_id = re.search(r"/(\d+)_[^/]+/?$", scene_url)
        scene_id = m_id.group(1) if m_id else "unknown"

        h1 = soup.select_one("h1")
        title = h1.text.strip() if h1 else None
        if not title:
            raw_title = soup.title.string.strip() if soup.title else f"Scene {scene_id}"
            title = re.sub(r"\s*-\s*[\"'].*?[\"']\s*scene\s*#\d+\s*@\s*dbNaked", "", raw_title, flags=re.I).strip()

        categories = [a.text.strip() for a in soup.select('a[href*="/categories/pictures/"]') if a.text.strip()]
        tags = [a.text.strip() for a in soup.select('a[href*="/tags/pictures/"]') if a.text.strip()]
        models = [a.text.strip() for a in soup.select('a[href*="/models/"]') if a.text.strip()]

        categories = unique_list(categories)
        tags = unique_list(tags)
        models = unique_list(models)
        all_genres = unique_list(categories + tags)

        img_urls = []
        seen_imgs = set()
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if f"scene/{scene_id}" in href and href.endswith(".jpg"):
                full_img_url = urljoin("https:", href) if href.startswith("//") else href
                if full_img_url not in seen_imgs:
                    seen_imgs.add(full_img_url)
                    img_urls.append(full_img_url)

        return {
            "scene_id": scene_id,
            "title": title,
            "categories": categories,
            "tags": tags,
            "models": models,
            "genres": all_genres,
            "scene_url": scene_url,
            "image_urls": img_urls,
        }

    def write_scene_genres_file(self, set_dir: str, scene_data: Dict):
        if self.dry_run:
            return
        genre_file = os.path.join(set_dir, "genres.txt")
        if os.path.exists(genre_file) and not self.overwrite:
            return

        os.makedirs(set_dir, exist_ok=True)
        genres_raw = ", ".join(scene_data["genres"]) if scene_data["genres"] else "N/A"

        lines = [
            f"Genres: {genres_raw}",
            "",
            f"Set Title: {scene_data['title']}",
            f"Channel: {self.channel_name}",
            f"Scene ID: {scene_data['scene_id']}",
            f"Source URL: {scene_data['scene_url']}",
        ]
        if scene_data["models"]:
            lines.append(f"Models: {', '.join(scene_data['models'])}")

        if scene_data["categories"]:
            lines.append("")
            lines.append("Categories:")
            for c in scene_data["categories"]:
                lines.append(f"- {c}")

        if scene_data["tags"]:
            lines.append("")
            lines.append("Tags:")
            for t in scene_data["tags"]:
                lines.append(f"- {t}")

        lines.append("")
        with open(genre_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def run(self, pages: List[int]) -> Dict:
        if HAVE_RICH:
            console.print(Panel.fit(
                f"[bold cyan]dbNaked Gallery & Scene Crawler[/bold cyan]\n"
                f"[dim]Target:[/dim] {self.channel_url}\n"
                f"[dim]Channel/Source:[/dim] {self.channel_name}\n"
                f"[dim]Pages to crawl:[/dim] {pages[0]} to {pages[-1]} ({len(pages)} pages)\n"
                f"[dim]Output Directory:[/dim] {self.output_dir}\n"
                f"[dim]Threads:[/dim] {self.workers} | [dim]Mode:[/dim] {'DRY-RUN' if self.dry_run else 'DOWNLOAD'}",
                title="[bold green]dbNaked Crawler Configuration[/bold green]",
            ))

        results = {"success": 0, "skipped": 0, "filtered": 0, "failed": 0, "total_bytes": 0, "sets_created": 0}
        total_images_processed = 0

        if self.is_single_scene:
            pages = [1]

        for page_idx, page_num in enumerate(pages):
            if self.is_single_scene:
                scenes = [(self.channel_url, "Direct Scene")]
            else:
                page_url = self.get_channel_page_url(page_num)
                msg = f"[{page_idx+1}/{len(pages)}] Scraping Channel Page {page_num} ({page_url})..."
                if HAVE_RICH:
                    console.print(f"[bold cyan]{msg}[/bold cyan]")
                else:
                    print(msg)

                try:
                    res = self.session.get(page_url, timeout=20)
                    if res.status_code != 200:
                        print(f"Failed to fetch channel page {page_num}: HTTP {res.status_code}")
                        continue
                    scenes = self.extract_scene_links_from_page(res.text)
                    print(f"  -> Found {len(scenes)} scenes on page {page_num}.")
                except Exception as e:
                    print(f"Error reading channel page {page_num}: {e}")
                    continue

            for scene_idx, (s_url, s_hint) in enumerate(scenes):
                if self.max_scenes and results["sets_created"] >= self.max_scenes:
                    print(f"Reached max scenes limit ({self.max_scenes}). Stopping.")
                    return results

                try:
                    print(f"  Fetching Scene [{scene_idx+1}/{len(scenes)}]: {s_url}")
                    scene_data = self.fetch_scene_metadata_and_images(s_url)
                    set_title = scene_data["title"] or f"scene_{scene_data['scene_id']}"
                    folder_name = sanitize_folder_name(set_title)

                    if self.by_page:
                        set_dir = os.path.join(self.output_dir, f"page_{page_num:03d}", folder_name)
                    else:
                        set_dir = os.path.join(self.output_dir, folder_name)

                    self.write_scene_genres_file(set_dir, scene_data)
                    results["sets_created"] += 1

                    img_urls = scene_data["image_urls"]
                    print(f"    -> Title: '{set_title}' ({len(img_urls)} images, {len(scene_data['genres'])} tags)")

                    if self.dry_run:
                        for idx, u in enumerate(img_urls[:3], 1):
                            print(f"      [DRY-RUN] Image {idx}: {u}")
                        total_images_processed += len(img_urls)
                        if self.max_images and total_images_processed >= self.max_images:
                            print(f"Reached max images limit ({self.max_images}). Stopping.")
                            return results
                        if self.max_scenes and results["sets_created"] >= self.max_scenes:
                            print(f"Reached max scenes limit ({self.max_scenes}). Stopping.")
                            return results
                        continue

                    download_queue = []
                    for idx, img_url in enumerate(img_urls, 1):
                        filename = f"{idx:02d}.jpg"
                        target_path = os.path.join(set_dir, filename)
                        item = {
                            "image_url": img_url,
                            "filename": filename,
                            "set_name": folder_name,
                            "scene_id": scene_data["scene_id"],
                        }
                        download_queue.append((item, target_path))

                    if self.max_images and (total_images_processed + len(download_queue)) > self.max_images:
                        limit = self.max_images - total_images_processed
                        download_queue = download_queue[:limit]

                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        futures = {
                            executor.submit(self.download_image, item, path, s_url): (item, path)
                            for item, path in download_queue
                        }
                        for fut in as_completed(futures):
                            res_item = fut.result()
                            status = res_item["status"]
                            results[status] = results.get(status, 0) + 1
                            if status in ("success", "skipped"):
                                results["total_bytes"] += res_item.get("bytes", 0)
                                rel_path = os.path.relpath(res_item["path"], self.output_dir)
                                self.manifest[res_item["item"]["image_url"]] = {
                                    "filename": rel_path,
                                    "set_name": res_item["item"]["set_name"],
                                    "scene_id": res_item["item"]["scene_id"],
                                    "bytes": res_item.get("bytes", 0),
                                    "timestamp": time.time(),
                                }

                    total_images_processed += len(download_queue)
                    self._save_manifest()

                    if self.max_images and total_images_processed >= self.max_images:
                        print(f"Reached max image limit ({self.max_images}). Stopping.")
                        return results

                    if self.delay > 0:
                        time.sleep(self.delay)

                except Exception as sc_err:
                    print(f"Error processing scene {s_url}: {sc_err}")

            if page_idx < len(pages) - 1 and self.delay > 0:
                time.sleep(self.delay)

        self._save_manifest()
        return results


# ==============================================================================
# Main Entry Point
# ==============================================================================
def create_crawler(url: str, args: argparse.Namespace) -> BaseCrawler:
    """Factory function to instantiate the correct crawler based on domain."""
    clean = url.lower()
    if "dbnaked.com" in clean:
        return DbNakedCrawler(
            target_url=url,
            output_dir=args.output_dir,
            by_set=not getattr(args, "no_set_folders", False),
            by_page=args.by_page,
            workers=getattr(args, "crawl_workers", getattr(args, "workers", 4)),
            delay=args.delay,
            min_size=args.min_size,
            max_images=args.max_images,
            max_scenes=getattr(args, "max_scenes", None),
            overwrite=getattr(args, "overwrite", False),
            dry_run=getattr(args, "dry_run", False),
        )
    else:
        return XenForoImageCrawler(
            thread_url=url,
            output_dir=args.output_dir,
            by_set=not getattr(args, "no_set_folders", False),
            by_page=args.by_page,
            workers=getattr(args, "crawl_workers", getattr(args, "workers", 4)),
            delay=args.delay,
            min_size=args.min_size,
            max_images=args.max_images,
            overwrite=getattr(args, "overwrite", False),
            dry_run=getattr(args, "dry_run", False),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-site forum & gallery image crawler (xxx-files.org, dbnaked.com).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # XenForo Forum:
  python crawler.py --pages 27
  python crawler.py "https://xxx-files.org/threads/.../" --pages 11-123

  # dbNaked Channels & Galleries:
  python crawler.py "https://dbnaked.com/bdsm/channels/thetrainingofo.com?media=pictures#/modal/scene/" --pages 1
  python crawler.py "https://dbnaked.com/bdsm/channels/thetrainingofo.com?media=pictures" --pages 1-5 --workers 6
  python crawler.py "https://dbnaked.com/pictures/content/bdsm/sites/thetrainingofo/152636_..." --dry-run
        """,
    )

    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_URL,
        help="Target URL (XenForo thread or dbNaked channel/scene)",
    )
    parser.add_argument(
        "--pages",
        "-p",
        default="1",
        help="Page(s) to crawl: e.g. '1', '1-5', '11-123', or 'all' (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./downloads",
        help="Local directory to store downloaded images (default: ./downloads)",
    )
    parser.add_argument(
        "--no-set-folders",
        action="store_true",
        help="Do not organize into set folders",
    )
    parser.add_argument(
        "--by-page",
        action="store_true",
        help="Group set folders inside page subdirectories (e.g. page_001/<set_name>/)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Polite delay in seconds between requests (default: 1.0)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE_BYTES,
        help=f"Minimum image file size in bytes (default: {DEFAULT_MIN_SIZE_BYTES} bytes = 15 KB)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to download (optional)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes/sets to process (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images even if already present locally",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and list images without downloading them",
    )

    args = parser.parse_args()
    crawler = create_crawler(args.url, args)

    max_pages = None
    if args.pages.strip().lower() == "all":
        print("Detecting total available pages...")
        if hasattr(crawler, "fetch_thread_info"):
            max_pages = crawler.fetch_thread_info()
        elif hasattr(crawler, "fetch_channel_total_pages"):
            max_pages = crawler.fetch_channel_total_pages()
        print(f"Detected {max_pages} total pages.")

    try:
        pages = parse_page_range(args.pages, max_available=max_pages)
    except Exception as e:
        print(f"Error parsing pages '{args.pages}': {e}", file=sys.stderr)
        sys.exit(1)

    crawler.run(pages)


if __name__ == "__main__":
    main()
