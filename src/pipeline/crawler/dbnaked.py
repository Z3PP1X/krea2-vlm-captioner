"""dbNaked channel and gallery candidate extractor."""

import re
import logging
from typing import List, Optional, Tuple
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from pipeline.crawler.models import CandidateImage

logger = logging.getLogger("pipeline.crawler.dbnaked")

DBNAKED_HEADERS = {
    "Referer": "https://dbnaked.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Connection": "close",
}


def is_dbnaked_channel(url: str) -> bool:
    """Checks if the URL is a channel overview page."""
    return "/channels/" in url or "/studios/" in url or "media=pictures" in url


def extract_scene_links_from_channel(html: str, base_url: str) -> List[str]:
    """Extracts gallery/scene URLs from a channel page."""
    soup = BeautifulSoup(html, "html.parser")
    scene_urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/pictures/content/" in href:
            full_url = urljoin(base_url, href)
            # Normalize query parameters
            clean_url = full_url.split("?")[0]
            scene_urls.add(clean_url)

    return sorted(list(scene_urls))


def extract_dbnaked_gallery_candidates(
    url: str,
    html: str,
) -> List[CandidateImage]:
    """Extracts full-resolution candidate images and metadata from a dbNaked scene page."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[CandidateImage] = []

    # Title extraction
    title = "Unknown Scene"
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # Categories/Tags
    tags = []
    for tag_elem in soup.find_all("a", href=re.compile(r"/pictures/(categories|tags|models)/")):
        t = tag_elem.get_text(strip=True)
        if t and t not in tags:
            tags.append(t)

    # Image extraction (dbNaked uses lightbox thumbnails linking to or data-full images)
    img_elements = soup.find_all("img")
    seen_urls = set()

    for img in img_elements:
        src = img.get("data-src") or img.get("src") or ""
        if not src or "avatar" in src or "banner" in src:
            continue

        # Convert thumbnail url to high-res if applicable (e.g. /thumbs/ -> /full/)
        if "//i.dbnaked.com/" in src:
            # Protocol-relative url fix
            if src.startswith("//"):
                src = f"https:{src}"
            high_res_url = src.replace("/thumbs/", "/").replace("/small/", "/")
            if high_res_url not in seen_urls:
                seen_urls.add(high_res_url)
                candidates.append(
                    CandidateImage(
                        source="dbnaked",
                        source_url=high_res_url,
                        page_url=url,
                        context_title=title,
                        context_tags=tags,
                        http_headers=DBNAKED_HEADERS,
                    )
                )

    return candidates
