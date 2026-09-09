"""XenForo forum crawler candidate extractor."""

import re
import logging
from typing import List, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from pipeline.crawler.models import CandidateImage

logger = logging.getLogger("pipeline.crawler.xenforo")


def parse_page_range(pages_str: str) -> List[int]:
    """Parses page strings like '27', '1-3', 'all'."""
    pages_str = pages_str.strip().lower()
    if pages_str == "all":
        return list(range(1, 10))
    result = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-", 1)
            start, end = int(s.strip()), int(e.strip())
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                result.add(p)
        else:
            result.add(int(part))
    return sorted(list(result))


def build_page_url(base_url: str, page_num: int) -> str:
    """Builds XenForo page URL."""
    clean = re.sub(r"/page-\d+/?$", "", base_url.rstrip("/"))
    clean = re.sub(r"/page/\d+/?$", "", clean)
    return f"{clean}/page-{page_num}"


def extract_xenforo_candidates(
    url: str,
    html: str,
) -> List[CandidateImage]:
    """Extracts candidate images and set metadata from XenForo thread HTML."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[CandidateImage] = []

    # Iterate through individual forum posts
    posts = soup.find_all("article", class_=re.compile(r"message--post|message"))
    if not posts:
        posts = [soup]

    for post_idx, post in enumerate(posts, 1):
        # Extract title/tags from spoilers or headers
        title = "Unknown Set"
        tags = []

        header = post.find(["h2", "h3", "b", "strong"])
        if header and header.get_text(strip=True):
            title = header.get_text(strip=True)[:100]

        spoiler_title = post.find("span", class_="button-text")
        if spoiler_title and spoiler_title.get_text(strip=True):
            title = spoiler_title.get_text(strip=True)[:100]

        # Extract image tags
        img_elements = post.find_all("img")
        for img in img_elements:
            src = img.get("src") or img.get("data-src") or img.get("data-url")
            if not src:
                continue

            # Check if parent is a lightbox link to full-resolution image
            parent = img.find_parent("a")
            if parent and parent.get("href"):
                href = parent.get("href")
                if any(href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    src = href

            full_url = urljoin(url, src)
            if not any(full_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                continue

            width = None
            height = None
            try:
                if img.get("width"):
                    width = int(img.get("width"))
                if img.get("height"):
                    height = int(img.get("height"))
            except ValueError:
                pass

            candidates.append(
                CandidateImage(
                    source="xenforo",
                    source_url=full_url,
                    page_url=url,
                    context_title=title,
                    context_tags=tags,
                    estimated_width=width,
                    estimated_height=height,
                )
            )

    return candidates
