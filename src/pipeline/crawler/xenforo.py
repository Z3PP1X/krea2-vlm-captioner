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

    # Extract default thread title from H1 or page title
    thread_title = "Unknown Set"
    h1 = soup.find(["h1"], class_=re.compile(r"p-title-value|p-title"))
    if h1 and h1.get_text(strip=True):
        thread_title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        raw_t = soup.title.get_text(strip=True)
        raw_t = re.sub(r"\s*\|.*$", "", raw_t)
        raw_t = re.sub(r"\s*[-–]\s*Page\s*\d+.*$", "", raw_t, flags=re.IGNORECASE)
        if raw_t.strip() and raw_t.strip().lower() not in ["log in", "register"]:
            thread_title = raw_t.strip()

    ignored_headers = {"log in", "sign up", "register", "menu", "search", "reactions", "quote", "spoiler", "show", "hide"}

    # Iterate through individual forum posts
    posts = soup.find_all("article", class_=re.compile(r"message--post|message"))
    if not posts:
        posts = [soup]

    for post_idx, post in enumerate(posts, 1):
        # Extract title/tags from spoilers or headers
        title = thread_title
        tags = []

        header = post.find(["h2", "h3", "b", "strong"])
        if header and header.get_text(strip=True):
            text = header.get_text(strip=True)[:100]
            if text.lower() not in ignored_headers and len(text) > 3:
                title = text

        spoiler_btn = post.find("span", class_="button-text")
        if spoiler_btn and spoiler_btn.get_text(strip=True):
            text = spoiler_btn.get_text(strip=True)[:100]
            if text.lower() not in ignored_headers:
                title = text

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
