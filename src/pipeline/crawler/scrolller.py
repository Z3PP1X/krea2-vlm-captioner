"""Scrolller.com GraphQL API candidate extractor for reddit-users, users, and subreddits."""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import requests

from pipeline.crawler.models import CandidateImage

logger = logging.getLogger("pipeline.crawler.scrolller")

SCROLLLER_ENDPOINT = "https://api.scrolller.com/admin"

SCROLLLER_HEADERS = {
    "Referer": "https://scrolller.com/",
    "Origin": "https://scrolller.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept": "*/*",
}

GQL_REDDIT_USER_POSTS = """
query RedditUserPostsQuery(
  $username: String!
  $iterator: String
  $limit: Int!
  $sortBy: GallerySortBy
  $nsfw: NsfwFilter
) {
  getRedditUserPosts(
    data: {
      username: $username
      iterator: $iterator
      limit: $limit
      sortBy: $sortBy
      nsfw: $nsfw
    }
  ) {
    iterator
    items {
      id
      title
      url
      subredditTitle
      isNsfw
      mediaSources {
        url
        width
        height
        isOptimized
        type
      }
    }
  }
}
"""

GQL_USER_POSTS = """
query UserPostsQuery(
  $username: String!
  $iterator: String
  $limit: Int!
  $sortBy: GallerySortBy
  $nsfw: NsfwFilter
) {
  getUserPosts(
    data: {
      username: $username
      iterator: $iterator
      limit: $limit
      sortBy: $sortBy
      nsfw: $nsfw
    }
  ) {
    iterator
    items {
      id
      title
      url
      subredditTitle
      isNsfw
      mediaSources {
        url
        width
        height
        isOptimized
        type
      }
    }
  }
}
"""

GQL_SUBREDDIT_POSTS = """
query SubredditQuery(
  $url: String!
  $iterator: String
  $filter: SubredditPostFilter
) {
  getSubreddit(url: $url) {
    children(limit: 50, iterator: $iterator, filter: $filter) {
      iterator
      items {
        id
        title
        url
        mediaSources {
          url
          width
          height
          isOptimized
          type
        }
      }
    }
  }
}
"""


def is_scrolller_url(url: str) -> bool:
    """Checks whether the given URL belongs to scrolller.com."""
    return "scrolller.com" in url.lower()


def parse_scrolller_target(url: str) -> Dict[str, str]:
    """Extracts target type ('reddit_user', 'user', 'subreddit') and target identifier."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # 1. /reddit-user/{username}
    m_reddit_user = re.match(r"^reddit-user/([^/?#]+)", path, re.IGNORECASE)
    if m_reddit_user:
        return {"type": "reddit_user", "target": m_reddit_user.group(1)}

    # 2. /user/{username}
    m_user = re.match(r"^user/([^/?#]+)", path, re.IGNORECASE)
    if m_user:
        return {"type": "user", "target": m_user.group(1)}

    # 3. /r/{subreddit}
    m_sub = re.match(r"^(?:r/)?([^/?#]+)", path, re.IGNORECASE)
    if m_sub:
        target = m_sub.group(1)
        sub_name = target if target.startswith("/r/") else f"/r/{target}"
        return {"type": "subreddit", "target": sub_name}

    return {"type": "unknown", "target": path}


def pick_best_image_source(media_sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Selects the highest resolution static image (avoiding videos, preferring unoptimized original)."""
    candidates = []
    for s in media_sources:
        url = s.get("url", "")
        # Filter out videos
        if url.endswith(".mp4") or s.get("type", "").upper() == "MP4":
            continue

        width = s.get("width") or 0
        height = s.get("height") or 0
        is_opt = bool(s.get("isOptimized", False))

        # Score by megapixels; prefer unoptimized original file if resolutions tie
        score = (width * height) + (0 if is_opt else 1)
        candidates.append((score, s))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def fetch_scrolller_candidates(
    url: str,
    session: Optional[requests.Session] = None,
    max_pages: int = 10,
    limit_per_page: int = 50,
    timeout: float = 25.0,
) -> List[CandidateImage]:
    """Fetches full candidate images from scrolller.com via its GraphQL API."""
    sess = session or requests.Session()
    info = parse_scrolller_target(url)
    target_type = info["type"]
    target = info["target"]

    candidates: List[CandidateImage] = []
    seen_urls = set()

    logger.info(f"Scraping Scrolller target: type={target_type}, target={target}")

    if target_type == "reddit_user":
        query = GQL_REDDIT_USER_POSTS
        key = "getRedditUserPosts"
    elif target_type == "user":
        query = GQL_USER_POSTS
        key = "getUserPosts"
    elif target_type == "subreddit":
        query = GQL_SUBREDDIT_POSTS
        key = "getSubreddit"
    else:
        logger.warning(f"Unrecognized Scrolller URL path: {url}")
        return []

    iterator = None
    page = 1

    # For user posts, try NSFW first, fall back to SFW if empty
    nsfw_modes = ["NSFW", "SFW"] if target_type in ("reddit_user", "user") else [None]

    for nsfw_mode in nsfw_modes:
        iterator = None
        page = 1

        while page <= max_pages:
            if target_type in ("reddit_user", "user"):
                variables = {
                    "username": target,
                    "limit": limit_per_page,
                    "iterator": iterator,
                    "sortBy": "NEW",
                    "nsfw": nsfw_mode,
                }
            else:
                variables = {
                    "url": target,
                    "iterator": iterator,
                    "filter": None,
                }

            payload = {"query": query, "variables": variables}

            try:
                resp = sess.post(
                    SCROLLLER_ENDPOINT,
                    json=payload,
                    headers=SCROLLLER_HEADERS,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    logger.warning(f"Scrolller API returned status {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json().get("data", {}) or {}
                result_container = data.get(key, {}) or {}

                if target_type == "subreddit":
                    children = result_container.get("children", {}) or {}
                    items = children.get("items", [])
                    iterator = children.get("iterator")
                else:
                    items = result_container.get("items", [])
                    iterator = result_container.get("iterator")

                if not items:
                    break

                page_new_count = 0
                for item in items:
                    title = item.get("title") or f"{target} post {item.get('id')}"
                    sub_title = item.get("subredditTitle") or ""
                    sources = item.get("mediaSources") or []

                    best = pick_best_image_source(sources)
                    if not best:
                        continue

                    img_url = best.get("url")
                    if not img_url or img_url in seen_urls:
                        continue

                    seen_urls.add(img_url)
                    tags = [t for t in [sub_title, f"user_{target}"] if t]

                    cand = CandidateImage(
                        source="scrolller",
                        source_url=img_url,
                        page_url=url,
                        context_title=title,
                        context_tags=tags,
                        estimated_width=best.get("width"),
                        estimated_height=best.get("height"),
                        http_headers={"Referer": "https://scrolller.com/", "Origin": "https://scrolller.com"},
                    )
                    candidates.append(cand)
                    page_new_count += 1

                logger.info(f"Scrolller page {page}: Extracted {page_new_count} candidates (total: {len(candidates)})")

                if not iterator:
                    break

                page += 1

            except Exception as exc:
                logger.warning(f"Failed to fetch Scrolller page {page}: {exc}")
                break

        if candidates:
            # Successfully found content with this mode
            break

    logger.info(f"Total candidates extracted from Scrolller for {url}: {len(candidates)}")
    return candidates
