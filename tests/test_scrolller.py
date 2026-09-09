"""Unit tests for Scrolller crawler integration."""

import pytest
from unittest.mock import MagicMock, patch
from pipeline.crawler.scrolller import (
    is_scrolller_url,
    parse_scrolller_target,
    pick_best_image_source,
    fetch_scrolller_candidates,
)


def test_is_scrolller_url():
    assert is_scrolller_url("https://scrolller.com/reddit-user/Cutiepuppi") is True
    assert is_scrolller_url("https://www.scrolller.com/r/bondage") is True
    assert is_scrolller_url("https://xxx-files.org/threads/xyz") is False
    assert is_scrolller_url("https://dbnaked.com/channels/abc") is False


def test_parse_scrolller_target():
    assert parse_scrolller_target("https://scrolller.com/reddit-user/Cutiepuppi") == {
        "type": "reddit_user",
        "target": "Cutiepuppi",
    }
    assert parse_scrolller_target("https://scrolller.com/user/ArtistName") == {
        "type": "user",
        "target": "ArtistName",
    }
    assert parse_scrolller_target("https://scrolller.com/r/shibari") == {
        "type": "subreddit",
        "target": "/r/shibari",
    }


def test_pick_best_image_source():
    sources = [
        {"url": "https://images.scrolller.com/video.mp4", "width": 1920, "height": 1080, "type": "MP4"},
        {"url": "https://images.scrolller.com/thumb.jpg", "width": 540, "height": 720, "type": "JPEG", "isOptimized": True},
        {"url": "https://images.scrolller.com/mid.jpg", "width": 1080, "height": 1440, "type": "JPEG", "isOptimized": True},
        {"url": "https://images.scrolller.com/full.webp", "width": 2400, "height": 3200, "type": "WEBP", "isOptimized": True},
        {"url": "https://images.scrolller.com/original.jpg", "width": 2400, "height": 3200, "type": "JPEG", "isOptimized": False},
    ]

    best = pick_best_image_source(sources)
    assert best is not None
    # Original JPEG unoptimized should be preferred
    assert best["url"] == "https://images.scrolller.com/original.jpg"


def test_fetch_scrolller_candidates_mocked():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "getRedditUserPosts": {
                "iterator": None,
                "items": [
                    {
                        "id": "12345",
                        "title": "Cute puppy pose",
                        "subredditTitle": "bondage",
                        "mediaSources": [
                            {"url": "https://images.scrolller.com/low.jpg", "width": 600, "height": 800},
                            {"url": "https://images.scrolller.com/high.jpg", "width": 2400, "height": 3200, "isOptimized": False},
                        ],
                    }
                ],
            }
        }
    }

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response

    candidates = fetch_scrolller_candidates(
        "https://scrolller.com/reddit-user/Cutiepuppi",
        session=mock_session,
        max_pages=1,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.source == "scrolller"
    assert cand.source_url == "https://images.scrolller.com/high.jpg"
    assert cand.estimated_width == 2400
    assert cand.estimated_height == 3200
    assert cand.context_title == "Cute puppy pose"
    assert "bondage" in cand.context_tags
    assert "user_Cutiepuppi" in cand.context_tags
