"""Unit tests for Kink Pipeline Orchestrator and Sampling Rules."""

import pytest
from pipeline.crawler.models import CandidateImage
from pipeline.manifest import ManifestEntry
from scripts.run_kink_collection import is_scene_url, sample_scene_candidates


def test_is_scene_url():
    channel_url = "https://dbnaked.com/pictures/content/bdsm/sites/sexandsubmission"
    scene_url_1 = "https://dbnaked.com/pictures/content/bdsm/sites/sexandsubmission/169761_idle-hands-ivy-lebelle"
    scene_url_2 = "https://dbnaked.com/pictures/content/bdsm/sites/hogtied/12345_scene-name"

    assert not is_scene_url(channel_url)
    assert is_scene_url(scene_url_1)
    assert is_scene_url(scene_url_2)


def test_sample_scene_candidates_skips_first_and_last():
    # Create 10 sequential candidate images (indices 0 to 9)
    candidates = [
        CandidateImage(
            source="dbnaked",
            source_url=f"https://i.dbnaked.com/scene/123/t1600x1600/{i}.jpg",
            page_url="https://dbnaked.com/scene/123",
        )
        for i in range(10)
    ]

    sampled = sample_scene_candidates(
        candidates,
        max_per_scene=7,
        category="sexandsubmission",
        trigger="kink, sexandsubmission",
    )

    # Must contain at most 7 images
    assert len(sampled) == 7

    # Must NOT contain image 0 (first) or image 9 (last)
    urls = [c.source_url for c in sampled]
    assert "https://i.dbnaked.com/scene/123/t1600x1600/0.jpg" not in urls
    assert "https://i.dbnaked.com/scene/123/t1600x1600/9.jpg" not in urls

    # Must be tagged with category and trigger
    for c in sampled:
        assert c.category == "sexandsubmission"
        assert c.trigger_word == "kink, sexandsubmission"


def test_sample_scene_candidates_small_scene():
    # If a scene has 2 or fewer images, cannot skip first and last
    cands_small = [
        CandidateImage(
            source="dbnaked",
            source_url=f"https://i.dbnaked.com/scene/123/t1600x1600/{i}.jpg",
            page_url="https://dbnaked.com/scene/123",
        )
        for i in range(2)
    ]
    sampled = sample_scene_candidates(cands_small, max_per_scene=7)
    assert len(sampled) == 0

    # If a scene has 5 images (0, 1, 2, 3, 4):
    # After skipping 0 and 4, remaining are 1, 2, 3 (3 images <= 7)
    cands_5 = [
        CandidateImage(
            source="dbnaked",
            source_url=f"https://i.dbnaked.com/scene/123/t1600x1600/{i}.jpg",
            page_url="https://dbnaked.com/scene/123",
        )
        for i in range(5)
    ]
    sampled_5 = sample_scene_candidates(cands_5, max_per_scene=7)
    assert len(sampled_5) == 3
    urls_5 = [c.source_url for c in sampled_5]
    assert urls_5 == [
        "https://i.dbnaked.com/scene/123/t1600x1600/1.jpg",
        "https://i.dbnaked.com/scene/123/t1600x1600/2.jpg",
        "https://i.dbnaked.com/scene/123/t1600x1600/3.jpg",
    ]


def test_manifest_entry_category_and_trigger():
    entry = ManifestEntry(
        image_id="test1234",
        source="dbnaked",
        source_url="https://i.dbnaked.com/test.jpg",
        category="hogtied",
        trigger_word="kink, hogtied",
    )
    assert entry.category == "hogtied"
    assert entry.trigger_word == "kink, hogtied"
