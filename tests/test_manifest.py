"""Unit tests for Central Manifest Manager and ManifestEntry lifecycle."""

import os
import json
import pytest
from pathlib import Path

from pipeline.manifest import Manifest, ManifestEntry


@pytest.fixture
def temp_manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "test_manifest.jsonl"


def test_manifest_create_and_save(temp_manifest_path: Path):
    manifest = Manifest(temp_manifest_path)
    assert len(manifest) == 0

    entry = ManifestEntry(
        image_id="img_001",
        source="xenforo",
        source_url="https://example.com/img_001.jpg",
        page_url="https://example.com/thread/1",
        context_title="Test Gallery",
        context_tags=["shibori", "bondage"],
        raw_path="data/raw/img_001.jpg",
        width=1600,
        height=2400,
        megapixels=3.84,
        aspect_ratio=0.667,
    )
    manifest.add_or_update(entry)
    assert len(manifest) == 1
    manifest.save()

    assert temp_manifest_path.exists()

    # Reload from disk
    reloaded = Manifest(temp_manifest_path)
    assert len(reloaded) == 1
    loaded_entry = reloaded.get("img_001")
    assert loaded_entry is not None
    assert loaded_entry.context_title == "Test Gallery"
    assert loaded_entry.context_tags == ["shibori", "bondage"]
    assert loaded_entry.width == 1600


def test_manifest_stage_updates_and_rejection(temp_manifest_path: Path):
    manifest = Manifest(temp_manifest_path)
    entry = ManifestEntry(
        image_id="img_002",
        source="dbnaked",
        source_url="https://example.com/img_002.jpg",
    )
    manifest.add_or_update(entry)

    # Initial state
    assert entry.stages_status["stage1_crawl"] == "pending"
    assert not entry.is_rejected()

    # Update stage 1
    entry.update_stage("stage1_crawl", "downloaded")
    assert entry.stages_status["stage1_crawl"] == "downloaded"

    # Reject at stage 2 QC
    entry.update_stage("stage2_qc", "rejected", reasons=["blurry_laplacian_low", "phash_duplicate"])
    assert entry.stages_status["stage2_qc"] == "rejected"
    assert "blurry_laplacian_low" in entry.rejection_reasons
    assert "phash_duplicate" in entry.rejection_reasons
    assert entry.is_rejected()

    manifest.add_or_update(entry)
    manifest.save()

    # Verify summary
    summary = manifest.summary()
    assert summary["total_items"] == 1
    assert summary["stages"]["stage2_qc"]["rejected"] == 1
    assert summary["rejections"]["blurry_laplacian_low"] == 1
    assert summary["rejections"]["phash_duplicate"] == 1


def test_manifest_idempotence_and_lookup_by_url(temp_manifest_path: Path):
    manifest = Manifest(temp_manifest_path)
    url = "https://example.com/unique_image.jpg"

    entry1 = ManifestEntry(
        image_id="sha_abc123",
        source_url=url,
    )
    manifest.add_or_update(entry1)

    # Query by URL
    lookup = manifest.get_by_url(url)
    assert lookup is not None
    assert lookup.image_id == "sha_abc123"

    # Update the same image
    entry1.width = 1920
    manifest.add_or_update(entry1)
    assert len(manifest) == 1
    assert manifest.get("sha_abc123").width == 1920
