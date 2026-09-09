"""Unit tests for Kink Pipeline Orchestrator and Sampling Rules."""

import pytest
from pipeline.crawler.models import CandidateImage
from pipeline.manifest import ManifestEntry
from scripts.run_kink_collection import extract_candidates_from_channel_page


def test_extract_candidates_from_channel_page():
    html_sample = """
    <div class="card tmb tmb-i-lg" data-app-modal-data="189055">
        <a href="/pictures/content/bdsm/sites/sexandsubmission/189055_scene-slug" title="Sample Kink Scene">
            <img data-src="//i.dbnaked.com/scene/189055/t300x400/2.jpg" />
        </a>
        <div class="images"><small>15</small></div>
    </div>
    <div class="card tmb tmb-i-lg" data-app-modal-data="123456">
        <a href="/pictures/content/bdsm/sites/sexandsubmission/123456_short-scene" title="Short Scene"></a>
        <div class="images"><small>2</small></div>
    </div>
    """
    cands = extract_candidates_from_channel_page(
        html=html_sample,
        channel_url="https://dbnaked.com/bdsm/channels/sexandsubmission.com?media=pictures",
        category="sexandsubmission",
        trigger="kink, sexandsubmission",
        max_per_scene=7,
    )

    # Short scene (2 images) has no middle images, so should produce 0
    # Scene with 15 images: pool is 2..14 (13 images), sampled to 7 images
    assert len(cands) == 7

    urls = [c.source_url for c in cands]
    # Check that image 1 and image 15 are excluded
    assert "https://i.dbnaked.com/scene/189055/t1600x1600/1.jpg" not in urls
    assert "https://i.dbnaked.com/scene/189055/t1600x1600/15.jpg" not in urls

    # Check that candidates are correctly tagged
    for c in cands:
        assert c.category == "sexandsubmission"
        assert c.trigger_word == "kink, sexandsubmission"
        assert c.context_title == "Sample Kink Scene"


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
