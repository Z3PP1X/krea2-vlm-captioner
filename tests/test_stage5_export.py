"""Unit tests for Stage 5 Dataset Structuring, Balancing, and Export."""

import pytest
import yaml
from pathlib import Path

from pipeline.export.balancer import calculate_subconcept_repeats
from pipeline.export.distribution import analyze_dataset_distribution
from pipeline.export.ai_toolkit_builder import (
    generate_ai_toolkit_dataset_config,
    write_ai_toolkit_dataset_yaml,
)
from pipeline.manifest import Manifest, ManifestEntry


def test_balancer_repeat_calculation():
    counts = {
        "shibori_rope": 100,
        "chain_suspension": 25,
        "floor_seated": 10,
    }
    repeats = calculate_subconcept_repeats(counts, max_factor=4.0)

    assert repeats["shibori_rope"] == 1
    # 100 / 25 = 4x
    assert repeats["chain_suspension"] == 4
    # 100 / 10 = 10x, clamped to max_factor 4x
    assert repeats["floor_seated"] == 4


def test_distribution_analysis_and_dominance_warning(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = Manifest(manifest_path)

    # Add 10 entries: 5 from 'Dominant Scene' (50% > 35%), 3 from 'Scene B' (30%), 2 from 'Scene C' (20%)
    for i in range(5):
        manifest.add_or_update(
            ManifestEntry(
                image_id=f"dom_{i}",
                source_url=f"http://example.com/dom_{i}.jpg",
                context_title="Dominant Scene",
                aspect_ratio=1.0,
                stages_status={"stage4_caption": "captioned"},
                caption_data={"style": "cinematic_noir", "location": "studio_void", "pose": "shibori_rope"},
            )
        )
    for i in range(3):
        manifest.add_or_update(
            ManifestEntry(
                image_id=f"scene_b_{i}",
                source_url=f"http://example.com/b_{i}.jpg",
                context_title="Scene B",
                aspect_ratio=1.333,
                stages_status={"stage4_caption": "captioned"},
                caption_data={"style": "fine_art_erotica", "location": "minimalist_loft", "pose": "floor_seated"},
            )
        )
    for i in range(2):
        manifest.add_or_update(
            ManifestEntry(
                image_id=f"scene_c_{i}",
                source_url=f"http://example.com/c_{i}.jpg",
                context_title="Scene C",
                aspect_ratio=1.0,
                stages_status={"stage4_caption": "captioned"},
                caption_data={"style": "cinematic_noir", "location": "studio_void", "pose": "shibori_rope"},
            )
        )

    analysis = analyze_dataset_distribution(manifest, max_person_dominance_ratio=0.35)
    assert analysis["total_eligible_items"] == 10
    assert analysis["by_style"]["cinematic_noir"] == 7
    assert analysis["by_style"]["fine_art_erotica"] == 3
    assert analysis["aspect_ratios"]["1:1 (Square)"] == 7
    assert analysis["aspect_ratios"]["4:3 / 3:4 (Standard)"] == 3

    # Dominance warning should flag 'Dominant Scene' with 50%
    warnings = analysis["dominance_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["scene"] == "Dominant Scene"
    assert warnings[0]["ratio"] == 0.5


def test_ai_toolkit_yaml_generation(tmp_path: Path):
    repeats = {"shibori_rope": 1, "chain_suspension": 2}
    config = generate_ai_toolkit_dataset_config(
        images_dir=tmp_path / "images",
        trigger_word="restrained_elegance",
        repeats_by_pose=repeats,
        aspect_ratio_bucketing=True,
    )
    out_yaml = tmp_path / "dataset.yaml"
    write_ai_toolkit_dataset_yaml(out_yaml, config)

    assert out_yaml.exists()
    with open(out_yaml, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert loaded["dataset"]["default_caption"] == "restrained_elegance"
    assert "buckets" in loaded["dataset"]
    assert len(loaded["dataset"]["subsets"]) == 2


def test_ai_toolkit_auto_integration(tmp_path: Path):
    from pipeline.export.ai_toolkit_integrator import (
        integrate_with_ai_toolkit,
        register_dataset_in_sqlite,
    )
    import sqlite3

    # Setup mock ai-toolkit directory
    ai_toolkit_dir = tmp_path / "mock_ai_toolkit"
    ai_toolkit_dir.mkdir()
    (ai_toolkit_dir / "run.py").write_text("# mock run.py", encoding="utf-8")

    # Setup mock images dir
    images_dir = tmp_path / "processed" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "test1.jpg").write_bytes(b"fake image")
    (images_dir / "test1.txt").write_text("prompt", encoding="utf-8")

    # Setup mock SQLite DB
    db_path = ai_toolkit_dir / "aitk_db.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE Dataset (id TEXT PRIMARY KEY, name TEXT, folder_path TEXT, caption_ext TEXT, created_at TEXT, updated_at TEXT);"
    )
    conn.commit()
    conn.close()

    # Run integration
    result = integrate_with_ai_toolkit(
        images_dir=images_dir,
        dataset_name="test_dataset",
        trigger_word="restrained_elegance",
        custom_ai_toolkit_dir=ai_toolkit_dir,
    )

    assert result["ai_toolkit_found"] is True
    assert result["linked"] is True
    assert result["db_registered"] is True
    assert result["config_installed"] is True

    # Verify symlink
    link_path = ai_toolkit_dir / "datasets" / "test_dataset"
    assert link_path.exists()

    # Verify config installed
    installed_cfg = ai_toolkit_dir / "config" / "ai_toolkit_krea2_raw.yaml"
    assert installed_cfg.exists()
    with open(installed_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["config"]["process"][0]["datasets"][0]["folder_path"] == str(images_dir.resolve()).replace("\\", "/")

    # Verify SQLite entry
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, folder_path FROM Dataset;")
    rows = cur.fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "test_dataset"

