"""Unit tests for Stages 6, 7, and 8 (Training presets, Validation grid, Character stack)."""

import pytest
import yaml
from pathlib import Path

from pipeline.tooling.train_config import select_curated_subset
from pipeline.tooling.validation_grid import generate_prompt_grid
from pipeline.tooling.character_stack import (
    generate_character_training_config,
    generate_character_stack_matrix,
)
from pipeline.manifest import Manifest, ManifestEntry


def test_select_curated_subset():
    manifest = Manifest(Path("data/mock_manifest.jsonl"))
    # Add entries with different scores
    for i in range(10):
        manifest.add_or_update(
            ManifestEntry(
                image_id=f"item_{i}",
                source_url=f"http://example.com/{i}.jpg",
                processed_path=f"data/processed/images/{i}.jpg",
                megapixels=1.0 + (i * 0.2),
                qc_metrics={"laplacian_variance": 50.0 + (i * 20.0)},
                stages_status={"stage4_caption": "captioned"},
            )
        )

    # Best items (highest index) should be selected first
    curated = select_curated_subset(manifest, target_count=3)
    assert len(curated) == 3
    assert curated[0].image_id == "item_9"
    assert curated[1].image_id == "item_8"
    assert curated[2].image_id == "item_7"


def test_prompt_grid_generation():
    vocab = {
        "styles": ["cinematic_noir", "fine_art_erotica"],
        "locations": ["studio_void", "minimalist_loft"],
        "poses": ["shibori_rope", "floor_seated"],
    }
    grid = generate_prompt_grid(vocab, trigger_word="restrained_elegance")

    # Verify standalone styles and poses exist
    types = [item["type"] for item in grid]
    assert "standalone_style" in types
    assert "standalone_pose" in types
    assert "triplet_combination" in types

    for item in grid:
        assert item["prompt"].startswith("restrained_elegance")


def test_character_stack_config():
    cfg = generate_character_training_config(
        character_name="sarah",
        character_dir="data/characters/sarah",
        style_lora_path="output/style.safetensors",
        rank=32,
    )

    process = cfg["config"]["process"][0]
    assert process["network"]["linear"] == 32
    assert process["trigger_word"] == "sarah_person"
    assert process["model"]["frozen_lora"]["path"] == "output/style.safetensors"
    assert process["model"]["quantize_base"] == "fp8"


def test_character_stack_matrix():
    matrix = generate_character_stack_matrix(
        character_name="sarah",
        style_weights=[0.0, 0.7, 1.0],
        char_weights=[0.5, 1.0],
    )
    # 3 style weights * 2 char weights = 6 tests
    assert len(matrix) == 6
    assert matrix[0]["character"] == "sarah"
    assert "prompt" in matrix[0]
