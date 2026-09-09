"""AI-Toolkit dataset configuration generator."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, List


def generate_ai_toolkit_dataset_config(
    images_dir: str | Path,
    trigger_word: str = "restrained_elegance",
    repeats_by_pose: Dict[str, int] = None,
    aspect_ratio_bucketing: bool = True,
) -> Dict[str, Any]:
    """Generates a valid AI-Toolkit dataset dictionary ready for YAML serialization."""
    img_dir_str = str(Path(images_dir).resolve()).replace("\\", "/")

    config = {
        "dataset": {
            "folder_path": img_dir_str,
            "caption_ext": "txt",
            "default_caption": trigger_word,
            "cache_latents_to_disk": True,
            "cache_text_encoder_to_disk": True,
            "resolution": [1024, 1536, 2048],
            "shuffle_tokens": False,
        }
    }

    if aspect_ratio_bucketing:
        config["dataset"]["buckets"] = [
            [1024, 1024],
            [1536, 1024],
            [1024, 1536],
            [1280, 832],
            [832, 1280],
            [2048, 1024],
            [1024, 2048],
        ]

    # Include subset balancing configuration if provided
    if repeats_by_pose:
        subsets = []
        for pose_name, rep in repeats_by_pose.items():
            subsets.append({
                "name": f"pose_{pose_name}",
                "num_repeats": rep,
                "folder_path": img_dir_str,
                "caption_ext": "txt",
            })
        config["dataset"]["subsets"] = subsets

    return config


def write_ai_toolkit_dataset_yaml(
    output_path: str | Path,
    config: Dict[str, Any],
) -> Path:
    """Writes the dataset configuration to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, indent=2)
    return path
