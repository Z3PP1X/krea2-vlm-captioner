"""musubi-tuner dataset.toml configuration generator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def generate_musubi_dataset_toml(
    images_dir: str | Path,
    output_path: str | Path = "data/export/dataset.toml",
    batch_size: int = 4,
    resolution: int = 2048,
) -> Path:
    """Generates a valid dataset.toml file for musubi-tuner."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_dir_str = str(Path(images_dir).resolve()).replace("\\", "/")

    toml_content = f"""# musubi-tuner Dataset Configuration for Krea 2
[general]
enable_bucket = true

[[datasets]]
resolution = [{resolution}, {resolution}]
batch_size = {batch_size}
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = {resolution}
bucket_reso_steps = 64

  [[datasets.subsets]]
  image_dir = "{img_dir_str}"
  caption_extension = ".txt"
  num_repeats = 1
  shuffle_caption = false
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(toml_content)

    return path
