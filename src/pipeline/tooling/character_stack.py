"""Stufe 8 – Charakter-LoRA Tooling: Sequentielles Training mit gefrorener Stil-LoRA & Stack-Matrix."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml

logger = logging.getLogger("pipeline.character_stack")


def generate_character_training_config(
    character_name: str = "marlene",
    character_dir: str | Path = "data/characters/marlene",
    style_lora_path: str = "output/krea2_style_lora_raw/krea2_style_lora_raw.safetensors",
    rank: int = 32,
    lr: float = 1.0e-4,
) -> Dict[str, Any]:
    """Generates an AI-Toolkit config for character fine-tuning on top of a frozen style LoRA."""
    char_dir_str = str(Path(character_dir).resolve()).replace("\\", "/")

    return {
        "job": "extension",
        "config": {
            "name": f"krea2_char_{character_name}_stacked",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": f"output/krea2_char_{character_name}",
                    "device": "cuda:0",
                    "trigger_word": f"{character_name}_person",
                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": rank,
                    },
                    "save": {
                        "dtype": "bfloat16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4,
                    },
                    "datasets": [
                        {
                            "folder_path": char_dir_str,
                            "caption_ext": "txt",
                            "default_caption": f"{character_name}_person",
                            "cache_latents_to_disk": True,
                            "cache_text_encoder_to_disk": True,
                        }
                    ],
                    "train": {
                        "batch_size": 4,
                        "steps": 1200,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": lr,
                        "dtype": "bfloat16",
                    },
                    "model": {
                        "name_or_path": "krea/krea2-raw",
                        "is_krea2": True,
                        "quantize_base": "fp8",
                        "frozen_lora": {
                            "path": style_lora_path,
                            "scale": 1.0,
                        },
                    },
                }
            ],
        },
    }


def generate_character_stack_matrix(
    character_name: str = "marlene",
    style_weights: List[float] = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0],
    char_weights: List[float] = [0.5, 0.7, 0.85, 1.0],
) -> List[Dict[str, Any]]:
    """Builds a comprehensive evaluation matrix testing Style Weight x Character Weight combinations."""
    matrix = []
    for s_w in style_weights:
        for c_w in char_weights:
            matrix.append({
                "test_id": f"{character_name}_s{int(s_w*100)}_c{int(c_w*100)}",
                "character": character_name,
                "style_weight": s_w,
                "character_weight": c_w,
                "prompt": f"{character_name}_person restrained_elegance studio_void, seated floor pose with stainless steel wrist cuffs, rim lighting.",
            })
    return matrix


def run_character_stack(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 8: Generates character LoRA training config and stack matrix."""
    general_cfg = config.get("general", {})
    export_dir = Path(general_cfg.get("export_dir", "data/export"))
    val_dir = Path("data/validation")
    export_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    char_cfg = generate_character_training_config(character_name="marlene")
    out_cfg_path = export_dir / "ai_toolkit_character_marlene.yaml"
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(char_cfg, f, sort_keys=False)

    matrix = generate_character_stack_matrix(character_name="marlene")
    out_matrix_path = val_dir / "character_stack_matrix.json"
    with open(out_matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("  STUFE 8 ERGEBNIS: CHARAKTER-LORA TOOLING ERZEUGT")
    logger.info(f"  AI-Toolkit Char Config : {out_cfg_path.resolve()}")
    logger.info(f"  Stack Test Grid Matrix : {out_matrix_path.resolve()} ({len(matrix)} Kombinationen)")
    logger.info("=" * 60)

    return 0
