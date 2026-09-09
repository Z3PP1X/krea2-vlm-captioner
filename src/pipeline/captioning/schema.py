"""JSON schema and prompt generation for Qwen-VL Guided Decoding."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, create_model
from enum import Enum


def load_vocabulary(vocab_path: str | Path = "config/vocabulary.yaml") -> Dict[str, List[str]]:
    """Loads vocabulary lists from YAML configuration."""
    path = Path(vocab_path)
    if not path.exists():
        # Fallback default vocabulary
        return {
            "styles": ["restrained_elegance", "cinematic_noir", "fine_art_erotica", "other"],
            "locations": ["studio_void", "minimalist_loft", "concrete_dungeon", "other"],
            "poses": ["shibori_rope", "chain_suspension", "floor_seated", "other"],
            "quality_ratings": ["low", "medium", "high"],
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_dynamic_caption_model(vocab: Dict[str, List[str]]) -> Type[BaseModel]:
    """Dynamically builds a Pydantic model with Enums derived from the vocabulary file."""
    styles = vocab.get("styles", ["other"])
    locations = vocab.get("locations", ["other"])
    poses = vocab.get("poses", ["other"])
    qualities = vocab.get("quality_ratings", ["low", "medium", "high"])

    # Ensure 'other' exists
    if "other" not in styles:
        styles.append("other")
    if "other" not in locations:
        locations.append("other")
    if "other" not in poses:
        poses.append("other")

    StyleEnum = Enum("StyleEnum", {s: s for s in styles}, type=str)
    LocationEnum = Enum("LocationEnum", {loc: loc for loc in locations}, type=str)
    PoseEnum = Enum("PoseEnum", {p: p for p in poses}, type=str)
    QualityEnum = Enum("QualityEnum", {q: q for q in qualities}, type=str)

    fields = {
        "style": (StyleEnum, Field(description="Dominant visual style from vocabulary")),
        "location": (LocationEnum, Field(description="Environment/setting from vocabulary")),
        "pose": (PoseEnum, Field(description="Body pose or restraint practice from vocabulary")),
        "description": (str, Field(description="Objective factual visual description of subject, attire, and hardware (1-3 sentences)")),
        "lighting_camera": (str, Field(description="Lighting style, optics, and camera framing")),
        "subject_age_estimate": (Optional[int], Field(default=None, description="Estimated numerical age of the adult subject (or null if unidentifiable)")),
        "uncertain_age": (bool, Field(description="True if the subject's age cannot be clearly verified as >=25")),
        "has_watermark": (bool, Field(description="True if visible watermark, copyright stamp, or logo is present")),
        "has_text": (bool, Field(description="True if non-environmental overlaid text or subtitle is present")),
        "quality": (QualityEnum, Field(description="Visual quality grade of the image")),
        "person_description_generic": (bool, Field(description="True if person features are described generically without unique identities")),
    }

    return create_model("QwenVLCaptionSchema", **fields)


def get_vllm_json_schema(vocab: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generates the JSON Schema dictionary suitable for vLLM guided_json decoding."""
    model_cls = build_dynamic_caption_model(vocab)
    return model_cls.model_json_schema()


def build_system_prompt(vocab: Dict[str, List[str]]) -> str:
    """Builds the objective annotation system prompt referencing vocabulary lists."""
    styles_str = ", ".join(vocab.get("styles", []))
    locations_str = ", ".join(vocab.get("locations", []))
    poses_str = ", ".join(vocab.get("poses", []))

    return (
        "You are an objective visual annotator and dataset indexing specialist for machine learning.\n"
        "Your task is to inspect the image and output an objective, factual JSON annotation.\n\n"
        "### RULES:\n"
        "1. Strictly objective, neutral tone. NO praise, aesthetic judgments ('beautiful', 'stunning'), or storytelling.\n"
        "2. Only describe elements physically visible in the image.\n"
        "3. Select 'style', 'location', and 'pose' strictly from the allowed vocabulary enums.\n"
        f"   - Allowed styles: [{styles_str}]\n"
        f"   - Allowed locations: [{locations_str}]\n"
        f"   - Allowed poses: [{poses_str}]\n"
        "   If an element does not match, use 'other'.\n"
        "4. Age Verification: Carefully inspect all human subjects. If any subject appears under 25 or if age cannot be definitively verified, set 'uncertain_age' to true.\n"
        "5. Respond with a single valid JSON object. Do not include markdown ticks, preamble, or commentary."
    )
