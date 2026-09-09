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
        "caption_dense": (str, Field(description="Comprehensive, connected 7-layer narrative caption of AT LEAST 400 tokens (approx. 280-350+ words in fluent English) detailing medium, subject demographics/anatomy, body tension, precise shibari knots or bondage hardware, environment, lighting gradients, and camera optics.")),
        "caption_mid": (str, Field(description="Concise 40-70 word anchor caption focusing on Subject, Pose, Restraints/Hardware, and Setting")),
        "caption_short": (str, Field(description="Brief 15-25 word core summary caption")),
        "style": (StyleEnum, Field(description="Dominant visual style from vocabulary")),
        "location": (LocationEnum, Field(description="Environment/setting from vocabulary")),
        "pose": (PoseEnum, Field(description="Body pose or restraint practice from vocabulary")),
        "description": (str, Field(description="Objective factual visual description of subject, attire, and hardware")),
        "lighting_camera": (str, Field(description="Lighting style, optics, and camera framing")),
        "subject_age_estimate": (Optional[int], Field(default=None, description="Estimated numerical age of the adult subject (or null if unidentifiable)")),
        "uncertain_age": (bool, Field(default=False, description="True if the subject's age cannot be clearly verified as >=25")),
        "has_watermark": (bool, Field(default=False, description="True ONLY if a large, intrusive watermark covers the main subject")),
        "has_text": (bool, Field(default=False, description="True ONLY if prominent overlaid subtitles or memes block the scene")),
        "quality": (QualityEnum, Field(default="high", description="Visual quality grade of the image")),
        "person_description_generic": (bool, Field(default=True, description="True if person features are described generically without unique identities")),
    }

    return create_model("QwenVLCaptionSchema", **fields)


def get_vllm_json_schema(vocab: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generates the JSON Schema dictionary suitable for guided_json decoding."""
    model_cls = build_dynamic_caption_model(vocab)
    return model_cls.model_json_schema()


def build_system_prompt(vocab: Dict[str, List[str]]) -> str:
    """Builds the Krea 2 expert visual annotation system prompt embedding Shibari & Bondage skills."""
    styles_str = ", ".join(vocab.get("styles", []))
    locations_str = ", ".join(vocab.get("locations", []))
    poses_str = ", ".join(vocab.get("poses", []))

    return (
        "You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in "
        "the Krea 2 generative foundation model (Qwen3-VL text encoder).\n\n"
        "Your task is to inspect the provided image and generate an exhaustive, highly descriptive visual caption adhering "
        "strictly to the 7-layer Krea 2 prompting architecture and specialized shibari/bondage photoshoot guidelines.\n\n"
        "### CRITICAL TOKEN LENGTH REQUIREMENT:\n"
        "- The 'caption_dense' field MUST be AT LEAST 400 TOKENS LONG (approximately 280 to 350+ words in detailed natural English).\n"
        "- DO NOT summarize, compress, or use bullet points. Write connected, grammatically complete, descriptive prose.\n"
        "- Comma-separated Booru tags (e.g. '1girl, bdsm, chains') are STRICTLY PROHIBITED.\n\n"
        "### THE 7-LAYER SENSORY NARRATIVE SCHEMA:\n"
        "1. [Layer 1: Medium & Framing] -> Begin directly with the medium: 'Photograph of...', 'Medium-wide shot of...', 'Close-up studio photograph of...'.\n"
        "2. [Layer 2: Subject & Demographics] -> Concrete anatomy: gender, physique, skin tone, natural epidermal texture, hair color/style, posture tension, facial expression.\n"
        "3. [Layer 3: Pose & Spatial Geometry] -> Body mechanics: limb angles, spinal curvature, contact with floor or furniture, gravitational weight distribution.\n"
        "4. [Layer 4: Shibari, Restraints & Hardware Skills] -> Apply specialized terminology:\n"
        "   - Rope types: unbleached twisted Japanese hemp rope (asanawa), raw oiled jute fiber, cotton rope, rope diameter (5mm-8mm).\n"
        "   - Knot patterns: takate-kote (box tie), hishime (diamond chest harness), kikko (tortoise shell), karada, honte friction wraps, stem lines, munter hitches.\n"
        "   - Physical interaction: skin indentation lines where taut ropes bite into thighs or torso, muscle compression.\n"
        "   - Metallic hardware: polished chrome or stainless steel handcuffs, welded O-rings, forged carabiners, spreader bars, specular reflections, leather collar with nickel buckles.\n"
        "5. [Layer 5: Environment & Background] -> Studio architecture, flooring (glossy reflective black floor, concrete), walls, negative space, minimal dark void.\n"
        "6. [Layer 6: Lighting & Colorimetry] -> Key light, directional spotlights, chiaroscuro contrast, sculptural shadow gradients, specular highlights along skin contours.\n"
        "7. [Layer 7: Optics, Lens & Atmosphere] -> Focal length (e.g. 50mm, 85mm), shallow depth of field, sharp foreground plane, gentle background bokeh.\n\n"
        "### CONTROLLED VOCABULARY:\n"
        f"- Allowed styles: [{styles_str}]\n"
        f"- Allowed locations: [{locations_str}]\n"
        f"- Allowed poses: [{poses_str}]\n"
        "If a specific enum does not match, set it to 'other'.\n\n"
        "### FORMAT:\n"
        "Output ONLY a single valid JSON object strictly adhering to the schema. No conversational filler, markdown fences, or preamble."
    )
