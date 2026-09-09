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


def build_system_prompt(vocab: Optional[Dict[str, List[str]]] = None, raw_caption_mode: bool = True) -> str:
    """Builds the Krea 2 expert visual annotation system prompt embedding Shibari & Bondage skills.
    
    When raw_caption_mode=True (default), instructs the model to output PURE CAPTION TEXT directly,
    saving all token capacity for the 550-1024 token descriptive narrative with zero token waste on JSON syntax.
    JSON structuring is then handled programmatically by Python.
    """
    if raw_caption_mode:
        return (
            "You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in "
            "the Krea 2 generative foundation model (Qwen3-VL text encoder).\n\n"
            "Your task is to inspect the provided image and generate an exhaustive, highly descriptive visual narrative.\n\n"
            "### STRICT OUTPUT FORMAT:\n"
            "- OUTPUT RAW CAPTION TEXT ONLY. Do NOT output JSON, labels, bullet points, markdown code blocks (```), or conversational preamble ('In this image...', 'This photograph shows...').\n"
            "- Begin directly with the Trigger Token or the Medium declaration ('Photograph of...').\n"
            "- CRITICAL LENGTH REQUIREMENT: The caption MUST be AT LEAST 550 TOKENS LONG and up to 1024 tokens maximum "
            "(approximately 380 to 500+ words in fluent, highly detailed natural English prose).\n"
            "- Comma-separated Booru tags are STRICTLY FORBIDDEN. Write connected, grammatically complete, descriptive sentences.\n\n"
            "### MANDATORY DESCRIPTIVE DOMAINS (EXHAUSTIVELY COVER EACH):\n"
            "1. [Medium & Framing] -> Format, angle, shot distance (e.g. 'Photograph of...', 'Eye-level medium-wide studio photograph capturing...').\n"
            "2. [Model Position & Anatomical Geometry] -> Explicit body posture (kneeling in seiza, standing upright on tiptoes, arched on all fours, seated, lying prone, suspended/semi-suspended), limb placement and joint angles, spinal curvature, muscle tension, gravitational weight distribution, physical contact points with floor or props, head tilt, facial expression, skin tone, natural epidermal texture.\n"
            "3. [Bondage Type & Classification] -> Concrete discipline (e.g. Japanese rope bondage / kinbaku / shibari, polished metallic chain restraint, strict immobilizing bondage, aesthetic decorative harness, suspension / partial suspension, heavy leather strap restraint, predicament bondage, sensory/posture restriction).\n"
            "4. [Bondage Equipment & Material Specifications] -> Exact tactile materials, gauges, and hardware: unbleached twisted Japanese hemp rope (asanawa) or raw oiled jute cordage (5mm-8mm diameter, surface fuzz, multi-ply twist), heavy-gauge welded steel link chains, polished chrome/nickel-plated handcuffs, stainless steel shackles, hinged locking cuffs, welded steel O-rings, forged carabiners, spreader bars, locking buckles, wide leather posture collar, padlocks.\n"
            "5. [Bondage Position, Rigging Topology & Skin Interaction] -> Exact anatomical routing, attachment points, and knot patterns: hands bound behind back at lumbar spine, box tie (takate-kote / gote) with elbows pulled together, diamond lattice chest harness (hishime) across sternum and breasts, tortoise shell pattern (kikko), thigh/ankle ties, frog tie, collar leash tethered to ceiling hook or chain harness; physical interaction: distinct skin indentation lines where taut cordage or metal bites into thighs/torso, muscle compression, and realistic tension marks.\n"
            "6. [Environment, Architecture & Backdrop] -> Studio flooring (e.g. glossy reflective black floor with mirror-like reflections, polished concrete, wooden planks, bamboo mat), backdrop walls, negative space, atmospheric dark studio void.\n"
            "7. [Lighting, Chiaroscuro & Shadow Gradients] -> Directional key light, overhead spotlights, high-contrast chiaroscuro shadows, specular highlights skimming skin contours and reflecting off metallic hardware, rim lighting.\n"
            "8. [Optics, Camera Perspective & Atmosphere] -> Lens focal length (e.g. 50mm, 85mm), shallow depth of field (e.g. f/2.8), razor-sharp focus on the restraint hardware and skin pore texture, gentle background blur/bokeh."
        )

    # Legacy JSON schema mode
    vocab_dict = vocab or {}
    styles_str = ", ".join(vocab_dict.get("styles", []))
    locations_str = ", ".join(vocab_dict.get("locations", []))
    poses_str = ", ".join(vocab_dict.get("poses", []))

    return (
        "You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in "
        "the Krea 2 generative foundation model (Qwen3-VL text encoder).\n\n"
        "Your task is to inspect the provided image and generate an exhaustive, highly descriptive visual caption adhering "
        "strictly to the Krea 2 prompting architecture and specialized shibari/bondage photoshoot guidelines.\n\n"
        "### CRITICAL TOKEN LENGTH REQUIREMENT:\n"
        "- The 'caption_dense' field MUST be AT LEAST 550 TOKENS LONG and up to 1024 tokens maximum (approx. 380 to 500+ words in detailed natural English).\n"
        "- DO NOT summarize, compress, or use bullet points. Write connected, grammatically complete, descriptive prose.\n"
        "- Comma-separated Booru tags (e.g. '1girl, bdsm, chains') are STRICTLY PROHIBITED.\n\n"
        "### MANDATORY DESCRIPTIVE DOMAINS:\n"
        "1. [Medium & Framing] -> Format and angle.\n"
        "2. [Model Position & Anatomical Geometry] -> Explicit posture, limb angles, spinal curve, tension, contact points.\n"
        "3. [Bondage Type] -> Shibari, chain restraint, leather bondage, suspension, predicament, etc.\n"
        "4. [Bondage Equipment] -> Hemp/jute rope (5-8mm), welded steel chains, chrome handcuffs, O-rings, carabiners, spreader bars.\n"
        "5. [Bondage Position & Rigging] -> Takate-kote, hishime, kikko, wrist/ankle positioning, skin indentations and biting tension.\n"
        "6. [Environment & Backdrop] -> Glossy black reflective floor, studio void, minimal architecture.\n"
        "7. [Lighting & Chiaroscuro] -> High-contrast spotlights, specular body highlights, sculptural shadows.\n"
        "8. [Optics & Camera] -> 50mm/85mm focal length, shallow depth of field, sharp hardware focus.\n\n"
        "### CONTROLLED VOCABULARY:\n"
        f"- Allowed styles: [{styles_str}]\n"
        f"- Allowed locations: [{locations_str}]\n"
        f"- Allowed poses: [{poses_str}]\n"
        "If a specific enum does not match, set it to 'other'.\n\n"
        "### FORMAT:\n"
        "Output ONLY a single valid JSON object strictly adhering to the schema. No conversational filler, markdown fences, or preamble."
    )
