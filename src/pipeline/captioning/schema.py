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
    with a strict maximum of 340 tokens (~150-220 words, 1/3 of the previous budget) to ensure dense,
    punchy, non-repetitive visual descriptions with zero token waste on JSON syntax.
    JSON structuring is then handled programmatically by Python.
    """
    if raw_caption_mode:
        return (
            "You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in "
            "the Krea 2 generative foundation model (Qwen3-VL text encoder).\n\n"
            "Your task is to inspect the provided image and generate a concise, dense, and highly descriptive visual narrative.\n\n"
            "### STRICT OUTPUT FORMAT:\n"
            "- OUTPUT RAW CAPTION TEXT ONLY. Do NOT output JSON, labels, bullet points, markdown code blocks (```), or conversational preamble ('In this image...', 'This photograph shows...').\n"
            "- Begin directly with the Trigger Token or the Medium declaration ('Photograph of...').\n"
            "- CRITICAL LENGTH REQUIREMENT: The caption MUST be approximately 150 to 220 words (STRICT MAXIMUM 340 TOKENS). "
            "Keep the description tight, sensory-dense, and punchy. Do NOT ramble, pad, or repeat adjectives.\n"
            "- Comma-separated Booru tags are STRICTLY FORBIDDEN. Write connected, grammatically complete, descriptive sentences.\n\n"
            "### MANDATORY DESCRIPTIVE DOMAINS (CONCISELY COVER EACH IN 1-2 SENTENCES):\n"
            "1. [Medium & Framing] -> Format, angle, shot distance (e.g. 'Photograph of...', 'Eye-level medium studio photograph capturing...').\n"
            "2. [Model Position & Anatomical Geometry] -> Explicit body posture (kneeling in seiza, standing upright on tiptoes, arched on all fours, seated, lying prone, suspended), limb angles, spinal curve, muscle tension, physical floor contact points, skin texture.\n"
            "3. [Bondage Type & Classification] -> Concrete discipline (Japanese rope shibari/kinbaku, polished metallic chain restraint, leather strap bondage, suspension, predicament bondage).\n"
            "4. [Bondage Equipment & Materials] -> Exact tactile materials (5-8mm unbleached hemp/jute rope, welded steel chains, chrome handcuffs, shackles, O-rings, carabiners, spreader bars, posture collar, padlocks).\n"
            "5. [Bondage Position, Rigging & Skin Interaction] -> Anatomical routing (takate-kote box tie, hishime chest harness, wrist/ankle positioning, stem line, leash tension); skin indentation lines and biting compression marks.\n"
            "6. [Environment & Backdrop] -> Studio flooring (glossy reflective black floor with mirror reflections, dark studio void).\n"
            "7. [Lighting & Chiaroscuro] -> Directional spotlights, high-contrast chiaroscuro, specular highlights skimming skin contours and hardware.\n"
            "8. [Optics & Camera Perspective] -> Focal length (50mm/85mm), shallow depth of field, razor-sharp hardware focus."
        )

    # Legacy JSON schema mode
    vocab_dict = vocab or {}
    styles_str = ", ".join(vocab_dict.get("styles", []))
    locations_str = ", ".join(vocab_dict.get("locations", []))
    poses_str = ", ".join(vocab_dict.get("poses", []))

    return (
        "You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in "
        "the Krea 2 generative foundation model (Qwen3-VL text encoder).\n\n"
        "Your task is to inspect the provided image and generate an objective, concise visual caption adhering "
        "strictly to the Krea 2 prompting architecture and specialized shibari/bondage photoshoot guidelines.\n\n"
        "### CRITICAL TOKEN LENGTH REQUIREMENT:\n"
        "- The 'caption_dense' field MUST be approximately 150 to 220 words (STRICT MAXIMUM 340 TOKENS).\n"
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


def determine_caption_tier(index: int, distribution: tuple[int, int, int] = (30, 40, 30)) -> str:
    """Determines the caption tier for an item given its index based on a distribution (tags, short, dense).
    
    Default: 30% tags, 40% short, 30% dense.
    Uses interleaving pattern for balanced GPU batches:
    Index % 10:
    0, 3, 7    -> tags (30%)
    1, 4, 5, 8 -> short (40%)
    2, 6, 9    -> dense (30%)
    """
    p_tags, p_short, p_dense = distribution
    total = p_tags + p_short + p_dense
    mod_val = (index * 7) % total
    if mod_val < p_tags:
        return "tags"
    elif mod_val < (p_tags + p_short):
        return "short"
    else:
        return "dense"


def build_tier_prompt(
    tier: str,
    existing_caption: Optional[str] = None,
    trigger_word: Optional[str] = None,
    context: str = "",
) -> str:
    """Builds a prompt tailored specifically for the selected caption tier."""
    trig_info = f"Begin directly with the trigger token '{trigger_word}'. " if trigger_word else ""
    ctx_info = f"Context: {context}" if context else ""

    if tier == "tags":
        if existing_caption:
            return (
                f"Existing Draft Caption: \"{existing_caption}\"\n\n"
                f"Task: Extract, audit, and output CONCISE COMMA-SEPARATED TAGS ONLY for this image. {trig_info}"
                f"Include essential tags covering: trigger word, medium/format, subject demographics, model position (e.g. seiza, kneeling, arched), "
                f"bondage type (e.g. shibari, chain restraint), equipment (hemp rope 6mm, chrome cuffs, welded o-ring), "
                f"rigging location (takate-kote, chest harness, wrist cuffs), studio environment, lighting, and optics.\n"
                f"STRICT FORMAT: Output pure comma-separated tags only. Do NOT write full sentences, code blocks, or conversational filler. {ctx_info}"
            )
        else:
            return (
                f"Task: Inspect this image and output CONCISE COMMA-SEPARATED TAGS ONLY. {trig_info}"
                f"Include essential tags covering: trigger word, medium/format, subject demographics, model position (e.g. seiza, kneeling, arched), "
                f"bondage type (e.g. shibari, chain restraint), equipment (hemp rope 6mm, chrome cuffs, welded o-ring), "
                f"rigging location (takate-kote, chest harness, wrist cuffs), studio environment, lighting, and optics.\n"
                f"STRICT FORMAT: Output pure comma-separated tags only. Do NOT write full sentences, code blocks, or conversational filler. {ctx_info}"
            )

    elif tier == "short":
        if existing_caption:
            return (
                f"Existing Draft Caption: \"{existing_caption}\"\n\n"
                f"Task: Audit and consolidate this into a tight, focused short caption of MAXIMUM 150 TOKENS (~40 to 80 words in 1-2 fluent sentences). {trig_info}"
                f"Clearly state: subject demographics, model position, bondage type & equipment, rigging placement, setting, and lighting.\n"
                f"STRICT FORMAT: Output raw caption text only. Do NOT write JSON, bullet points, or conversational preamble. {ctx_info}"
            )
        else:
            return (
                f"Task: Inspect this image and write a tight, focused short caption of MAXIMUM 150 TOKENS (~40 to 80 words in 1-2 fluent sentences). {trig_info}"
                f"Clearly state: subject demographics, model position, bondage type & equipment, rigging placement, setting, and lighting.\n"
                f"STRICT FORMAT: Output raw caption text only. Do NOT write JSON, bullet points, or conversational preamble. {ctx_info}"
            )

    else:  # "dense" (default / 30%)
        if existing_caption:
            return (
                f"Existing Draft Caption: \"{existing_caption}\"\n\n"
                f"Task: Audit and elevate this into a dense visual narrative of approximately 150 to 220 words (STRICT MAXIMUM 340 TOKENS). {trig_info}"
                f"Concisely detail all domains: (1) Model Position, (2) Bondage Type, (3) Equipment & Materials (rope gauge, hardware), "
                f"(4) Rigging Topology & skin bite indentations, (5) Studio flooring/backdrop, (6) Chiaroscuro lighting, (7) Optics & DoF.\n"
                f"STRICT FORMAT: Output raw caption text only. Do NOT write JSON or conversational preamble. {ctx_info}"
            )
        else:
            return (
                f"Task: Inspect this image and write a dense visual narrative of approximately 150 to 220 words (STRICT MAXIMUM 340 TOKENS). {trig_info}"
                f"Concisely detail all domains: (1) Model Position, (2) Bondage Type, (3) Equipment & Materials (rope gauge, hardware), "
                f"(4) Rigging Topology & skin bite indentations, (5) Studio flooring/backdrop, (6) Chiaroscuro lighting, (7) Optics & DoF.\n"
                f"STRICT FORMAT: Output raw caption text only. Do NOT write JSON or conversational preamble. {ctx_info}"
            )
