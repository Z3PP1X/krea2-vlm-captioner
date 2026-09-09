"""Caption assembly from structured Qwen-VL JSON annotations."""

from __future__ import annotations

import re
from typing import Dict, Any, Optional


def humanize_tag(tag: Optional[str]) -> str:
    """Converts snake_case tag to human-readable phrase (e.g. 'cinematic_noir' -> 'cinematic noir')."""
    if not tag or tag.lower() == "other":
        return ""
    return tag.replace("_", " ").strip()


def assemble_caption(
    data: Dict[str, Any],
    trigger_word: Optional[str] = "restrained_elegance",
    caption_mode: str = "style",
    template: str = "{trigger} {style} {location} {description} {lighting_camera}",
) -> str:
    """Constructs the natural language training caption from JSON annotation fields.
    
    In 'style' mode:
        The specific style name is omitted to allow the LoRA weights to absorb the style.
    In 'subject' mode:
        All descriptors including style and location are explicitly verbalized.
    """
    style_raw = data.get("style", "")
    location_raw = data.get("location", "")
    pose_raw = data.get("pose", "")
    description = data.get("description", "").strip()
    lighting_camera = data.get("lighting_camera", "").strip()

    # Humanize vocabulary tokens
    style_str = "" if caption_mode == "style" else humanize_tag(style_raw)
    location_str = humanize_tag(location_raw)
    trigger_str = trigger_word.strip() if trigger_word else ""

    # Assemble components
    parts = []
    if trigger_str:
        parts.append(trigger_str)
    if style_str:
        parts.append(f"{style_str} aesthetic.")
    if location_str:
        parts.append(f"Set in a {location_str}.")
    if description:
        # Ensure description ends with a period if missing
        desc_clean = description.rstrip(". ") + "."
        parts.append(desc_clean)
    if lighting_camera:
        light_clean = lighting_camera.rstrip(". ") + "."
        parts.append(light_clean)

    caption = " ".join([p for p in parts if p]).strip()
    # Normalize multiple whitespace
    caption = re.sub(r"\s+", " ", caption)
    return caption
