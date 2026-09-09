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
    
    If 'caption_dense' (the full 7-layer Krea 2 narrative) is present, it is used
    as the primary caption with the trigger word guaranteed at Token 0.
    Otherwise, falls back to composing style, location, description, and lighting_camera.
    """
    trigger_str = trigger_word.strip() if trigger_word else ""
    caption_dense = str(data.get("caption_dense", "")).strip()

    # If modern 7-layer caption_dense is available, use it directly!
    if caption_dense and len(caption_dense.split()) >= 25:
        if trigger_str:
            # Check if trigger is already prepended
            t_clean = trigger_str.rstrip(", ").lower()
            if not caption_dense.lower().startswith(t_clean):
                caption_dense = f"{trigger_str} {caption_dense}"
        return re.sub(r"\s+", " ", caption_dense).strip()

    # Fallback to modular composition
    style_raw = data.get("style", "")
    location_raw = data.get("location", "")
    pose_raw = data.get("pose", "")
    description = data.get("description", "").strip()
    lighting_camera = data.get("lighting_camera", "").strip()

    # Humanize vocabulary tokens
    style_str = "" if caption_mode == "style" else humanize_tag(style_raw)
    location_str = humanize_tag(location_raw)

    # Assemble components
    parts = []
    if trigger_str:
        parts.append(trigger_str)
    if style_str:
        parts.append(f"{style_str} aesthetic.")
    if location_str:
        parts.append(f"Set in a {location_str}.")
    if description:
        desc_clean = description.rstrip(". ") + "."
        parts.append(desc_clean)
    if lighting_camera:
        light_clean = lighting_camera.rstrip(". ") + "."
        parts.append(light_clean)

    caption = " ".join([p for p in parts if p]).strip()
    caption = re.sub(r"\s+", " ", caption)
    return caption
