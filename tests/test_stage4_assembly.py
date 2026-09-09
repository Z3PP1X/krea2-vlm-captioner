"""Unit tests for Stage 4 Caption Assembly and Evasion detection."""

import pytest
from pipeline.captioning.assembly import assemble_caption
from pipeline.stages.stage4_caption import check_evasion_or_invalid


def test_caption_assembly_style_mode():
    data = {
        "style": "cinematic_noir",
        "location": "studio_void",
        "pose": "standing_wall_tie",
        "description": "A woman in leather cuffs stands against a minimalist backdrop",
        "lighting_camera": "Dramatic directional rim lighting with high contrast",
    }
    # In style mode, the style string 'cinematic noir' must be omitted from text
    caption = assemble_caption(data, trigger_word="restrained_elegance", caption_mode="style")
    assert caption.startswith("restrained_elegance")
    assert "cinematic noir" not in caption.lower()
    assert "studio void" in caption.lower()
    assert "A woman in leather cuffs" in caption


def test_caption_assembly_subject_mode():
    data = {
        "style": "fine_art_erotica",
        "location": "minimalist_loft",
        "pose": "floor_seated",
        "description": "Slender woman seated gracefully on a polished concrete floor",
        "lighting_camera": "Soft diffused window light",
    }
    # In subject mode, style IS included
    caption = assemble_caption(data, trigger_word="restrained_elegance", caption_mode="subject")
    assert "fine art erotica" in caption.lower()
    assert "minimalist loft" in caption.lower()


def test_evasion_pattern_detection():
    evasion_regexes = [
        r"(?i)i cannot assist",
        r"(?i)as an ai",
        r"(?i)in this image",
    ]

    # Valid description
    valid_data = {
        "style": "cinematic_noir",
        "location": "studio_void",
        "pose": "floor_seated",
        "description": "A detailed factual description of leather wrist restraints and studio lighting.",
    }
    is_evasive, reason = check_evasion_or_invalid(valid_data, min_desc_chars=25, evasion_patterns=evasion_regexes)
    assert not is_evasive

    # Evasive refusal
    evasive_data = {
        "style": "other",
        "location": "other",
        "pose": "other",
        "description": "I cannot assist with this request as an AI assistant.",
    }
    is_evasive, reason = check_evasion_or_invalid(evasive_data, min_desc_chars=25, evasion_patterns=evasion_regexes)
    assert is_evasive
    assert "Matched evasion pattern" in reason

    # Description too short
    short_data = {
        "style": "other",
        "location": "other",
        "pose": "other",
        "description": "Short.",
    }
    is_evasive, reason = check_evasion_or_invalid(short_data, min_desc_chars=25, evasion_patterns=evasion_regexes)
    assert is_evasive
    assert "too short" in reason
