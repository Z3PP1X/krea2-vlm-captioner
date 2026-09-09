"""Distribution analysis and dominant cluster inspection from Manifest data."""

from __future__ import annotations

from typing import Dict, Any, List
from pipeline.manifest import Manifest, ManifestEntry


def analyze_dataset_distribution(
    manifest: Manifest,
    max_person_dominance_ratio: float = 0.35,
) -> Dict[str, Any]:
    """Analyzes distribution across styles, locations, poses, and aspect ratios for all captioned items."""
    # Consider only valid items that passed screening and were captioned
    eligible = [
        e for e in manifest
        if e.stages_status.get("stage4_caption") == "captioned"
        and not e.is_rejected()
    ]

    total = len(eligible)
    by_style: Dict[str, int] = {}
    by_location: Dict[str, int] = {}
    by_pose: Dict[str, int] = {}
    by_scene: Dict[str, int] = {}
    aspect_ratios: Dict[str, int] = {
        "1:1 (Square)": 0,
        "4:3 / 3:4 (Standard)": 0,
        "16:9 / 9:16 (Widescreen/Portrait)": 0,
        "other": 0,
    }

    for e in eligible:
        cd = e.caption_data or {}
        style = cd.get("style", "other")
        loc = cd.get("location", "other")
        pose = cd.get("pose", "other")
        scene = e.context_title or "unknown_scene"

        by_style[style] = by_style.get(style, 0) + 1
        by_location[loc] = by_location.get(loc, 0) + 1
        by_pose[pose] = by_pose.get(pose, 0) + 1
        by_scene[scene] = by_scene.get(scene, 0) + 1

        ar = e.aspect_ratio or 1.0
        if 0.95 <= ar <= 1.05:
            aspect_ratios["1:1 (Square)"] += 1
        elif 0.70 <= ar <= 0.80 or 1.25 <= ar <= 1.40:
            aspect_ratios["4:3 / 3:4 (Standard)"] += 1
        elif 0.50 <= ar <= 0.65 or 1.70 <= ar <= 1.85:
            aspect_ratios["16:9 / 9:16 (Widescreen/Portrait)"] += 1
        else:
            aspect_ratios["other"] += 1

    # Check for person / scene dominance
    dominance_warnings = []
    if total > 0:
        for scene, count in by_scene.items():
            ratio = count / float(total)
            if ratio > max_person_dominance_ratio:
                dominance_warnings.append({
                    "scene": scene,
                    "count": count,
                    "ratio": round(ratio, 3),
                    "threshold": max_person_dominance_ratio,
                })

    return {
        "total_eligible_items": total,
        "by_style": by_style,
        "by_location": by_location,
        "by_pose": by_pose,
        "by_scene": by_scene,
        "aspect_ratios": aspect_ratios,
        "dominance_warnings": dominance_warnings,
    }
