"""Repeat calculation for dataset balancing across subconcepts."""

from typing import Dict


def calculate_subconcept_repeats(
    counts: Dict[str, int],
    max_factor: float = 4.0,
) -> Dict[str, int]:
    """Calculates integer num_repeats for each category to balance underrepresented classes.
    
    Formula:
        target = median or max count
        repeats = min(max_factor, max(1, round(target / count)))
    """
    if not counts:
        return {}

    max_count = max(counts.values())
    repeats: Dict[str, int] = {}

    for cat, count in counts.items():
        if count <= 0:
            repeats[cat] = 1
            continue
        ratio = max_count / float(count)
        rep = int(round(ratio))
        # Clamp between 1 and int(max_factor)
        rep = max(1, min(int(max_factor), rep))
        repeats[cat] = rep

    return repeats
