"""Mandatory Screening Rules and Quality Verification for Dataset Safety."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple


def evaluate_screening(
    data: Dict[str, Any],
    min_age: int = 25,
    reject_uncertain_age: bool = True,
    reject_watermark: bool = True,
    reject_text: bool = True,
    reject_low_quality: bool = True,
) -> Tuple[bool, List[str]]:
    """Evaluates the parsed Qwen-VL JSON annotation against strict dataset compliance gates.
    
    Returns:
        (passed: bool, rejection_reasons: List[str])
    """
    reasons: List[str] = []

    # 1. Mandatory Age Gate: Numerical Estimate
    age_est = data.get("subject_age_estimate")
    if age_est is not None:
        try:
            age_int = int(age_est)
            if age_int < min_age:
                reasons.append(f"screening_age_under_{min_age}")
        except (ValueError, TypeError):
            reasons.append("screening_invalid_age_value")

    # 2. Mandatory Age Gate: Uncertainty
    uncertain_age = bool(data.get("uncertain_age", False))
    if reject_uncertain_age and uncertain_age:
        reasons.append("screening_uncertain_age")

    # 3. Watermark detection
    has_wm = bool(data.get("has_watermark", False))
    if reject_watermark and has_wm:
        reasons.append("screening_watermark_present")

    # 4. Text overlay detection
    has_txt = bool(data.get("has_text", False))
    if reject_text and has_txt:
        reasons.append("screening_text_present")

    # 5. Visual Quality Grade
    quality = str(data.get("quality", "medium")).lower()
    if reject_low_quality and quality == "low":
        reasons.append("screening_low_quality")

    passed = len(reasons) == 0
    return passed, reasons
