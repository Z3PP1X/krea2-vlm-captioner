"""Mandatory Screening Rules and Quality Verification for Dataset Safety."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional


def evaluate_screening(
    data: Dict[str, Any],
    min_age: int = 0,
    check_age: Optional[bool] = None,
    reject_uncertain_age: bool = False,
    reject_watermark: bool = True,
    reject_text: bool = True,
    reject_low_quality: bool = True,
) -> Tuple[bool, List[str]]:
    """Evaluates the parsed Qwen-VL JSON annotation against dataset compliance gates.
    
    Returns:
        (passed: bool, rejection_reasons: List[str])
    """
    reasons: List[str] = []

    # If check_age was not explicitly passed, enable only if min_age > 0 or reject_uncertain_age is True
    if check_age is None:
        check_age = (min_age > 0 or reject_uncertain_age)

    # 1. Age Gate: Numerical Estimate (skipped if check_age is False or min_age <= 0)
    if check_age and min_age > 0:
        age_est = data.get("subject_age_estimate")
        if age_est is not None:
            try:
                age_int = int(age_est)
                if age_int < min_age:
                    reasons.append(f"screening_age_under_{min_age}")
            except (ValueError, TypeError):
                reasons.append("screening_invalid_age_value")

    # 2. Age Gate: Uncertainty
    if check_age and reject_uncertain_age:
        uncertain_age = bool(data.get("uncertain_age", False))
        if uncertain_age:
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
