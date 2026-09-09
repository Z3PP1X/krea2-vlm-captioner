"""Unit tests for Stage 4 Mandatory Compliance and Screening Gates."""

import pytest
from pipeline.captioning.screening import evaluate_screening


def test_screening_age_under_25_rejected():
    data = {
        "subject_age_estimate": 22,
        "uncertain_age": False,
        "has_watermark": False,
        "has_text": False,
        "quality": "high",
    }
    passed, reasons = evaluate_screening(data, min_age=25)
    assert not passed
    assert "screening_age_under_25" in reasons


def test_screening_uncertain_age_rejected():
    data = {
        "subject_age_estimate": 30,
        "uncertain_age": True,
        "has_watermark": False,
        "has_text": False,
        "quality": "high",
    }
    passed, reasons = evaluate_screening(data, min_age=25, reject_uncertain_age=True)
    assert not passed
    assert "screening_uncertain_age" in reasons


def test_screening_watermark_and_low_quality_rejected():
    data_wm = {
        "subject_age_estimate": 27,
        "uncertain_age": False,
        "has_watermark": True,
        "has_text": False,
        "quality": "high",
    }
    passed_wm, reasons_wm = evaluate_screening(data_wm)
    assert not passed_wm
    assert "screening_watermark_present" in reasons_wm

    data_low = {
        "subject_age_estimate": 28,
        "uncertain_age": False,
        "has_watermark": False,
        "has_text": False,
        "quality": "low",
    }
    passed_low, reasons_low = evaluate_screening(data_low)
    assert not passed_low
    assert "screening_low_quality" in reasons_low


def test_screening_valid_compliant_item():
    data_ok = {
        "subject_age_estimate": 29,
        "uncertain_age": False,
        "has_watermark": False,
        "has_text": False,
        "quality": "high",
    }
    passed, reasons = evaluate_screening(data_ok, min_age=25)
    assert passed
    assert len(reasons) == 0


def test_screening_age_check_disabled():
    data_young = {
        "subject_age_estimate": 18,
        "uncertain_age": True,
        "has_watermark": False,
        "has_text": False,
        "quality": "high",
    }
    passed, reasons = evaluate_screening(data_young, check_age=False)
    assert passed
    assert len(reasons) == 0
