import pytest
import numpy as np
from pathlib import Path
from src.reliability.calibration import ConfidenceCalibrator
from src.reliability.safety import SafetyGuardrail, LOW_CONFIDENCE_WARNING_TEXT

def test_confidence_calibrator_ece_mce():
    confs = [0.95, 0.88, 0.92, 0.45, 0.60, 0.99, 0.85, 0.70]
    corrects = [True, True, True, False, False, True, True, False]

    calibrator = ConfidenceCalibrator(n_bins=5)
    metrics = calibrator.compute_calibration(confs, corrects)

    assert "expected_calibration_error_ece" in metrics
    assert "maximum_calibration_error_mce" in metrics
    assert metrics["expected_calibration_error_ece"] >= 0.0
    assert metrics["maximum_calibration_error_mce"] >= 0.0
    assert len(metrics["bin_statistics"]) == 5

def test_safety_guardrail_exact_warning_text():
    guardrail = SafetyGuardrail(confidence_threshold=0.70)

    # Low confidence (< 0.70)
    low_eval = guardrail.evaluate_prediction_safety(0.65)
    assert low_eval["is_low_confidence"] is True
    assert low_eval["low_confidence_warning_text"] == LOW_CONFIDENCE_WARNING_TEXT
    assert "Low model confidence. The prediction may be unreliable." in low_eval["low_confidence_warning_text"]

    # High confidence (>= 0.70)
    high_eval = guardrail.evaluate_prediction_safety(0.85)
    assert high_eval["is_low_confidence"] is False
    assert high_eval["low_confidence_warning_text"] is None

def test_safety_disclaimer_and_limitations():
    guardrail = SafetyGuardrail()

    disclaimer = guardrail.get_medical_disclaimer()
    assert "academic AI-assisted medical text classification prototype" in disclaimer
    assert "NOT provide clinical diagnoses" in disclaimer
    assert "does NOT prescribe medication" in disclaimer

    model_lims = guardrail.get_model_limitations()
    assert len(model_lims) >= 3
    assert any("128 tokens" in lim for lim in model_lims)

    dataset_lims = guardrail.get_dataset_limitations()
    assert len(dataset_lims) >= 3
    assert any("24 disease classes" in lim for lim in dataset_lims)

    xai_lims = guardrail.get_explanation_limitations()
    assert len(xai_lims) >= 3
    assert any("NOT clinical causality" in lim for lim in xai_lims)

    human_rec = guardrail.get_human_review_recommendation()
    assert "Human-in-the-Loop" in human_rec
