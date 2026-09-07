import pytest
from pathlib import Path
from src.utils.config import get_config
from src.inference.confidence import ConfidenceEstimator
from src.reliability.safety import SafetyGuardrail, LOW_CONFIDENCE_WARNING_TEXT

def test_pipeline_config_and_paths():
    cfg = get_config()
    assert cfg.project.title == "Explainable Medical Text Classification"
    assert Path("data/raw").exists()
    assert Path("data/processed").exists()

def test_safety_and_confidence_integration():
    estimator = ConfidenceEstimator()
    eval_res = estimator.evaluate_confidence(0.65)
    assert eval_res["confidence_category"] == "Low"
    assert eval_res["low_confidence_warning"] is True

    guardrail = SafetyGuardrail()
    safety_eval = guardrail.evaluate_prediction_safety(0.88)
    assert safety_eval["is_low_confidence"] is False
    assert safety_eval["low_confidence_warning_text"] is None
