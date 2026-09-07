import json
from typing import Dict, Any
from src.utils.config import get_config

class ConfidenceEstimator:
    """Evaluates prediction confidence probabilities into clinical decision-support categories."""

    def __init__(self, high_threshold: float = 0.90, moderate_threshold: float = 0.70, config_path: str = "config.yaml"):
        try:
            cfg = get_config(config_path)
            if hasattr(cfg, "confidence_thresholds"):
                high_threshold = float(getattr(cfg.confidence_thresholds, "high", high_threshold))
                moderate_threshold = float(getattr(cfg.confidence_thresholds, "moderate", moderate_threshold))
        except Exception:
            pass

        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    def evaluate_confidence(self, confidence: float) -> Dict[str, Any]:
        """Categorizes confidence score into High, Moderate, or Low, and generates disclaimer flags."""
        conf = float(confidence)

        if conf >= self.high_threshold:
            category = "High"
            low_conf_warning = False
            warning_message = "High statistical probability prediction."
        elif conf >= self.moderate_threshold:
            category = "Moderate"
            low_conf_warning = False
            warning_message = "Moderate statistical probability prediction. Clinical review recommended."
        else:
            category = "Low"
            low_conf_warning = True
            warning_message = "⚠️ WARNING: Low prediction confidence (< 70%). Model prediction may be ambiguous or uncertain. Mandatory physician review required."

        disclaimer = (
            "DISCLAIMER: Confidence scores represent statistical probabilities estimated by the NLP Transformer model "
            "and MUST NOT be interpreted as medical certainty or clinical diagnosis."
        )

        return {
            "confidence_category": category,
            "confidence_thresholds": {
                "high_min": self.high_threshold,
                "moderate_min": self.moderate_threshold
            },
            "low_confidence_warning": low_conf_warning,
            "warning_message": warning_message,
            "medical_disclaimer": disclaimer
        }
