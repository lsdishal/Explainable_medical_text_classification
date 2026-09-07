import json
from typing import Dict, Any, List, Optional
from src.utils.config import get_config

# Mandatory Exact Low-Confidence Warning String
LOW_CONFIDENCE_WARNING_TEXT = (
    "Low model confidence. The prediction may be unreliable. "
    "Review the alternative predictions and consult a qualified healthcare professional."
)

class SafetyGuardrail:
    """Responsible AI safety engine enforcing clinical decision-support disclaimers, system limitations, and low-confidence warning flags."""

    def __init__(self, confidence_threshold: float = 0.70, config_path: str = "config.yaml"):
        try:
            cfg = get_config(config_path)
            if hasattr(cfg, "confidence_thresholds"):
                confidence_threshold = float(getattr(cfg.confidence_thresholds, "moderate", confidence_threshold))
        except Exception:
            pass

        self.confidence_threshold = confidence_threshold

    def evaluate_prediction_safety(self, confidence: float, top_k_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluates prediction safety and returns low-confidence warning text, disclaimers, and limitations."""
        conf = float(confidence)
        is_low_confidence = (conf < self.confidence_threshold)

        warning_text = LOW_CONFIDENCE_WARNING_TEXT if is_low_confidence else None

        return {
            "is_low_confidence": is_low_confidence,
            "confidence_threshold": self.confidence_threshold,
            "confidence_score": conf,
            "low_confidence_warning_text": warning_text,
            "medical_disclaimer": self.get_medical_disclaimer(),
            "model_limitations": self.get_model_limitations(),
            "dataset_limitations": self.get_dataset_limitations(),
            "explanation_limitations": self.get_explanation_limitations(),
            "human_review_recommendation": self.get_human_review_recommendation()
        }

    @staticmethod
    def get_medical_disclaimer() -> str:
        """Returns non-clinical decision-support medical disclaimer."""
        return (
            "MEDICAL DISCLAIMER: This application is an academic AI-assisted medical text classification prototype "
            "designed exclusively for educational and research evaluation. It does NOT provide clinical diagnoses, "
            "does NOT prescribe medication, does NOT recommend treatment plans, and does NOT claim clinical validity. "
            "All model outputs must be independently reviewed by a qualified healthcare professional."
        )

    @staticmethod
    def get_model_limitations() -> List[str]:
        """Returns documented model limitations."""
        return [
            "Input text sequence length is truncated to a maximum of 128 tokens.",
            "Model accuracy depends heavily on patient symptom description clarity and phrasing.",
            "Model is vulnerable to out-of-vocabulary medical terms, typos, or colloquial regional idioms.",
            "Transformer predictions represent statistical probabilities, not medical certainty."
        ]

    @staticmethod
    def get_dataset_limitations() -> List[str]:
        """Returns documented dataset limitations."""
        return [
            "Trained on the Symptom2Disease dataset containing ~1,200 text samples across 24 disease classes.",
            "Lacks patient clinical vitals (blood pressure, heart rate, temperature, oxygen saturation).",
            "Lacks patient medical history, demographic context, lab results, or imaging data.",
            "Does not cover rare diseases or complex multi-morbid conditions outside the 24 target categories."
        ]

    @staticmethod
    def get_explanation_limitations() -> List[str]:
        """Returns documented explainability (XAI) limitations."""
        return [
            "Feature importance attributions indicate correlation with model output, NOT clinical causality.",
            "Model Attention Signals reflect internal Transformer sequence alignment, not medical etiology.",
            "LIME and SHAP provide local surrogate approximations that may vary with perturbation sampling.",
            "XAI highlights should be used to inspect model reasoning, not to formulate clinical diagnoses."
        ]

    @staticmethod
    def get_human_review_recommendation() -> str:
        """Returns human-in-the-loop clinical review guidance."""
        return (
            "RECOMMENDATION: Always maintain a Human-in-the-Loop workflow. A licensed medical practitioner "
            "must evaluate the patient's full clinical history, physical exam, and laboratory diagnostics before making medical decisions."
        )
