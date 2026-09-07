import pytest
import pandas as pd
from src.explainability.explanation_manager import ExplanationManager

def test_explanation_manager_consistency_formatting():
    # Verify method signature and formatting logic
    att_scores = [("fever", 0.8), ("cough", 0.6)]
    shap_scores = [("fever", 0.5), ("cough", 0.4)]
    lime_scores = [("fever", 0.3), ("cough", 0.2)]
    captum_scores = [("fever", 0.7), ("cough", 0.5)]

    manager = ExplanationManager(None, None, ["DiseaseA"], device="cpu")
    records, consistent = manager._compute_consistency_analysis(att_scores, shap_scores, lime_scores, captum_scores)

    words = [r["word"] for r in records]
    assert "fever" in words
    assert "cough" in words
    assert len(consistent) == 2
