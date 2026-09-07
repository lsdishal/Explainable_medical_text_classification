import json
import pytest
from src.inference.confidence import ConfidenceEstimator
from src.inference.predict import MedicalSymptomClassifier

def test_confidence_estimator_categories():
    estimator = ConfidenceEstimator(high_threshold=0.90, moderate_threshold=0.70)

    # High Confidence (>= 0.90)
    high_eval = estimator.evaluate_confidence(0.95)
    assert high_eval["confidence_category"] == "High"
    assert high_eval["low_confidence_warning"] is False

    # Moderate Confidence (0.70 - 0.89)
    mod_eval = estimator.evaluate_confidence(0.78)
    assert mod_eval["confidence_category"] == "Moderate"
    assert mod_eval["low_confidence_warning"] is False

    # Low Confidence (< 0.70)
    low_eval = estimator.evaluate_confidence(0.45)
    assert low_eval["confidence_category"] == "Low"
    assert low_eval["low_confidence_warning"] is True
    assert "WARNING" in low_eval["warning_message"]

def test_inference_empty_input():
    classifier = MedicalSymptomClassifier(model_dir="models/biobert")
    empty_result = classifier.predict("   ")
    
    assert empty_result["status"] == "invalid_input"
    assert "error" in empty_result
    assert empty_result["low_confidence_warning"] is True
    # Verify JSON serializability
    assert json.dumps(empty_result)

def test_inference_normal_and_unusual_input():
    classifier = MedicalSymptomClassifier(model_dir="models/biobert")
    
    # Text with HTML tags, strange spaces, and unusual characters
    text = "<br> I have severe itching &amp; skin redness with scaly patches! \xa0 "
    res = classifier.predict(text)

    assert res["status"] == "success"
    assert "predicted_disease" in res
    assert "confidence" in res
    assert "top_3_predictions" in res
    assert len(res["top_3_predictions"]) == 3
    assert "tokenized_input" in res
    assert isinstance(res["tokenized_input"], list)
    assert "medical_disclaimer" in res
    assert "DISCLAIMER" in res["medical_disclaimer"]

    # Test 100% JSON serializability
    serialized = json.dumps(res)
    assert isinstance(serialized, str)

def test_inference_extremely_long_input():
    classifier = MedicalSymptomClassifier(model_dir="models/biobert")
    
    # Generate 2000 character long text input
    long_text = "Patient complains of high fever, severe headache, muscle pain, and joint pain. " * 30
    assert len(long_text) > 2000

    res = classifier.predict(long_text)
    assert res["status"] == "success"
    assert "predicted_disease" in res
    assert json.dumps(res)
