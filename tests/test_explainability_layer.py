import pytest
import torch
from unittest.mock import MagicMock

from src.explainability.attention import get_attention_signal
from src.explainability.explanation_manager import ExplanationManager

def test_attention_signal_disclaimer_and_title():
    # Mock model and tokenizer for attention test
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2054, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "fever", "[SEP]"]

    # Mock outputs.attentions
    mock_attn = torch.ones((1, 1, 12, 3, 3))
    mock_outputs = MagicMock()
    mock_outputs.attentions = (mock_attn,)
    mock_model.return_value = mock_outputs

    device = torch.device("cpu")
    result = get_attention_signal(mock_model, mock_tokenizer, "fever", device)

    assert result["section_title"] == "Model Attention Signal"
    assert "attention_disclaimer" in result
    assert "IS NOT automatically a causal explanation" in result["attention_disclaimer"]
    assert len(result["token_scores"]) > 0

def test_explanation_consistency_labeling():
    # Test internal consistency analysis logic
    manager = ExplanationManager(None, None, ["Psoriasis", "Typhoid"], torch.device("cpu"))

    att_scores = [("fever", 0.8), ("headache", 0.9), ("cough", 0.1)]
    shap_scores = [("fever", 0.4), ("headache", 0.5), ("cough", 0.0)]
    lime_scores = [("fever", 0.3), ("headache", 0.2), ("cough", 0.0)]
    captum_scores = [("fever", 0.6), ("headache", 0.7), ("cough", 0.0)]

    records, consistent = manager._compute_consistency_analysis(att_scores, shap_scores, lime_scores, captum_scores)

    consistent_words = [item["word"] for item in consistent]
    assert "fever" in consistent_words
    assert "headache" in consistent_words
    assert "cough" not in consistent_words

    for item in consistent:
        assert item["feature_label"] == "Consistently influential feature"
        assert item["is_consistent"] is True

def test_explanation_manager_caching():
    manager = ExplanationManager(None, None, ["Psoriasis"], torch.device("cpu"))

    # Seed mock cache entry
    cache_key = manager._generate_cache_key("severe skin rash", 0)
    fake_output = {"prediction": "Psoriasis", "cached": True}
    manager._cache[cache_key] = fake_output

    cached_res = manager.explain("severe skin rash", 0, "Psoriasis", 0.95, use_cache=True)
    assert cached_res["cached"] is True
    assert cached_res["prediction"] == "Psoriasis"
