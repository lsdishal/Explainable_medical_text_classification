import os
import json
import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing.text_cleaner import clean_medical_text, calculate_text_stats, is_outlier_length, NEGATION_WORDS
from src.preprocessing.label_encoder import DiseaseLabelEncoder

def test_clean_medical_text_html_and_whitespace():
    raw = "   Patient has <br> high fever &amp; chills!   "
    cleaned = clean_medical_text(raw)
    assert cleaned == "Patient has high fever & chills!"
    assert "<br>" not in cleaned
    assert "&amp;" not in cleaned

def test_preservation_of_negation_words():
    # Negations must be strictly preserved in medical symptom context
    sample = "No chest pain, not feeling dizzy, without shortness of breath."
    cleaned = clean_medical_text(sample)

    for word in ["No", "not", "without"]:
        assert word in cleaned.split() or word.lower() in cleaned.lower().split()

def test_no_stemming_or_lemmatization():
    # Medical terms should retain exact suffixes for Transformer tokenization
    sample = "experiencing swelling, inflammation, and coughing"
    cleaned = clean_medical_text(sample)
    assert "experiencing" in cleaned
    assert "swelling" in cleaned
    assert "inflammation" in cleaned
    assert "coughing" in cleaned

def test_label_encoder_mapping_and_serialization(tmp_path):
    labels = ["Psoriasis", "Typhoid", "Dengue", "Psoriasis", "Typhoid"]
    encoder = DiseaseLabelEncoder()
    encoded = encoder.fit_transform(labels)

    assert len(encoder.classes_) == 3
    assert set(encoder.classes_) == {"Dengue", "Psoriasis", "Typhoid"}
    assert encoded[0] == encoder.label2id["Psoriasis"]

    decoded = encoder.inverse_transform(encoded)
    assert decoded == labels

    # Test JSON saving and loading
    save_file = tmp_path / "label_mapping.json"
    encoder.save(save_file)
    assert save_file.exists()

    loaded_encoder = DiseaseLabelEncoder.load(save_file)
    assert loaded_encoder.classes_ == encoder.classes_
    assert loaded_encoder.label2id == encoder.label2id

def test_text_stats_and_outliers():
    short_text = "Fever"
    stats = calculate_text_stats(short_text)
    assert stats["char_length"] == 5
    assert stats["word_count"] == 1

    is_out, out_type = is_outlier_length(short_text, min_chars=10, min_words=3)
    assert is_out is True
    assert out_type == "unusually_short"

    normal_text = "I have been experiencing high fever, headache, and chills for 3 days."
    is_out_norm, out_type_norm = is_outlier_length(normal_text)
    assert is_out_norm is False
    assert out_type_norm == "normal"
