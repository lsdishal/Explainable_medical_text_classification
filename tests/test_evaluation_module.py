import pytest
import pandas as pd
from src.evaluation.error_analysis import analyze_model_errors

def test_analyze_model_errors_sorting():
    texts = [
        "high fever and severe cough",
        "skin rash with scaly patches",
        "frequent urination and thirst",
        "persistent chest pain"
    ]
    y_true = ["Pneumonia", "Psoriasis", "Diabetes", "Hypertension"]
    y_pred = ["Common Cold", "Psoriasis", "Diabetes", "Pneumonia"] # 2 errors
    confs = [0.92, 0.98, 0.95, 0.74] # High confidence error at 0.92 (Common Cold), lower confidence error at 0.74 (Pneumonia)

    error_df, summary = analyze_model_errors(texts, y_true, y_pred, confs)

    assert summary["total_samples"] == 4
    assert summary["correct_count"] == 2
    assert summary["incorrect_count"] == 2

    # Verify that errors are placed first and sorted by confidence descending
    errors_only = error_df[error_df["is_error"]].reset_index(drop=True)
    assert len(errors_only) == 2
    assert errors_only.iloc[0]["confidence"] == 0.92
    assert errors_only.iloc[1]["confidence"] == 0.74
    assert errors_only.iloc[0]["predicted_label"] == "Common Cold"

def test_confused_pairs_calculation():
    texts = ["t1", "t2", "t3"]
    y_true = ["DiseaseA", "DiseaseA", "DiseaseB"]
    y_pred = ["DiseaseB", "DiseaseB", "DiseaseA"]
    confs = [0.8, 0.9, 0.85]

    _, summary = analyze_model_errors(texts, y_true, y_pred, confs)
    top_pairs = summary["top_confused_class_pairs"]

    assert len(top_pairs) > 0
    first_pair = top_pairs[0]
    assert first_pair["ground_truth"] == "DiseaseA"
    assert first_pair["predicted_as"] == "DiseaseB"
    assert first_pair["error_count"] == 2
