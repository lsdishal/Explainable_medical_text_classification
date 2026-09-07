import pytest
import json
import pandas as pd
from pathlib import Path

from src.training.metrics import compute_classification_metrics
from src.training.model_loader import load_label_mapping, get_compute_device

def test_compute_classification_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1] # 5/6 correct
    target_names = ["DiseaseA", "DiseaseB", "DiseaseC"]

    metrics = compute_classification_metrics(y_true, y_pred, target_names=target_names)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["accuracy"] == pytest.approx(5.0 / 6.0)
    assert metrics["macro_f1"] > 0.0

def test_load_label_mapping(tmp_path):
    mapping_data = {
        "num_classes": 2,
        "classes": ["DiseaseA", "DiseaseB"],
        "label2id": {"DiseaseA": 0, "DiseaseB": 1},
        "id2label": {"0": "DiseaseA", "1": "DiseaseB"}
    }
    mapping_file = tmp_path / "label_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f)

    l2i, i2l, num_labels = load_label_mapping(str(mapping_file))
    assert num_labels == 2
    assert l2i["DiseaseA"] == 0
    assert i2l[1] == "DiseaseB"

def test_get_compute_device():
    device = get_compute_device("auto")
    assert device.type in ["cuda", "mps", "cpu"]
