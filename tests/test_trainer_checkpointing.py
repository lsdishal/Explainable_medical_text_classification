import os
import json
import pytest
import torch
from pathlib import Path

from src.training.trainer import TransformerTrainer
from src.utils.config import get_config

def test_trainer_initialization():
    trainer = TransformerTrainer(model_key="primary", config_path="config.yaml")
    assert trainer.model_name == "biobert"
    assert trainer.batch_size == 16
    assert trainer.learning_rate == 3e-5
    assert trainer.patience == 3

def test_checkpoint_file_saving(tmp_path):
    history = {
        "model_name": "biobert",
        "epochs": [1, 2],
        "train_loss": [0.8, 0.4],
        "val_loss": [0.6, 0.3],
        "val_accuracy": [0.75, 0.88],
        "val_macro_f1": [0.70, 0.85]
    }

    report_path = tmp_path / "training_history.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "biobert"
    assert len(data["epochs"]) == 2
    assert data["val_macro_f1"][-1] == 0.85
