import pytest
from pathlib import Path
from src.utils.config import get_config, load_config

def test_load_config():
    config_dict = load_config("config.yaml")
    assert "project" in config_dict
    assert "dataset" in config_dict
    assert "models" in config_dict

def test_config_object():
    cfg = get_config("config.yaml")
    assert cfg.project.name == "Explainable Medical Text Classification"
    assert cfg.models.primary.name == "biobert"
    assert cfg.training.max_length == 128
