import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.logger import logger
from src.utils.config import get_config

def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")

def get_compute_device(device_setting: str = "auto") -> torch.device:
    """Automatically detects best compute hardware (CUDA GPU, Apple Silicon MPS GPU, or CPU)."""
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon MPS GPU detected.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device.")
    else:
        device = torch.device(device_setting)
        logger.info(f"Using explicitly configured device: {device}")
    return device

def load_label_mapping(label_mapping_path: str = "data/processed/label_mapping.json") -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Loads disease label mapping dictionary from processed data directory."""
    path = Path(label_mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found at '{path}'. Run dataset preparation pipeline first.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    num_labels = data["num_classes"]

    return label2id, id2label, num_labels

def load_model_and_tokenizer(
    model_name_or_path: str,
    config_path: str = "config.yaml",
    label_mapping_path: str = "data/processed/label_mapping.json"
) -> Tuple[Any, Any, torch.device, Dict[str, int], Dict[int, str]]:
    """Loads Hugging Face tokenizer and AutoModelForSequenceClassification initialized with label mapping."""
    cfg = get_config(config_path)
    
    # 1. Set seed
    set_seed(getattr(cfg.training, "random_seed", getattr(cfg.project, "seed", 42)))

    # 2. Get hardware device
    device = get_compute_device(cfg.training.device)

    # 3. Load label mappings
    label2id, id2label, num_labels = load_label_mapping(label_mapping_path)

    logger.info(f"Loading pretrained tokenizer & model: '{model_name_or_path}' for {num_labels} disease classes...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    model.to(device)
    logger.info(f"Successfully loaded '{model_name_or_path}' onto device '{device}'")

    return model, tokenizer, device, label2id, id2label

if __name__ == "__main__":
    model, tokenizer, device, l2i, i2l = load_model_and_tokenizer("dmis-lab/biobert-v1.1")
    print("Device:", device)
    print("Num labels:", len(l2i))
