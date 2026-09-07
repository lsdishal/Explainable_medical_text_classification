import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import logger
from src.utils.config import get_config
from src.training.trainer import TransformerTrainer

def plot_training_curves(history: Dict[str, Any], save_path: Path):
    """Generates 4-panel training curve figure (training loss, val loss, val accuracy, val macro F1)."""
    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = history["val_accuracy"]
    val_f1 = history["val_macro_f1"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_title = history.get("model_name", "Transformer").upper()

    # Panel 1: Training Loss
    axes[0, 0].plot(epochs, train_loss, marker="o", color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Training Loss over Epochs", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross Entropy Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Loss
    axes[0, 1].plot(epochs, val_loss, marker="o", color="#d62728", linewidth=2)
    axes[0, 1].set_title("Validation Loss over Epochs", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Validation Accuracy
    axes[1, 0].plot(epochs, [a * 100 for a in val_acc], marker="s", color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Validation Accuracy (%) over Epochs", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Validation Macro F1
    axes[1, 1].plot(epochs, [f * 100 for f in val_f1], marker="^", color="#9467bd", linewidth=2)
    axes[1, 1].set_title("Validation Macro F1 Score (%) over Epochs", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Macro F1 (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(f"Transformer Training Dynamics: {model_title}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curve figure to '{save_path}'")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT, BERT, or DistilBERT for Medical Symptom Classification.")
    parser.add_argument(
        "--model",
        type=str,
        default="primary",
        choices=["primary", "baseline_bert", "baseline_distilbert", "biobert", "bert", "distilbert"],
        help="Model identifier to train ('primary' / 'biobert', 'baseline_bert' / 'bert', 'baseline_distilbert' / 'distilbert')."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint if available.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")

    args = parser.parse_args()

    # Map model shortcut alias
    model_map = {
        "biobert": "primary",
        "primary": "primary",
        "bert": "baseline_bert",
        "baseline_bert": "baseline_bert",
        "distilbert": "baseline_distilbert",
        "baseline_distilbert": "baseline_distilbert"
    }

    model_key = model_map[args.model]

    trainer = TransformerTrainer(
        model_key=model_key,
        config_path=args.config,
        resume=args.resume,
        epochs_override=args.epochs,
        learning_rate_override=args.learning_rate,
        batch_size_override=args.batch_size
    )

    history = trainer.train()

    # Generate training curves
    cfg = get_config(args.config)
    fig_dir = Path(cfg.paths.figures_dir)
    curve_path = fig_dir / f"training_curves_{trainer.model_name}.png"
    plot_training_curves(history, curve_path)

if __name__ == "__main__":
    main()
