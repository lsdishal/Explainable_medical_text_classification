import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.config import get_config
from src.preprocessing.dataset import SymptomDataset
from src.training.model_loader import load_model_and_tokenizer, set_seed
from src.training.metrics import compute_classification_metrics

class TransformerTrainer:
    """Trainer class for fine-tuning Transformer-based NLP sequence classifiers."""

    def __init__(
        self,
        model_key: str = "primary",
        config_path: str = "config.yaml",
        resume: bool = False,
        epochs_override: Optional[int] = None,
        learning_rate_override: Optional[float] = None,
        batch_size_override: Optional[int] = None
    ):
        self.cfg = get_config(config_path)
        self.config_path = config_path
        self.model_key = model_key
        self.resume = resume

        # Resolve model configuration
        if model_key == "primary":
            self.model_meta = self.cfg.models.primary
        elif model_key == "baseline_bert":
            self.model_meta = self.cfg.models.baseline_bert
        elif model_key == "baseline_distilbert":
            self.model_meta = self.cfg.models.baseline_distilbert
        else:
            raise ValueError(f"Unknown model_key: '{model_key}'. Allowed: 'primary', 'baseline_bert', 'baseline_distilbert'")

        self.model_name = self.model_meta.name
        self.hf_model_id = self.model_meta.hf_model_id
        self.model_save_dir = Path(self.model_meta.save_dir)
        
        # Best model global output directory
        self.best_model_dir = Path(getattr(self.cfg.models, "best_model_dir", "models/best_model"))
        self.tokenizer_dir = Path(getattr(self.cfg.models, "tokenizer_dir", "models/tokenizer"))

        # Configurable Hyperparameters
        self.learning_rate = float(learning_rate_override or self.cfg.training.learning_rate)
        self.batch_size = int(batch_size_override or self.cfg.training.batch_size)
        self.eval_batch_size = int(self.cfg.training.eval_batch_size)
        self.epochs = int(epochs_override or self.cfg.training.epochs)
        self.max_length = int(self.cfg.training.max_length)
        self.weight_decay = float(self.cfg.training.weight_decay)
        self.warmup_ratio = float(self.cfg.training.warmup_ratio)
        self.seed = int(getattr(self.cfg.training, "random_seed", self.cfg.project.seed))
        self.patience = int(getattr(self.cfg.training, "early_stopping_patience", 3))

    def train(self) -> Dict[str, Any]:
        """Runs fine-tuning loop with per-epoch validation, early stopping, and checkpoint resumption."""
        set_seed(self.seed)

        # 1. Load Data Splits
        processed_dir = Path(self.cfg.dataset.processed_dir)
        train_path = processed_dir / "processed_train.csv"
        val_path = processed_dir / "processed_validation.csv"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Processed datasets not found in data/processed/. Run dataset_preparation.py first.")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # 2. Load Model, Tokenizer & Hardware Device
        model, tokenizer, device, label2id, id2label = load_model_and_tokenizer(
            self.hf_model_id,
            config_path=self.config_path,
            label_mapping_path=str(processed_dir / "label_mapping.json")
        )

        # 3. Create PyTorch DataLoaders
        train_dataset = SymptomDataset(train_df["text"].values, train_df["label_id"].values, tokenizer, self.max_length)
        val_dataset = SymptomDataset(val_df["text"].values, val_df["label_id"].values, tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.eval_batch_size, shuffle=False)

        # 4. Optimizer & Linear Warmup Scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Resumption & Early Stopping Setup
        start_epoch = 1
        best_val_macro_f1 = 0.0
        patience_counter = 0
        history = {
            "model_name": self.model_name,
            "hf_model_id": self.hf_model_id,
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_weighted_f1": []
        }

        checkpoint_path = self.model_save_dir / "checkpoint.pt"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        if self.resume and checkpoint_path.exists():
            logger.info(f"Resuming training from checkpoint: '{checkpoint_path}'")
            ckpt = torch.load(checkpoint_path, map_location=device)
            start_epoch = ckpt["epoch"] + 1
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_macro_f1 = ckpt["best_val_macro_f1"]
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", history)
            logger.info(f"Resumed at epoch {start_epoch} with Best Val Macro F1: {best_val_macro_f1:.4f}")

        logger.info(f"=== Starting Training ({self.model_name.upper()}) | Total Epochs: {self.epochs} | Device: {device} ===")

        target_names = [id2label[i] for i in sorted(id2label.keys())]

        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            train_loss_sum = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

                outputs = model(**kwargs)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # Validation Loop
            model.eval()
            val_loss_sum = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                    if "token_type_ids" in batch:
                        kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

                    outputs = model(**kwargs)
                    val_loss_sum += outputs.loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss_sum / len(val_loader)
            metrics = compute_classification_metrics(all_labels, all_preds, target_names=target_names)
            val_acc = metrics["accuracy"]
            val_f1 = metrics["macro_f1"]
            val_wf1 = metrics["weighted_f1"]

            # Record history
            history["epochs"].append(epoch)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_macro_f1"].append(val_f1)
            history["val_weighted_f1"].append(val_wf1)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} | Val Weighted F1: {val_wf1:.4f}"
            )

            # Save Checkpoint for Resumption
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "patience_counter": patience_counter,
                "history": history
            }, checkpoint_path)

            # Check for Best Model Checkpointing
            if val_f1 > best_val_macro_f1:
                logger.info(f"🔥 Validation Macro F1 improved ({best_val_macro_f1:.4f} -> {val_f1:.4f}). Saving best checkpoint...")
                best_val_macro_f1 = val_f1
                patience_counter = 0

                # Save to specific model directory
                model.save_pretrained(self.model_save_dir)
                tokenizer.save_pretrained(self.model_save_dir)

                # Save to global best model directory ONLY if this is primary model
                if self.model_key == "primary":
                    self.best_model_dir.mkdir(parents=True, exist_ok=True)
                    self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(self.best_model_dir)
                    tokenizer.save_pretrained(self.tokenizer_dir)

                # Save label mapping to model directory
                with open(self.model_save_dir / "label2id.json", "w", encoding="utf-8") as f:
                    json.dump(label2id, f, indent=2)
                with open(self.model_save_dir / "id2label.json", "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

                if self.model_key == "primary":
                    with open(self.best_model_dir / "label2id.json", "w", encoding="utf-8") as f:
                        json.dump(label2id, f, indent=2)
                    with open(self.best_model_dir / "id2label.json", "w", encoding="utf-8") as f:
                        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
            else:
                patience_counter += 1
                logger.info(f"Validation Macro F1 did not improve. Early Stopping counter: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    logger.info(f"🛑 Early stopping triggered after {epoch} epochs.")
                    break

        # Save training history JSON report
        reports_dir = Path(self.cfg.paths.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / f"training_history_{self.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(reports_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training completed for '{self.model_name}'. Best Val Macro F1: {best_val_macro_f1:.4f}")
        return history

if __name__ == "__main__":
    trainer = TransformerTrainer(model_key="primary")
    trainer.train()
