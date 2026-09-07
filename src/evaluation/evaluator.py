import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, Any, List

from src.utils.logger import logger
from src.utils.config import get_config
from src.utils.metrics import compute_metrics
from src.preprocessing.dataset import SymptomDataset
from src.training.model_loader import get_compute_device as get_device

class ModelEvaluator:
    """Evaluates fine-tuned transformer models on test data and performs error analysis."""

    def __init__(self, model_key: str = "primary", config_path: str = "config.yaml"):
        self.cfg = get_config(config_path)
        self.config_path = config_path
        self.model_key = model_key

        if model_key == "primary":
            self.model_cfg = self.cfg.models.primary
        elif model_key == "baseline_bert":
            self.model_cfg = self.cfg.models.baseline_bert
        elif model_key == "baseline_distilbert":
            self.model_cfg = self.cfg.models.baseline_distilbert
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        self.model_name = self.model_cfg.name
        self.model_dir = Path(self.model_cfg.save_dir)
        self.device = get_device(self.cfg.training.device)

    def evaluate(self) -> Dict[str, Any]:
        """Runs test evaluation and saves error analysis report and confusion matrix plot."""
        logger.info(f"=== Evaluating Model: {self.model_name} on Test Set ===")
        
        test_path = Path(self.cfg.dataset.processed_dir) / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {test_path}. Run downloader/trainer first.")
        
        test_df = pd.read_csv(test_path)

        # Load saved model & tokenizer
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory '{self.model_dir}' does not exist. Train the model first.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model.to(self.device)
        model.eval()

        with open(self.model_dir / "id2label.json", "r", encoding="utf-8") as f:
            id2label_raw = json.load(f)
            id2label = {int(k): v for k, v in id2label_raw.items()}

        with open(self.model_dir / "label2id.json", "r", encoding="utf-8") as f:
            label2id = json.load(f)

        max_len = self.cfg.training.max_length
        test_dataset = SymptomDataset(test_df["text"].values, test_df["label_id"].values, tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.eval_batch_size, shuffle=False)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = model(**kwargs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        target_names = [id2label[i] for i in sorted(id2label.keys())]
        metrics = compute_metrics(all_labels, all_preds, target_names=target_names)

        # Plot & save Confusion Matrix
        fig_dir = Path(self.cfg.paths.figures_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        cm_path = fig_dir / f"confusion_matrix_{self.model_name}.png"
        self._plot_confusion_matrix(metrics["confusion_matrix"], target_names, cm_path)

        # Build Error Analysis Report
        error_records = []
        for idx, (true_id, pred_id, probs) in enumerate(zip(all_labels, all_preds, all_probs)):
            true_label = id2label[true_id]
            pred_label = id2label[pred_id]
            conf = float(probs[pred_id])
            symptom_text = test_df["text"].iloc[idx]
            is_error = (true_id != pred_id)

            error_records.append({
                "sample_index": idx,
                "symptom_text": symptom_text,
                "ground_truth": true_label,
                "predicted_label": pred_label,
                "confidence": conf,
                "is_error": is_error
            })

        error_df = pd.DataFrame(error_records)
        reports_dir = Path(self.cfg.paths.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        error_df.to_csv(reports_dir / f"error_analysis_{self.model_name}.csv", index=False)
        with open(reports_dir / f"test_metrics_{self.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        accuracy = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        total_errors = error_df["is_error"].sum()
        logger.info(f"Model '{self.model_name}' Evaluation Completed. Test Acc: {accuracy:.4f} | Macro F1: {f1:.4f} | Total Errors: {total_errors}/{len(test_df)}")

        return metrics

    def _plot_confusion_matrix(self, cm: List[List[int]], target_names: List[str], save_path: Path):
        """Generates heatmap visualization of confusion matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.xlabel("Predicted Disease Label", fontsize=12)
        plt.ylabel("Ground Truth Disease Label", fontsize=12)
        plt.title(f"Confusion Matrix - {self.model_name.upper()}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def generate_model_comparison(config_path: str = "config.yaml") -> pd.DataFrame:
    """Aggregates metrics from all trained models for comparative analysis."""
    cfg = get_config(config_path)
    reports_dir = Path(cfg.paths.reports_dir)
    records = []

    models = [cfg.models.primary, cfg.models.baseline_bert, cfg.models.baseline_distilbert]
    for m in models:
        metrics_path = reports_dir / f"test_metrics_{m.name}.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append({
                "Model": m.name.upper(),
                "Hugging Face ID": m.hf_model_id,
                "Accuracy": data.get("accuracy", 0.0),
                "Macro Precision": data.get("macro_precision", 0.0),
                "Macro Recall": data.get("macro_recall", 0.0),
                "Macro F1": data.get("macro_f1", 0.0),
                "Weighted F1": data.get("weighted_f1", 0.0)
            })

    comparison_df = pd.DataFrame(records)
    if not comparison_df.empty:
        comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
        logger.info(f"Model comparison saved to {reports_dir / 'model_comparison.csv'}")
    return comparison_df

if __name__ == "__main__":
    import sys
    model_key = sys.argv[1] if len(sys.argv) > 1 else "primary"
    evaluator = ModelEvaluator(model_key=model_key)
    evaluator.evaluate()
    generate_model_comparison()
