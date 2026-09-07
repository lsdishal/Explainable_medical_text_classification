import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.logger import logger
from src.utils.config import get_config
from src.utils.metrics import compute_metrics
from src.preprocessing.dataset import SymptomDataset
from src.training.model_loader import get_compute_device
from src.evaluation.error_analysis import analyze_model_errors
from src.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_confidence_distribution,
    plot_model_comparison
)

class ModelEvaluator:
    """Evaluator class for sequence classification models on test dataset."""

    def __init__(self, model_key: str = "primary", config_path: str = "config.yaml"):
        self.cfg = get_config(config_path)
        self.config_path = config_path
        self.model_key = model_key

        if model_key == "primary":
            self.model_meta = self.cfg.models.primary
        elif model_key == "baseline_bert":
            self.model_meta = self.cfg.models.baseline_bert
        elif model_key == "baseline_distilbert":
            self.model_meta = self.cfg.models.baseline_distilbert
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        self.model_name = self.model_meta.name
        self.model_dir = Path(self.model_meta.save_dir)
        self.device = get_compute_device(getattr(self.cfg.training, "device", "auto"))

    def evaluate(self) -> Dict[str, Any]:
        """Runs test evaluation, generates error analysis, plots figures, and exports reports."""
        logger.info(f"=== Evaluating Model: {self.model_name.upper()} on Test Set ===")

        test_path = Path(self.cfg.dataset.processed_dir) / "processed_test.csv"
        if not test_path.exists():
            test_path = Path(self.cfg.dataset.processed_dir) / "test.csv"

        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at '{test_path}'. Run dataset preparation first.")

        test_df = pd.read_csv(test_path)
        text_col = "text" if "text" in test_df.columns else "cleaned_text"

        # Load saved model & tokenizer
        if not self.model_dir.exists():
            if Path("models/best_model").exists():
                self.model_dir = Path("models/best_model")
            else:
                logger.warning(f"Trained checkpoint '{self.model_dir}' not found. Using pretrained Hugging Face checkpoint: '{self.model_meta.hf_model_id}'")
                self.model_dir = Path(self.model_meta.hf_model_id)

        if self.model_key == "primary":
            tokenizer_dir = Path(getattr(self.cfg.models, "tokenizer_dir", "models/tokenizer"))
            tok_path = tokenizer_dir if tokenizer_dir.exists() else self.model_dir
        else:
            tok_path = self.model_dir if (self.model_dir / "tokenizer_config.json").exists() else self.model_meta.hf_model_id

        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model.to(self.device)
        model.eval()

        label_file = self.model_dir / "id2label.json"
        if not label_file.exists():
            label_file = Path(self.cfg.dataset.processed_dir) / "label_mapping.json"

        with open(label_file, "r", encoding="utf-8") as f:
            raw_id2label = json.load(f)
            if "id2label" in raw_id2label:
                id2label = {int(k): v for k, v in raw_id2label["id2label"].items()}
            else:
                id2label = {int(k): v for k, v in raw_id2label.items()}

        max_len = int(getattr(self.cfg.training, "max_length", 128))
        eval_batch_size = int(getattr(self.cfg.training, "eval_batch_size", 32))

        test_dataset = SymptomDataset(test_df[text_col].values, test_df["label_id"].values, tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

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

        y_true_names = [id2label[i] for i in all_labels]
        y_pred_names = [id2label[i] for i in all_preds]
        confidences = [float(probs[pred_id]) for probs, pred_id in zip(all_probs, all_preds)]
        symptom_texts = test_df[text_col].tolist()

        # Run Error Analysis
        error_df, error_summary = analyze_model_errors(symptom_texts, y_true_names, y_pred_names, confidences)

        # Save Reports
        reports_dir = Path(self.cfg.paths.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        error_df.to_csv(reports_dir / f"error_analysis_{self.model_name}.csv", index=False)
        with open(reports_dir / f"error_analysis_{self.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(error_summary, f, indent=2)

        with open(reports_dir / f"test_evaluation_{self.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Generate Visualizations
        fig_dir = Path(self.cfg.paths.figures_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(metrics["confusion_matrix"], target_names, fig_dir / f"confusion_matrix_{self.model_name}.png", title=f"Confusion Matrix - {self.model_name.upper()}")
        plot_per_class_metrics(metrics["classification_report"], target_names, fig_dir / f"per_class_metrics_{self.model_name}.png", title=f"Per-Class Metrics - {self.model_name.upper()}")
        plot_confidence_distribution(error_df, fig_dir / f"confidence_distribution_{self.model_name}.png", title=f"Confidence Distribution - {self.model_name.upper()}")

        logger.info(
            f"Model '{self.model_name.upper()}' Test Results | Acc: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}"
        )

        return {
            "model_name": self.model_name,
            "metrics": metrics,
            "error_summary": error_summary
        }

def build_model_comparison_table(config_path: str = "config.yaml") -> pd.DataFrame:
    """Builds comparative evaluation table across BioBERT, BERT, and DistilBERT using actual test predictions."""
    cfg = get_config(config_path)
    reports_dir = Path(cfg.paths.reports_dir)

    models = [cfg.models.primary, cfg.models.baseline_bert, cfg.models.baseline_distilbert]
    records = []

    for m in models:
        eval_path = reports_dir / f"test_evaluation_{m.name}.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            records.append({
                "Model": m.name.upper(),
                "Hugging Face Model ID": m.hf_model_id,
                "Accuracy": float(data.get("accuracy", 0.0)),
                "Precision": float(data.get("macro_precision", 0.0)),
                "Recall": float(data.get("macro_recall", 0.0)),
                "Macro F1": float(data.get("macro_f1", 0.0)),
                "Weighted F1": float(data.get("weighted_f1", 0.0))
            })

    comparison_df = pd.DataFrame(records)

    if not comparison_df.empty:
        comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
        with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_df.to_dict(orient="records"), f, indent=2)

        fig_dir = Path(cfg.paths.figures_dir)
        plot_model_comparison(comparison_df, fig_dir / "model_comparison.png")
        logger.info(f"Saved model comparison table to '{reports_dir / 'model_comparison.csv'}'")

    return comparison_df

def run_evaluation_pipeline(model_key: str = "primary", config_path: str = "config.yaml"):
    """Executes model evaluation and builds model comparison table."""
    model_map = {
        "biobert": "primary",
        "primary": "primary",
        "bert": "baseline_bert",
        "baseline_bert": "baseline_bert",
        "distilbert": "baseline_distilbert",
        "baseline_distilbert": "baseline_distilbert"
    }
    resolved_key = model_map.get(model_key, "primary")
    evaluator = ModelEvaluator(model_key=resolved_key, config_path=config_path)
    res = evaluator.evaluate()
    build_model_comparison_table(config_path)
    return res

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Transformer models on test dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="primary",
        choices=["primary", "baseline_bert", "baseline_distilbert", "biobert", "bert", "distilbert", "all"],
        help="Model key to evaluate or 'all'."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")

    args = parser.parse_args()

    if args.model == "all":
        for key in ["primary", "baseline_bert", "baseline_distilbert"]:
            try:
                evaluator = ModelEvaluator(model_key=key, config_path=args.config)
                evaluator.evaluate()
            except Exception as e:
                logger.warning(f"Could not evaluate model '{key}': {e}")
        build_model_comparison_table(args.config)
    else:
        run_evaluation_pipeline(model_key=args.model, config_path=args.config)

if __name__ == "__main__":
    main()
