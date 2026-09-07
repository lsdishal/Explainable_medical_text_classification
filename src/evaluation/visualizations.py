import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import logger

def plot_confusion_matrix(cm: List[List[int]], target_names: List[str], save_path: Path, title: str = "Confusion Matrix"):
    """Generates heat map visualization of confusion matrix."""
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.xlabel("Predicted Disease Label", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth Disease Label", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure to '{save_path}'")

def plot_per_class_metrics(report_dict: Dict[str, Any], target_names: List[str], save_path: Path, title: str = "Per-Class Performance Metrics"):
    """Generates grouped bar chart comparing per-class Precision, Recall, and F1-Score."""
    records = []
    for cls_name in target_names:
        if cls_name in report_dict:
            records.append({
                "Disease Category": cls_name,
                "Precision": report_dict[cls_name]["precision"],
                "Recall": report_dict[cls_name]["recall"],
                "F1-Score": report_dict[cls_name]["f1-score"]
            })

    if not records:
        return

    df = pd.DataFrame(records)
    df_melted = df.melt(id_vars=["Disease Category"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_melted, y="Disease Category", x="Score", hue="Metric", palette="Set2")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Metric Value (0.0 to 1.0)", fontsize=12)
    plt.ylabel("Disease Category", fontsize=12)
    plt.xlim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved per-class metrics figure to '{save_path}'")

def plot_confidence_distribution(error_df: pd.DataFrame, save_path: Path, title: str = "Prediction Confidence Distribution"):
    """Generates distribution plot comparing prediction confidence scores for correct vs incorrect predictions."""
    plt.figure(figsize=(10, 6))

    if "is_correct" in error_df.columns:
        error_df["Prediction Type"] = error_df["is_correct"].map({True: "Correct Prediction", False: "Incorrect Prediction"})
        sns.histplot(
            data=error_df,
            x="confidence",
            hue="Prediction Type",
            kde=True,
            bins=20,
            palette={"Correct Prediction": "#2ca02c", "Incorrect Prediction": "#d62728"},
            alpha=0.6
        )
    else:
        sns.histplot(error_df["confidence"], kde=True, color="#1f77b4", bins=20)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Model Confidence Score (Softmax Probability)", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confidence distribution figure to '{save_path}'")

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path):
    """Generates comparative bar chart comparing BioBERT, BERT, and DistilBERT on test set metrics."""
    if comparison_df.empty:
        return

    metrics_cols = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
    present_cols = [c for c in metrics_cols if c in comparison_df.columns]

    df_melted = comparison_df.melt(id_vars=["Model"], value_vars=present_cols, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Model Comparison on Test Set (BERT vs BioBERT vs DistilBERT)", fontsize=14, fontweight="bold")
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotate bars
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.01), ha="center", va="bottom", fontsize=8, rotation=90)

    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison figure to '{save_path}'")
