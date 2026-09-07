import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.utils.config import get_config
from src.data.load_dataset import load_raw_dataset
from src.preprocessing.text_cleaner import clean_medical_text, calculate_text_stats, is_outlier_length
from src.preprocessing.label_encoder import DiseaseLabelEncoder

def run_preprocessing_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Executes complete dataset preprocessing, auditing, label encoding, stratified splitting, and visualization generation."""
    cfg = get_config(config_path)
    processed_dir = Path(cfg.dataset.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(cfg.paths.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Dataset Preprocessing & Cleaning Pipeline ===")

    # 1. Load Raw Dataset & Audit
    raw_df, load_audit = load_raw_dataset(config_path)
    original_sample_count = len(raw_df)

    # Keep original samples for comparison chart
    comparison_samples = raw_df[["label", "text"]].head(5).copy()

    # 2. Drop Missing Values
    clean_df = raw_df.dropna(subset=["label", "text"]).copy()
    missing_removed = original_sample_count - len(clean_df)

    # 3. Clean Text Formatting
    clean_df["cleaned_text"] = clean_df["text"].apply(clean_medical_text)

    # Remove empty text entries post-cleaning
    clean_df = clean_df[clean_df["cleaned_text"].str.strip() != ""].copy()

    # 4. Remove Duplicates
    pre_dedup_count = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["label", "cleaned_text"]).reset_index(drop=True)
    duplicates_removed = pre_dedup_count - len(clean_df)

    cleaned_sample_count = len(clean_df)
    logger.info(f"Cleaning complete: Original={original_sample_count} | Missing Removed={missing_removed} | Duplicates Removed={duplicates_removed} | Cleaned Count={cleaned_sample_count}")

    # 5. Fit Label Encoder
    label_encoder = DiseaseLabelEncoder()
    clean_df["label_id"] = label_encoder.fit_transform(clean_df["label"].values)
    label_encoder.save(processed_dir / "label_mapping.json")

    # 6. Text Length Statistics & Outlier Audit
    stats_list = [calculate_text_stats(t) for t in clean_df["cleaned_text"]]
    clean_df["char_length"] = [s["char_length"] for s in stats_list]
    clean_df["word_count"] = [s["word_count"] for s in stats_list]

    outliers = [is_outlier_length(t) for t in clean_df["cleaned_text"]]
    clean_df["is_outlier"] = [o[0] for o in outliers]
    clean_df["outlier_type"] = [o[1] for o in outliers]

    text_statistics = {
        "char_length": {
            "min": int(clean_df["char_length"].min()),
            "max": int(clean_df["char_length"].max()),
            "mean": float(clean_df["char_length"].mean()),
            "median": float(clean_df["char_length"].median()),
            "std": float(clean_df["char_length"].std())
        },
        "word_count": {
            "min": int(clean_df["word_count"].min()),
            "max": int(clean_df["word_count"].max()),
            "mean": float(clean_df["word_count"].mean()),
            "median": float(clean_df["word_count"].median()),
            "std": float(clean_df["word_count"].std())
        },
        "outlier_counts": {
            "total_outliers": int(clean_df["is_outlier"].sum()),
            "unusually_short": int((clean_df["outlier_type"] == "unusually_short").sum()),
            "unusually_long": int((clean_df["outlier_type"] == "unusually_long").sum())
        }
    }

    # 7. Stratified Train / Validation / Test Split
    test_size = cfg.dataset.test_split # 0.10
    val_size = cfg.dataset.val_split / (cfg.dataset.train_split + cfg.dataset.val_split) # 0.10 / 0.90 = ~0.1111

    train_val_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=cfg.project.seed,
        stratify=clean_df["label_id"]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=cfg.project.seed,
        stratify=train_val_df["label_id"]
    )

    # Standardize columns for processed output CSV files
    out_cols = ["label", "label_id", "text", "cleaned_text", "char_length", "word_count"]
    
    train_out = train_df[out_cols].copy().rename(columns={"text": "raw_text", "cleaned_text": "text"})
    val_out = val_df[out_cols].copy().rename(columns={"text": "raw_text", "cleaned_text": "text"})
    test_out = test_df[out_cols].copy().rename(columns={"text": "raw_text", "cleaned_text": "text"})

    train_out.to_csv(processed_dir / "processed_train.csv", index=False)
    val_out.to_csv(processed_dir / "processed_validation.csv", index=False)
    test_out.to_csv(processed_dir / "processed_test.csv", index=False)

    # Class distribution statistics
    class_counts = clean_df["label"].value_counts().to_dict()
    train_class_counts = train_out["label"].value_counts().to_dict()
    val_class_counts = val_out["label"].value_counts().to_dict()
    test_class_counts = test_out["label"].value_counts().to_dict()

    dataset_stats = {
        "dataset_name": "Symptom2Disease",
        "total_cleaned_samples": cleaned_sample_count,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_,
        "samples_per_class": class_counts,
        "splits": {
            "train_samples": len(train_out),
            "val_samples": len(val_out),
            "test_samples": len(test_out),
            "train_ratio": len(train_out) / cleaned_sample_count,
            "val_ratio": len(val_out) / cleaned_sample_count,
            "test_ratio": len(test_out) / cleaned_sample_count
        },
        "text_statistics": text_statistics
    }

    with open(processed_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)

    # 8. Preprocessing Comprehensive Audit Report
    report = {
        "original_sample_count": original_sample_count,
        "cleaned_sample_count": cleaned_sample_count,
        "duplicates_removed": duplicates_removed,
        "missing_values_removed": missing_removed,
        "num_classes": len(label_encoder.classes_),
        "train_size": len(train_out),
        "validation_size": len(val_out),
        "test_size": len(test_out),
        "samples_per_class": class_counts,
        "train_class_distribution": train_class_counts,
        "val_class_distribution": val_class_counts,
        "test_class_distribution": test_class_counts
    }

    with open(reports_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 9. Generate Visualizations
    _plot_class_distribution(clean_df, fig_dir / "class_distribution.png")
    _plot_text_length_distribution(clean_df, fig_dir / "text_length_distribution.png")
    _plot_before_after_comparison(comparison_samples, fig_dir / "sample_preprocessing_comparison.png")

    logger.info("Preprocessing pipeline finished successfully. All datasets, statistics, and figures exported.")
    return report

def _plot_class_distribution(df: pd.DataFrame, save_path: Path):
    """Generates bar chart visualization of sample count per disease category."""
    plt.figure(figsize=(12, 6))
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Disease Label", "Sample Count"]

    ax = sns.barplot(data=counts, y="Disease Label", x="Sample Count", palette="viridis")
    plt.title("Symptom2Disease: Disease Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Symptom Samples", fontsize=12)
    plt.ylabel("Disease Category", fontsize=12)

    # Annotate bars with values
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 0.5, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _plot_text_length_distribution(df: pd.DataFrame, save_path: Path):
    """Generates histogram and boxplot visualization of character and word counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["word_count"], kde=True, ax=axes[0], color="teal", bins=25)
    axes[0].set_title("Word Count Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Words per Sample")
    axes[0].set_ylabel("Frequency")

    sns.histplot(df["char_length"], kde=True, ax=axes[1], color="coral", bins=25)
    axes[1].set_title("Character Length Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Characters per Sample")
    axes[1].set_ylabel("Frequency")

    plt.suptitle("Medical Symptom Text Length Distribution Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _plot_before_after_comparison(df: pd.DataFrame, save_path: Path):
    """Generates visual comparison table of text before and after cleaning."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table_data = [["Label", "Original Raw Text", "Cleaned & Normalized Text"]]
    for _, row in df.iterrows():
        raw_text = str(row["text"])
        cleaned_text = clean_medical_text(raw_text)

        # Truncate text for display formatting
        raw_disp = (raw_text[:45] + "...") if len(raw_text) > 45 else raw_text
        clean_disp = (cleaned_text[:45] + "...") if len(cleaned_text) > 45 else cleaned_text

        table_data.append([str(row["label"]), raw_disp, clean_disp])

    table = ax.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(3):
        cell = table[0, i]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#2c3e50")

    plt.title("Sample Comparison: Before vs After Medical Text Preprocessing", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    run_preprocessing_pipeline()
