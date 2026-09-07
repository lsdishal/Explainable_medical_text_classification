import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
from src.utils.logger import logger

def analyze_model_errors(
    texts: List[str],
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs detailed error analysis on model predictions.
    
    Returns:
    - error_df: DataFrame containing all sample predictions sorted by confidence descending for errors.
    - summary_dict: Dictionary containing error counts, confused class pairs, and confidence statistics.
    """
    records = []
    confused_pairs_list = []

    correct_confidences = []
    incorrect_confidences = []

    for idx, (text, true_lbl, pred_lbl, conf) in enumerate(zip(texts, y_true, y_pred, confidences)):
        is_correct = (str(true_lbl).strip() == str(pred_lbl).strip())
        conf_val = float(conf)

        if is_correct:
            correct_confidences.append(conf_val)
        else:
            incorrect_confidences.append(conf_val)
            confused_pairs_list.append((str(true_lbl).strip(), str(pred_lbl).strip()))

        records.append({
            "sample_index": idx,
            "symptom_text": text,
            "ground_truth": str(true_lbl).strip(),
            "predicted_label": str(pred_lbl).strip(),
            "confidence": conf_val,
            "is_correct": is_correct,
            "is_error": not is_correct
        })

    df = pd.DataFrame(records)

    # Sort incorrect predictions by confidence descending for high-confidence error investigation
    error_df = df[df["is_error"]].sort_values(by="confidence", ascending=False).reset_index(drop=True)
    correct_df = df[df["is_correct"]].sort_values(by="confidence", ascending=False).reset_index(drop=True)
    full_sorted_df = pd.concat([error_df, correct_df], ignore_index=True)

    # Identify most confused class pairs
    pair_counts = Counter(confused_pairs_list)
    top_confused_pairs = [
        {"ground_truth": p[0], "predicted_as": p[1], "error_count": cnt}
        for p, cnt in pair_counts.most_common(10)
    ]

    total_samples = len(df)
    total_correct = int(df["is_correct"].sum())
    total_incorrect = int(df["is_error"].sum())

    summary_dict = {
        "total_samples": total_samples,
        "correct_count": total_correct,
        "incorrect_count": total_incorrect,
        "accuracy_pct": float(total_correct / total_samples * 100) if total_samples > 0 else 0.0,
        "top_confused_class_pairs": top_confused_pairs,
        "confidence_stats": {
            "correct_predictions": {
                "mean_confidence": float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                "median_confidence": float(np.median(correct_confidences)) if correct_confidences else 0.0,
                "std_confidence": float(np.std(correct_confidences)) if correct_confidences else 0.0
            },
            "incorrect_predictions": {
                "mean_confidence": float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                "median_confidence": float(np.median(incorrect_confidences)) if incorrect_confidences else 0.0,
                "std_confidence": float(np.std(incorrect_confidences)) if incorrect_confidences else 0.0
            }
        }
    }

    logger.info(
        f"Error analysis completed: Correct={total_correct}/{total_samples} | Incorrect={total_incorrect} | "
        f"Top Confused Pair: {top_confused_pairs[0] if top_confused_pairs else 'None'}"
    )

    return full_sorted_df, summary_dict

if __name__ == "__main__":
    texts = ["rash on skin", "cough and fever", "high blood pressure"]
    true_l = ["Psoriasis", "Pneumonia", "Hypertension"]
    pred_l = ["Psoriasis", "Common Cold", "Hypertension"]
    confs = [0.95, 0.88, 0.99]

    df, summary = analyze_model_errors(texts, true_l, pred_l, confs)
    print("Summary:", summary)
    print("Errors dataframe:\n", df[df["is_error"]])
