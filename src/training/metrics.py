import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def compute_classification_metrics(y_true: List[int], y_pred: List[int], target_names: List[str] = None) -> Dict[str, Any]:
    """Computes comprehensive classification metrics (Accuracy, Precision, Recall, F1)."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0) if target_names else {}

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict
    }
