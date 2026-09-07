import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger
from src.utils.config import get_config

class ConfidenceCalibrator:
    """Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and generates reliability plots."""

    def __init__(self, n_bins: int = 10, config_path: str = "config.yaml"):
        self.n_bins = n_bins
        self.cfg = get_config(config_path)

    def compute_calibration(self, confidences: List[float], correct_flags: List[bool]) -> Dict[str, Any]:
        """Calculates ECE, MCE, and per-bin calibration statistics."""
        confidences = np.array(confidences, dtype=float)
        correct_flags = np.array(correct_flags, dtype=bool)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        bin_stats = []

        total_samples = len(confidences)

        for bin_idx, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Find samples in this bin
            in_bin = (confidences > lower) & (confidences <= upper) if bin_idx > 0 else (confidences >= lower) & (confidences <= upper)
            bin_size = int(np.sum(in_bin))

            if bin_size > 0:
                bin_acc = float(np.mean(correct_flags[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                bin_gap = abs(bin_acc - bin_conf)

                ece += (bin_size / total_samples) * bin_gap
                mce = max(mce, bin_gap)
            else:
                bin_acc = 0.0
                bin_conf = float((lower + upper) / 2.0)
                bin_gap = 0.0

            bin_stats.append({
                "bin_index": bin_idx,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_size": bin_size,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": bin_gap
            })

        metrics = {
            "total_samples": total_samples,
            "n_bins": self.n_bins,
            "expected_calibration_error_ece": float(ece),
            "maximum_calibration_error_mce": float(mce),
            "bin_statistics": bin_stats
        }

        logger.info(f"Calibration metrics computed: ECE={ece:.4f} | MCE={mce:.4f} across {self.n_bins} bins.")
        return metrics

    def generate_reliability_plots(
        self,
        confidences: List[float],
        correct_flags: List[bool],
        fig_dir: str = "outputs/figures",
        reports_dir: str = "outputs/reports"
    ) -> Dict[str, Path]:
        """Generates Reliability Diagram, Confidence Histogram, and Accuracy vs Confidence plot."""
        fig_path = Path(fig_dir)
        fig_path.mkdir(parents=True, exist_ok=True)
        rep_path = Path(reports_dir)
        rep_path.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_calibration(confidences, correct_flags)
        
        # Save JSON calibration report
        with open(rep_path / "calibration_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        bin_stats = metrics["bin_statistics"]
        confs_arr = np.array(confidences)
        corrects_arr = np.array(correct_flags)

        # 1. Reliability Diagram
        p1 = fig_path / "reliability_diagram.png"
        self._plot_reliability_diagram(bin_stats, metrics["expected_calibration_error_ece"], p1)

        # 2. Confidence Histogram
        p2 = fig_path / "confidence_histogram.png"
        self._plot_confidence_histogram(confs_arr, corrects_arr, p2)

        # 3. Accuracy vs Confidence Threshold Curve
        p3 = fig_path / "accuracy_vs_confidence.png"
        self._plot_accuracy_vs_confidence(confs_arr, corrects_arr, p3)

        return {"reliability_diagram": p1, "confidence_histogram": p2, "accuracy_vs_confidence": p3}

    def _plot_reliability_diagram(self, bin_stats: List[Dict[str, Any]], ece: float, save_path: Path):
        """Plots binned confidence vs accuracy reliability diagram."""
        plt.figure(figsize=(8, 7))

        bin_confs = [b["confidence"] for b in bin_stats if b["bin_size"] > 0]
        bin_accs = [b["accuracy"] for b in bin_stats if b["bin_size"] > 0]

        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
        plt.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#1f77b4", edgecolor="black", label="Model Outputs")
        
        plt.title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=14, fontweight="bold")
        plt.xlabel("Average Prediction Confidence", fontsize=12)
        plt.ylabel("Empirical Accuracy", fontsize=12)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_confidence_histogram(self, confidences: np.ndarray, corrects: np.ndarray, save_path: Path):
        """Plots histogram of model confidence distributions for correct vs incorrect predictions."""
        plt.figure(figsize=(9, 6))

        sns.histplot(confidences[corrects], color="#2ca02c", label="Correct Predictions", bins=20, alpha=0.6, kde=True)
        sns.histplot(confidences[~corrects], color="#d62728", label="Incorrect Predictions", bins=20, alpha=0.6, kde=True)

        plt.title("Model Prediction Confidence Histogram", fontsize=14, fontweight="bold")
        plt.xlabel("Softmax Confidence Score", fontsize=12)
        plt.ylabel("Sample Count", fontsize=12)
        plt.xlim(0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_accuracy_vs_confidence(self, confidences: np.ndarray, corrects: np.ndarray, save_path: Path):
        """Plots curve showing accuracy and sample retention as confidence threshold increases."""
        thresholds = np.linspace(0.0, 0.95, 20)
        accuracies = []
        retention = []

        for thresh in thresholds:
            mask = confidences >= thresh
            if np.sum(mask) > 0:
                acc = np.mean(corrects[mask])
                ret = np.sum(mask) / len(confidences)
            else:
                acc = 1.0
                ret = 0.0

            accuracies.append(acc * 100)
            retention.append(ret * 100)

        fig, ax1 = plt.subplots(figsize=(9, 6))

        color = "#1f77b4"
        ax1.set_xlabel("Confidence Threshold", fontsize=12)
        ax1.set_ylabel("Accuracy (%)", color=color, fontsize=12)
        ax1.plot(thresholds, accuracies, color=color, marker="o", linewidth=2, label="Accuracy")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(0, 105)

        ax2 = ax1.twinx()
        color = "#ff7f0e"
        ax2.set_ylabel("Data Retention (%)", color=color, fontsize=12)
        ax2.plot(thresholds, retention, color=color, linestyle="--", marker="s", linewidth=2, label="Retention")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(0, 105)

        plt.title("Accuracy vs Confidence Threshold Trade-off", fontsize=14, fontweight="bold")
        fig.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    confs = [0.95, 0.88, 0.92, 0.45, 0.60, 0.99, 0.85, 0.70]
    corrects = [True, True, True, False, False, True, True, False]

    calibrator = ConfidenceCalibrator()
    paths = calibrator.generate_reliability_plots(confs, corrects)
    print("Reliability plot paths:", paths)
