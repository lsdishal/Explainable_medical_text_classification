import os
import json
import string
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from scipy.stats import pearsonr, spearmanr

from src.utils.logger import logger
from src.utils.config import get_config
from src.inference.predict import MedicalSymptomClassifier
from src.explainability.explanation_manager import ExplanationManager

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SPECIAL_TOKENS = {"[cls]", "[sep]", "[pad]", "<s>", "</s>", "<pad>", "<unk>", "[unk]"}

def normalize_token(token: str) -> str:
    """Normalizes a single token by stripping subword prefixes, punctuation, and converting to lowercase."""
    tok = token.lower().strip()
    if tok in SPECIAL_TOKENS:
        return ""
    if tok.startswith("##"):
        tok = tok[2:]
    if tok.startswith("Ġ"):
        tok = tok[1:]
    tok = tok.strip(string.punctuation + " ")
    if tok in SPECIAL_TOKENS or not tok:
        return ""
    return tok

def extract_top_k_tokens(token_scores: List[Tuple[str, float]], k: int = 10) -> Set[str]:
    """Extracts top-K unique normalized tokens based on attribution magnitude."""
    normalized_dict = {}
    for tok, score in token_scores:
        norm = normalize_token(tok)
        if norm:
            # Keep maximum score if duplicate subword/tokens map to same normalized word
            if norm not in normalized_dict or abs(score) > abs(normalized_dict[norm]):
                normalized_dict[norm] = abs(score)

    sorted_tokens = sorted(normalized_dict.items(), key=lambda x: x[1], reverse=True)
    return set([t for t, _ in sorted_tokens[:k]])

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard Similarity metric: J(A, B) = |A ∩ B| / |A ∪ B|."""
    union_size = len(set_a.union(set_b))
    if union_size == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / float(union_size)

class ResearchXAIEvaluator:
    """Executes research-level explainability evaluation across Attention, SHAP, LIME, and Integrated Gradients."""

    def __init__(self, config_path: str = "config.yaml", sample_size: int = 50, top_k: int = 10, seed: int = 42):
        self.cfg = get_config(config_path)
        self.sample_size = sample_size
        self.top_k = top_k
        self.seed = seed
        
        self.classifier = MedicalSymptomClassifier(config_path=config_path)
        self.model = self.classifier.model
        self.tokenizer = self.classifier.tokenizer
        self.label_list = self.classifier.label_list
        self.device = self.classifier.device

        self.explainer = ExplanationManager(self.model, self.tokenizer, self.label_list, self.device)

        self.reports_dir = Path(self.cfg.paths.reports_dir)
        self.figures_dir = Path(self.cfg.paths.figures_dir) / "xai"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_test_dataset(self) -> pd.DataFrame:
        """Loads test set and selects reproducible subset using fixed seed."""
        test_path = Path(self.cfg.dataset.processed_dir) / "processed_test.csv"
        if not test_path.exists():
            test_path = Path(self.cfg.dataset.processed_dir) / "test.csv"
            
        test_df = pd.read_csv(test_path)
        text_col = "text" if "text" in test_df.columns else "cleaned_text"

        if len(test_df) > self.sample_size:
            sampled_df = test_df.sample(n=self.sample_size, random_state=self.seed).reset_index(drop=True)
        else:
            sampled_df = test_df.reset_index(drop=True)

        return sampled_df, text_col

    def evaluate_stability(self, stability_samples: List[str], num_evals: int = 15) -> Dict[str, float]:
        """Evaluates method stability by running explanations twice with identical configuration."""
        logger.info(f"=== Evaluating Explanation Stability on {num_evals} samples ===")
        stability_results = {"Attention": [], "SHAP": [], "LIME": [], "Integrated Gradients": []}

        eval_texts = stability_samples[:num_evals]
        for idx, text in enumerate(eval_texts):
            pred_info = self.classifier.predict(text)
            top_pred = pred_info["top_3_predictions"][0]
            cid, cname, conf = top_pred["class_id"], top_pred["disease"], top_pred["confidence"]

            # Run 1 (uncached)
            xai1 = self.explainer.explain(text, cid, cname, conf, use_cache=False)
            top1_att = extract_top_k_tokens(xai1["attention_importance"].get("token_scores", []), k=self.top_k)
            top1_shap = extract_top_k_tokens(xai1["shap_importance"].get("token_scores", []), k=self.top_k)
            top1_lime = extract_top_k_tokens(xai1["lime_importance"].get("token_scores", []), k=self.top_k)
            top1_ig = extract_top_k_tokens(xai1["integrated_gradients_importance"].get("token_scores", []), k=self.top_k)

            # Run 2 (uncached)
            xai2 = self.explainer.explain(text, cid, cname, conf, use_cache=False)
            top2_att = extract_top_k_tokens(xai2["attention_importance"].get("token_scores", []), k=self.top_k)
            top2_shap = extract_top_k_tokens(xai2["shap_importance"].get("token_scores", []), k=self.top_k)
            top2_lime = extract_top_k_tokens(xai2["lime_importance"].get("token_scores", []), k=self.top_k)
            top2_ig = extract_top_k_tokens(xai2["integrated_gradients_importance"].get("token_scores", []), k=self.top_k)

            stability_results["Attention"].append(jaccard_similarity(top1_att, top2_att))
            stability_results["SHAP"].append(jaccard_similarity(top1_shap, top2_shap))
            stability_results["LIME"].append(jaccard_similarity(top1_lime, top2_lime))
            stability_results["Integrated Gradients"].append(jaccard_similarity(top1_ig, top2_ig))

        mean_stability = {m: float(np.mean(scores)) for m, scores in stability_results.items()}
        return mean_stability

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes full research explainability evaluation pipeline."""
        df_test, text_col = self.load_test_dataset()
        logger.info(f"=== Running Research XAI Evaluation on {len(df_test)} Test Examples (K={self.top_k}) ===")

        eval_records = []
        method_pairs = [
            ("Attention", "SHAP"),
            ("Attention", "LIME"),
            ("Attention", "Integrated Gradients"),
            ("SHAP", "LIME"),
            ("SHAP", "Integrated Gradients"),
            ("LIME", "Integrated Gradients")
        ]

        pair_jaccards = {f"{m1} vs {m2}": [] for m1, m2 in method_pairs}
        hit_counts_distribution = {1: 0, 2: 0, 3: 0, 4: 0}

        for idx, row in df_test.iterrows():
            text = row[text_col]
            true_label = row["label"] if "label" in row else row.get("disease", "Unknown")

            # Inference
            pred_res = self.classifier.predict(text)
            top_pred = pred_res["top_3_predictions"][0]
            pred_label = top_pred["disease"]
            conf = float(top_pred["confidence"])
            is_correct = (str(true_label).strip().lower() == str(pred_label).strip().lower())

            # XAI Explicability
            xai_out = self.explainer.explain(text, top_pred["class_id"], pred_label, conf, use_cache=True)

            top_att = extract_top_k_tokens(xai_out["attention_importance"].get("token_scores", []), k=self.top_k)
            top_shap = extract_top_k_tokens(xai_out["shap_importance"].get("token_scores", []), k=self.top_k)
            top_lime = extract_top_k_tokens(xai_out["lime_importance"].get("token_scores", []), k=self.top_k)
            top_ig = extract_top_k_tokens(xai_out["integrated_gradients_importance"].get("token_scores", []), k=self.top_k)

            method_tokens = {
                "Attention": top_att,
                "SHAP": top_shap,
                "LIME": top_lime,
                "Integrated Gradients": top_ig
            }

            # Calculate pairwise Jaccards
            example_pair_scores = {}
            for m1, m2 in method_pairs:
                key = f"{m1} vs {m2}"
                j_score = jaccard_similarity(method_tokens[m1], method_tokens[m2])
                example_pair_scores[key] = j_score
                pair_jaccards[key].append(j_score)

            # Overall Agreement Score for example i = mean of the 6 pairwise Jaccard similarities
            example_agreement = float(np.mean(list(example_pair_scores.values())))

            # 4-way token hit counts
            all_unique_tokens = set().union(top_att, top_shap, top_lime, top_ig)
            token_hits = {}
            for tok in all_unique_tokens:
                hits = sum([1 for m_tokens in method_tokens.values() if tok in m_tokens])
                token_hits[tok] = hits
                if hits in hit_counts_distribution:
                    hit_counts_distribution[hits] += 1

            consensus_tokens = sorted([t for t, h in token_hits.items() if h >= 2])

            rec = {
                "sample_id": idx,
                "input_text": text,
                "true_disease": true_label,
                "predicted_disease": pred_label,
                "confidence": conf,
                "is_correct": is_correct,
                "agreement_score": example_agreement,
                "top_attention": sorted(list(top_att)),
                "top_shap": sorted(list(top_shap)),
                "top_lime": sorted(list(top_lime)),
                "top_integrated_gradients": sorted(list(top_ig)),
                "consensus_tokens": consensus_tokens,
                "pairwise_jaccard": example_pair_scores
            }
            eval_records.append(rec)

        eval_df = pd.DataFrame(eval_records)

        # Statistical Aggregations
        agreement_scores = eval_df["agreement_score"].values
        mean_agree = float(np.mean(agreement_scores))
        median_agree = float(np.median(agreement_scores))
        min_agree = float(np.min(agreement_scores))
        max_agree = float(np.max(agreement_scores))
        std_agree = float(np.std(agreement_scores))

        # Correct vs Incorrect Split
        correct_df = eval_df[eval_df["is_correct"] == True]
        incorrect_df = eval_df[eval_df["is_correct"] == False]
        mean_agree_correct = float(correct_df["agreement_score"].mean()) if not correct_df.empty else 0.0
        mean_agree_incorrect = float(incorrect_df["agreement_score"].mean()) if not incorrect_df.empty else 0.0

        # Confidence vs Agreement Correlation
        confidences = eval_df["confidence"].values
        p_corr, p_val = pearsonr(confidences, agreement_scores)
        s_corr, s_val = spearmanr(confidences, agreement_scores)

        # Stability Analysis
        stability_scores = self.evaluate_stability(df_test[text_col].tolist(), num_evals=15)

        # Mean Pairwise Matrix
        mean_pairwise_matrix = {k: float(np.mean(v)) for k, v in pair_jaccards.items()}

        # Select Case Studies
        high_conf_correct = correct_df.sort_values(by="confidence", ascending=False).head(3).to_dict(orient="records")
        low_conf_correct = correct_df.sort_values(by="confidence", ascending=True).head(3).to_dict(orient="records")
        incorrect_cases = incorrect_df.head(3).to_dict(orient="records")

        # Compile Summary Dict
        summary_results = {
            "num_test_examples": len(df_test),
            "top_k": self.top_k,
            "overall_agreement_stats": {
                "mean": mean_agree,
                "median": median_agree,
                "min": min_agree,
                "max": max_agree,
                "std": std_agree
            },
            "correct_vs_incorrect_agreement": {
                "correct_count": len(correct_df),
                "mean_agreement_correct": mean_agree_correct,
                "incorrect_count": len(incorrect_df),
                "mean_agreement_incorrect": mean_agree_incorrect
            },
            "correlation_confidence_vs_agreement": {
                "pearson_r": float(p_corr),
                "pearson_p_value": float(p_val),
                "spearman_rho": float(s_corr),
                "spearman_p_value": float(s_val)
            },
            "method_stability": stability_scores,
            "pairwise_jaccard_means": mean_pairwise_matrix,
            "four_way_hit_counts": hit_counts_distribution,
            "case_studies": {
                "high_confidence_correct": high_conf_correct,
                "low_confidence_correct": low_conf_correct,
                "incorrect_predictions": incorrect_cases
            }
        }

        # Save Outputs
        eval_df.to_csv(self.reports_dir / "xai_evaluation.csv", index=False)
        with open(self.reports_dir / "xai_agreement.json", "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)

        # Generate Figures
        self._generate_plots(eval_df, mean_pairwise_matrix, stability_scores, hit_counts_distribution)

        # Generate Markdown Report
        self._generate_markdown_report(summary_results)

        logger.info(f"Research XAI evaluation complete. Mean Agreement: {mean_agree:.4f} | Pearson r: {p_corr:.4f}")
        return summary_results

    def _generate_plots(self, eval_df: pd.DataFrame, pairwise_means: Dict[str, float], stability: Dict[str, float], hits_dist: Dict[int, int]):
        """Generates all 5 required XAI figures and saves under outputs/figures/xai/."""
        sns.set_theme(style="whitegrid")

        # 1. Pairwise Agreement Heatmap
        methods_list = ["Attention", "SHAP", "LIME", "Integrated Gradients"]
        heatmap_matrix = np.ones((4, 4))
        mapping = {
            ("Attention", "SHAP"): pairwise_means["Attention vs SHAP"],
            ("Attention", "LIME"): pairwise_means["Attention vs LIME"],
            ("Attention", "Integrated Gradients"): pairwise_means["Attention vs Integrated Gradients"],
            ("SHAP", "LIME"): pairwise_means["SHAP vs LIME"],
            ("SHAP", "Integrated Gradients"): pairwise_means["SHAP vs Integrated Gradients"],
            ("LIME", "Integrated Gradients"): pairwise_means["LIME vs Integrated Gradients"]
        }
        for i, m1 in enumerate(methods_list):
            for j, m2 in enumerate(methods_list):
                if i < j:
                    score = mapping.get((m1, m2), mapping.get((m2, m1), 0.0))
                    heatmap_matrix[i, j] = score
                    heatmap_matrix[j, i] = score

        plt.figure(figsize=(7, 6))
        sns.heatmap(heatmap_matrix, annot=True, fmt=".3f", cmap="Greens", xticklabels=methods_list, yticklabels=methods_list, vmin=0, vmax=1)
        plt.title("Pairwise Explanation Agreement (Jaccard Similarity)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "pairwise_agreement_heatmap.png", dpi=300)
        plt.close()

        # 2. Method Stability Bar Chart
        plt.figure(figsize=(8, 5))
        stab_df = pd.DataFrame(list(stability.items()), columns=["Method", "Stability"])
        ax = sns.barplot(data=stab_df, x="Method", y="Stability", palette="Greens_r")
        plt.ylim(0, 1.05)
        plt.title("Method Self-Stability (Token-Set Overlap Across Runs)", fontsize=12, fontweight="bold")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "method_stability.png", dpi=300)
        plt.close()

        # 3. Confidence vs Agreement Scatter Plot
        plt.figure(figsize=(8, 5))
        sns.regplot(data=eval_df, x="confidence", y="agreement_score", color="#2e7d32", scatter_kws={'alpha':0.7, 's':40})
        plt.title("Model Confidence vs Explanation Agreement Score", fontsize=12, fontweight="bold")
        plt.xlabel("Model Prediction Confidence")
        plt.ylabel("Mean Pairwise Agreement Score")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confidence_vs_agreement.png", dpi=300)
        plt.close()

        # 4. Correct vs Incorrect Agreement Bar Chart
        plt.figure(figsize=(6, 5))
        correct_mean = eval_df[eval_df["is_correct"] == True]["agreement_score"].mean()
        incorrect_mean = eval_df[eval_df["is_correct"] == False]["agreement_score"].mean()
        
        comp_df = pd.DataFrame([
            {"Prediction Status": "Correct", "Mean Agreement": correct_mean},
            {"Prediction Status": "Incorrect", "Mean Agreement": incorrect_mean}
        ])
        ax = sns.barplot(data=comp_df, x="Prediction Status", y="Mean Agreement", palette=["#2e7d32", "#d32f2f"])
        plt.ylim(0, 1.0)
        plt.title("Explanation Agreement: Correct vs Incorrect Predictions", fontsize=12, fontweight="bold")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correct_vs_incorrect_agreement.png", dpi=300)
        plt.close()

        # 5. Four-Way Agreement Distribution
        plt.figure(figsize=(8, 5))
        dist_df = pd.DataFrame(list(hits_dist.items()), columns=["Number of Methods Flagging Token", "Token Count"])
        ax = sns.barplot(data=dist_df, x="Number of Methods Flagging Token", y="Token Count", palette="Greens_d")
        plt.title("Four-Way Agreement Distribution Across Tokens", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Methods Identifying Token in Top-10")
        plt.ylabel("Total Token Occurrences")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "four_way_agreement_distribution.png", dpi=300)
        plt.close()

    def _generate_markdown_report(self, summary: Dict[str, Any]):
        """Generates comprehensive outputs/reports/xai_evaluation.md report."""
        stats = summary["overall_agreement_stats"]
        c_v_i = summary["correct_vs_incorrect_agreement"]
        corr = summary["correlation_confidence_vs_agreement"]
        stab = summary["method_stability"]
        pw = summary["pairwise_jaccard_means"]
        cases = summary["case_studies"]

        # Identify key scientific conclusions
        most_stable = max(stab.items(), key=lambda x: x[1])
        least_stable = min(stab.items(), key=lambda x: x[1])
        top_pair = max(pw.items(), key=lambda x: x[1])
        bottom_pair = min(pw.items(), key=lambda x: x[1])

        md = f"""# Research-Level Explainability Evaluation Report

## 1. Executive Summary
This report presents a formal, empirical research evaluation of explanation consistency and stability across **four distinct Explainable AI (XAI) attribution methods** on the fine-tuned **BioBERT** model:
1. **Model Attention Signal** (Multi-head self-attention rollout/CLS signal)
2. **SHAP** (Kernel / Partition Shapley additive feature attributions)
3. **LIME** (Local interpretable model-agnostic explanations with 250 perturbations)
4. **Integrated Gradients** (Captum axiomatic path-integral gradient attributions)

---

## 2. Experimental Setup & Token Normalization

- **Evaluated Sample Size**: {summary['num_test_examples']} held-out test set examples (sampled with seed `42`)
- **Top-K Parameter ($K$)**: `{summary['top_k']}` tokens per method
- **Token Normalization Layer**:
  - **Special Token Filtering**: Stripped `[CLS]`, `[SEP]`, `[PAD]`, `<s>`, `</s>`, `<pad>`.
  - **Subword Prefix Removal**: Stripped Hugging Face WordPiece `##` and Byte-Pair `Ġ` prefixes.
  - **Punctuation & Clean-up**: Stripped surrounding punctuation (`.`, `,`, `!`, `?`, `"`, `'`).
  - **Lowercase Standardization**: Standardized all tokens to lowercase representation.
  - **Subword Max Aggregation**: Aggregated subwords mapping to identical root words by taking the maximum attribution magnitude.

---

"""
        md += r"""## 3. Pairwise Method Agreement (Jaccard Similarity)

Pairwise agreement is calculated using the **Jaccard Similarity Index**:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

| XAI Method Pair | Mean Jaccard Similarity |
|---|:---:|
| **Attention vs SHAP** | """ + f"`{pw['Attention vs SHAP']:.4f}`" + r""" |
| **Attention vs LIME** | """ + f"`{pw['Attention vs LIME']:.4f}`" + r""" |
| **Attention vs Integrated Gradients** | """ + f"`{pw['Attention vs Integrated Gradients']:.4f}`" + r""" |
| **SHAP vs LIME** | """ + f"`{pw['SHAP vs LIME']:.4f}`" + r""" |
| **SHAP vs Integrated Gradients** | """ + f"`{pw['SHAP vs Integrated Gradients']:.4f}`" + r""" |
| **LIME vs Integrated Gradients** | """ + f"`{pw['LIME vs Integrated Gradients']:.4f}`" + r""" |

> **Highest Agreement Pair**: """ + f"`{top_pair[0]}` (Jaccard: `{top_pair[1]:.4f}`)" + r"""  
> **Most Divergent Pair**: """ + f"`{bottom_pair[0]}` (Jaccard: `{bottom_pair[1]:.4f}`)" + r"""

---

## 4. Overall Explanation Agreement Score

For each test example $i$, the **Agreement Score** $A_i$ is defined as the mean pairwise Jaccard similarity across all 6 method pairs:
$$A_i = \frac{1}{6} \sum_{(M_1, M_2) \in \text{Pairs}} J(S_{M_1, i}, S_{M_2, i})$$

- **Mean Agreement ($\bar{A}$)**: """ + f"`{stats['mean']:.4f}`\n" + r"""
- **Median Agreement**: """ + f"`{stats['median']:.4f}`\n" + r"""
- **Min Agreement**: """ + f"`{stats['min']:.4f}`\n" + r"""
- **Max Agreement**: """ + f"`{stats['max']:.4f}`\n" + r"""
- **Standard Deviation ($\sigma$)**: """ + f"`{stats['std']:.4f}`\n" + r"""

---

## 5. Explanation Stability Analysis

Method self-stability measures the repeatability of token attributions when running each algorithm twice on the exact same input and model configuration ($J(Run_1, Run_2)$):

| XAI Method | Self-Stability (Jaccard Overlap) | Notes |
|---|:---:|---|
| **Attention** | """ + f"`{stab['Attention']:.4f}`" + r""" | Completely deterministic (exact matrix rollout) |
| **SHAP** | """ + f"`{stab['SHAP']:.4f}`" + r""" | Deterministic partitioning |
| **Integrated Gradients** | """ + f"`{stab['Integrated Gradients']:.4f}`" + r""" | Axiomatic gradient integral (deterministic steps) |
| **LIME** | """ + f"`{stab['LIME']:.4f}`" + r""" | Perturbation-based (250 random samples) |

> **Most Stable Method**: """ + f"`{most_stable[0]}` (Stability: `{most_stable[1]:.4f}`)" + r"""  
> **Least Stable Method**: """ + f"`{least_stable[0]}` (Stability: `{least_stable[1]:.4f}`)" + r"""

---

## 6. Correct vs Incorrect Prediction Comparison

| Group | Sample Count | Mean Agreement Score |
|---|:---:|:---:|
| **Correct Predictions** | """ + f"`{c_v_i['correct_count']}` | `{c_v_i['mean_agreement_correct']:.4f}`" + r""" |
| **Incorrect Predictions** | """ + f"`{c_v_i['incorrect_count']}` | `{c_v_i['mean_agreement_incorrect']:.4f}`" + r""" |

---

## 7. Model Confidence vs Explanation Agreement

- **Pearson Correlation ($r$)**: """ + f"`{corr['pearson_r']:.4f}` ($p$-value: `{corr['pearson_p_value']:.4e}`)\n" + r"""
- **Spearman Rank Correlation ($\rho$)**: """ + f"`{corr['spearman_rho']:.4f}` ($p$-value: `{corr['spearman_p_value']:.4e}`)\n" + r"""

---

## 8. Representative Case Studies

### A. High-Confidence Correct Predictions
"""
        for i, cs in enumerate(cases["high_confidence_correct"], 1):
            md += f"""
#### Case {i}: "{cs['input_text']}"
- **True Disease**: `{cs['true_disease']}` | **Predicted**: `{cs['predicted_disease']}` (Conf: `{cs['confidence']*100:.2f}%`)
- **Attention Top Tokens**: `{', '.join(cs['top_attention'])}`
- **SHAP Top Tokens**: `{', '.join(cs['top_shap'])}`
- **LIME Top Tokens**: `{', '.join(cs['top_lime'])}`
- **Integrated Gradients Top Tokens**: `{', '.join(cs['top_integrated_gradients'])}`
- **Consensus Tokens ($\ge 2$ methods)**: `{', '.join(cs['consensus_tokens']) if cs['consensus_tokens'] else 'None'}`
"""

        md += "\n### B. Low-Confidence Correct Predictions\n"
        for i, cs in enumerate(cases["low_confidence_correct"], 1):
            md += f"""
#### Case {i}: "{cs['input_text']}"
- **True Disease**: `{cs['true_disease']}` | **Predicted**: `{cs['predicted_disease']}` (Conf: `{cs['confidence']*100:.2f}%`)
- **Attention Top Tokens**: `{', '.join(cs['top_attention'])}`
- **SHAP Top Tokens**: `{', '.join(cs['top_shap'])}`
- **LIME Top Tokens**: `{', '.join(cs['top_lime'])}`
- **Integrated Gradients Top Tokens**: `{', '.join(cs['top_integrated_gradients'])}`
- **Consensus Tokens ($\ge 2$ methods)**: `{', '.join(cs['consensus_tokens']) if cs['consensus_tokens'] else 'None'}`
"""

        if cases["incorrect_predictions"]:
            md += "\n### C. Incorrect Predictions\n"
            for i, cs in enumerate(cases["incorrect_predictions"], 1):
                md += f"""
#### Case {i}: "{cs['input_text']}"
- **True Disease**: `{cs['true_disease']}` | **Predicted**: `{cs['predicted_disease']}` (Conf: `{cs['confidence']*100:.2f}%`)
- **Attention Top Tokens**: `{', '.join(cs['top_attention'])}`
- **SHAP Top Tokens**: `{', '.join(cs['top_shap'])}`
- **LIME Top Tokens**: `{', '.join(cs['top_lime'])}`
- **Integrated Gradients Top Tokens**: `{', '.join(cs['top_integrated_gradients'])}`
- **Consensus Tokens ($\ge 2$ methods)**: `{', '.join(cs['consensus_tokens']) if cs['consensus_tokens'] else 'None'}`
"""

        md += """
---

## 9. Important Scientific & Medical Limitations

1. **Non-Causality of Attention**: Attention maps reflect contextual representation alignment inside BioBERT's transformer layers. They do not constitute a causal explanation of model prediction.
2. **Attribution Convergence vs Medical Causality**: High agreement across XAI methods indicates that multiple algorithmic surrogate perspectives focus on the same symptom tokens. It does **NOT** prove medical or clinical causality.
3. **Surrogate Perturbation Noise**: Perturbation-based explainers (such as LIME) exhibit slight non-deterministic variance across runs due to stochastic sampling around local decision boundaries.
"""

        with open(self.reports_dir / "xai_evaluation.md", "w", encoding="utf-8") as f:
            f.write(md)

def main():
    evaluator = ResearchXAIEvaluator(sample_size=50, top_k=10, seed=42)
    evaluator.run_pipeline()

if __name__ == "__main__":
    main()
