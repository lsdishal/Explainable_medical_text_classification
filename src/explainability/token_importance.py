import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.explainability.attention import get_attention_attributions
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.captum_explainer import CaptumExplainer
from src.utils.logger import logger

class XAIEngine:
    """Unified Explainable AI engine running Attention, SHAP, LIME, and Captum explanations."""

    def __init__(self, model, tokenizer, label_list: List[str], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device

        self.shap_explainer = SHAPExplainer(model, tokenizer, label_list, device)
        self.lime_explainer = LIMEExplainer(model, tokenizer, label_list, device)
        self.captum_explainer = CaptumExplainer(model, tokenizer, device)

    def explain_all(self, text: str, target_class_id: int) -> Dict[str, Any]:
        """Runs all 4 XAI attribution techniques on the given text and target class."""
        logger.info(f"Generating XAI explanations (Attention, SHAP, LIME, Integrated Gradients) for input text...")

        # 1. Attention Weights
        tokens, attention_matrix, att_info = get_attention_attributions(self.model, self.tokenizer, text, self.device)
        attention_scores = att_info["token_scores"]

        # 2. SHAP
        shap_scores = self.shap_explainer.explain(text)

        # 3. LIME
        lime_scores = self.lime_explainer.explain(text)

        # 4. Integrated Gradients (Captum)
        captum_scores = self.captum_explainer.explain(text, target_class=target_class_id)

        # Format comparison table
        summary_df = self._format_summary_table(tokens, attention_scores, shap_scores, lime_scores, captum_scores)

        return {
            "tokens": tokens,
            "attention_matrix": attention_matrix,
            "attention_scores": attention_scores,
            "shap_scores": shap_scores,
            "lime_scores": lime_scores,
            "captum_scores": captum_scores,
            "summary_df": summary_df
        }

    def _format_summary_table(self, tokens, attention, shap_list, lime_list, captum_list) -> pd.DataFrame:
        """Standardizes token scores across methods into a unified DataFrame."""
        att_dict = {t.replace("##", ""): s for t, s in attention if t not in ["[CLS]", "[SEP]", "[PAD]"]}
        shap_dict = {t: s for t, s in shap_list}
        lime_dict = {t: s for t, s in lime_list}
        captum_dict = {t: s for t, s in captum_list}

        all_words = list(dict.fromkeys(list(att_dict.keys()) + list(shap_dict.keys()) + list(lime_dict.keys()) + list(captum_dict.keys())))

        records = []
        for w in all_words:
            records.append({
                "Word": w,
                "Attention Score": float(att_dict.get(w, 0.0)),
                "SHAP Score": float(shap_dict.get(w, 0.0)),
                "LIME Score": float(lime_dict.get(w, 0.0)),
                "Integrated Gradients": float(captum_dict.get(w, 0.0))
            })

        df = pd.DataFrame(records)
        return df
