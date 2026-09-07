import hashlib
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from src.utils.logger import logger
from src.explainability.attention import get_attention_signal
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.integrated_gradients import CaptumIGExplainer
from src.explainability.subword_aggregator import aggregate_subwords

class ExplanationManager:
    """Unified Explainable AI manager executing Attention, SHAP, LIME, and Captum Integrated Gradients with subword aggregation."""

    def __init__(self, model, tokenizer, label_list: List[str], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device

        self.shap_explainer = SHAPExplainer(model, tokenizer, label_list, device)
        self.lime_explainer = LIMEExplainer(model, tokenizer, label_list, device)
        self.captum_explainer = CaptumIGExplainer(model, tokenizer, device)

        # In-memory explanation cache
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _generate_cache_key(self, text: str, target_class_id: int) -> str:
        """Generates MD5 hash key for caching explanation outputs."""
        raw_str = f"{text.strip().lower()}__class_{target_class_id}"
        return hashlib.md5(raw_str.encode("utf-8")).hexdigest()

    def explain(
        self,
        text: str,
        target_class_id: int,
        target_class_name: str,
        confidence: float,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generates unified XAI explanations with both Raw Tokens and Human-Readable Word Attributions."""
        cache_key = self._generate_cache_key(text, target_class_id)
        if use_cache and cache_key in self._cache:
            logger.info("Retrieved XAI explanation output from cache.")
            return self._cache[cache_key]

        logger.info(f"Generating XAI explanations for text: '{text[:50]}...' [Class: {target_class_name}]")

        # 1. Raw Method Outputs
        attention_res = get_attention_signal(self.model, self.tokenizer, text, self.device)
        shap_res = self.shap_explainer.explain(text, target_class_name=target_class_name)
        lime_res = self.lime_explainer.explain(text, target_class_id=target_class_id)
        captum_res = self.captum_explainer.explain(text, target_class_id=target_class_id)

        # Preserve RAW token scores
        att_raw = attention_res.get("token_scores", [])
        shap_raw = shap_res.get("token_scores", [])
        lime_raw = lime_res.get("token_scores", [])
        captum_raw = captum_res.get("token_scores", [])

        attention_res["raw_token_scores"] = att_raw
        shap_res["raw_token_scores"] = shap_raw
        lime_res["raw_token_scores"] = lime_raw
        captum_res["raw_token_scores"] = captum_raw

        # 2. Human-Readable Subword Aggregation Layer
        att_words = aggregate_subwords(att_raw)
        shap_words = aggregate_subwords(shap_raw)
        lime_words = aggregate_subwords(lime_raw)
        captum_words = aggregate_subwords(captum_raw)

        attention_res["word_attributions"] = att_words
        shap_res["word_attributions"] = shap_words
        lime_res["word_attributions"] = lime_words
        captum_res["word_attributions"] = captum_words

        # For user-facing display, default token_scores to clean aggregated words
        attention_res["token_scores"] = att_words if att_words else att_raw
        shap_res["token_scores"] = shap_words if shap_words else shap_raw
        lime_res["token_scores"] = lime_words if lime_words else lime_raw
        captum_res["token_scores"] = captum_words if captum_words else captum_raw

        # 3. Compute Feature Consistency on Normalized Human-Readable Words
        summary_records, consistent_features = self._compute_consistency_analysis(
            att_words, shap_words, lime_words, captum_words
        )

        unified_output = {
            "prediction": target_class_name,
            "confidence": float(confidence),
            "input_text": text,
            "attention_importance": attention_res,
            "shap_importance": shap_res,
            "lime_importance": lime_res,
            "integrated_gradients_importance": captum_res,
            "summary_table": summary_records,
            "consistent_influential_features": consistent_features,
            "consistency_summary_text": (
                f"Consistently influential medical features identified across multiple XAI methods: "
                f"{', '.join([f['word'].upper() for f in consistent_features]) if consistent_features else 'None'}"
            )
        }

        if use_cache:
            self._cache[cache_key] = unified_output

        return unified_output

    def _compute_consistency_analysis(
        self,
        att_list: List[Tuple[str, float]],
        shap_list: List[Tuple[str, float]],
        lime_list: List[Tuple[str, float]],
        captum_list: List[Tuple[str, float]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculates explanation agreement on normalized human-readable words."""
        att_map = {t.lower(): s for t, s in att_list}
        shap_map = {t.lower(): s for t, s in shap_list}
        lime_map = {t.lower(): s for t, s in lime_list}
        captum_map = {t.lower(): s for t, s in captum_list}

        # Gather all unique human-readable words
        all_words = list(dict.fromkeys(
            list(att_map.keys()) + list(shap_map.keys()) + list(lime_map.keys()) + list(captum_map.keys())
        ))

        records = []
        consistent_features = []

        for w in all_words:
            a_s = float(att_map.get(w, 0.0))
            s_s = float(shap_map.get(w, 0.0))
            l_s = float(lime_map.get(w, 0.0))
            c_s = float(captum_map.get(w, 0.0))

            has_att = (a_s > 0.15 or (a_s > 0.0 and len(att_map) <= 5))
            has_shap = (abs(s_s) > 0.005)
            has_lime = (abs(l_s) > 0.005)
            has_ig = (abs(c_s) > 0.02)

            method_hits = sum([1 for h in [has_att, has_shap, has_lime, has_ig] if h])

            is_consistent = (method_hits >= 2)
            label = "Consistently influential feature" if is_consistent else "Method-specific feature"

            rec = {
                "word": w,
                "attention_score": a_s,
                "shap_score": s_s,
                "lime_score": l_s,
                "integrated_gradients_score": c_s,
                "has_attention": has_att,
                "has_shap": has_shap,
                "has_lime": has_lime,
                "has_ig": has_ig,
                "highlight_count": method_hits,
                "agreement_ratio": f"{method_hits}/4",
                "is_consistent": is_consistent,
                "feature_label": label
            }
            records.append(rec)

            if is_consistent:
                consistent_features.append(rec)

        # Sort consistent features by highlight count descending and total attribution sum
        consistent_features.sort(key=lambda x: (x["highlight_count"], abs(x["shap_score"]) + abs(x["lime_score"]) + abs(x["integrated_gradients_score"])), reverse=True)

        return records, consistent_features

