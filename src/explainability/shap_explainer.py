import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger

class SHAPExplainer:
    """SHAP Explainer wrapper for Transformer text classification models."""

    def __init__(self, model, tokenizer, label_list: List[str], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            from transformers import pipeline
            # MPS and CPU both use device=-1 for HF pipeline (CPU inference)
            # SHAP requires numpy-compatible outputs which MPS tensors don't support directly
            device_id = 0 if self.device.type == "cuda" else -1
            pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,       # returns scores for all classes (replaces return_all_scores)
                device=device_id  # CPU (-1) for MPS/CPU hosts
            )
            self._explainer = shap.Explainer(pipe)
        return self._explainer

    def explain(self, text: str, target_class_name: str = None, max_evals: int = 100) -> Dict[str, Any]:
        """Computes SHAP values for words in input text corresponding to target disease class."""
        try:
            explainer = self._get_explainer()
            shap_values = explainer([text], max_evals=max_evals)

            # Determine index of target class
            if target_class_name and target_class_name in self.label_list:
                class_idx = self.label_list.index(target_class_name)
            else:
                # Default to index of highest output score
                vals = shap_values.values[0]
                class_idx = int(np.argmax(vals.sum(axis=0)))

            tokens = shap_values.data[0]
            values = shap_values.values[0][:, class_idx]

            token_scores = []
            for tok, val in zip(tokens, values):
                clean_tok = tok.strip()
                if clean_tok and clean_tok not in ["[CLS]", "[SEP]", "[PAD]"]:
                    token_scores.append((clean_tok, float(val)))

            return {
                "method": "SHAP",
                "token_scores": token_scores,
                "target_class": target_class_name or self.label_list[class_idx]
            }
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
            return {"error": f"SHAP calculation unavailable: {str(e)}"}
