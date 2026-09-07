import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger

class LIMEExplainer:
    """LIME Explainer wrapper for local linear surrogate explanations."""

    def __init__(self, model, tokenizer, label_list: List[str], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device
        self._lime = None

    def _get_lime(self):
        if self._lime is None:
            from lime.lime_text import LimeTextExplainer
            self._lime = LimeTextExplainer(class_names=self.label_list)
        return self._lime

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Probability predictor function wrapper for LIME perturbations."""
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in inputs:
            kwargs["token_type_ids"] = inputs["token_type_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(**kwargs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs

    def explain(self, text: str, target_class_id: int = None, num_samples: int = 250, num_features: int = 10) -> Dict[str, Any]:
        """Generates LIME explanation feature weights for input symptom text."""
        try:
            lime_engine = self._get_lime()
            probs = self._predict_proba([text])[0]

            if target_class_id is None or target_class_id >= len(probs):
                target_class_id = int(np.argmax(probs))

            exp = lime_engine.explain_instance(
                text_instance=text,
                classifier_fn=self._predict_proba,
                labels=[target_class_id],
                num_samples=num_samples,
                num_features=num_features
            )

            explanation_list = exp.as_list(label=target_class_id)
            token_scores = [(word, float(score)) for word, score in explanation_list]

            target_disease = self.label_list[target_class_id] if target_class_id < len(self.label_list) else str(target_class_id)

            return {
                "method": "LIME",
                "token_scores": token_scores,
                "target_class": target_disease
            }
        except Exception as e:
            logger.warning(f"LIME explanation generation failed: {e}")
            return {"error": f"LIME calculation unavailable: {str(e)}"}
