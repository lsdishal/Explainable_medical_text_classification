import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger

class CaptumIGExplainer:
    """Captum IntegratedGradients wrapper targeting Transformer word embeddings."""

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def explain(self, text: str, target_class_id: int = 0, n_steps: int = 25) -> Dict[str, Any]:
        """Calculates Integrated Gradients token attributions."""
        try:
            from captum.attr import IntegratedGradients

            self.model.eval()
            inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            # Retrieve embedding layer dynamically
            if hasattr(self.model, "get_input_embeddings"):
                embed_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
                embed_layer = self.model.bert.embeddings.word_embeddings
            else:
                embed_layer = list(self.model.children())[0].embeddings.word_embeddings

            input_embeds = embed_layer(input_ids)

            def forward_func(inputs_embeds, attention_mask=None):
                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                return outputs.logits

            ig = IntegratedGradients(forward_func)

            attributions, delta = ig.attribute(
                inputs=input_embeds,
                target=target_class_id,
                additional_forward_args=(attention_mask,),
                n_steps=n_steps,
                return_convergence_delta=True
            )

            # Sum over embedding dimensions
            attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

            # Normalize scores
            norm = np.linalg.norm(attributions)
            if norm > 0:
                attributions = attributions / norm

            token_scores = []
            for tok, score in zip(tokens, attributions):
                if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                    clean_tok = tok.replace("##", "")
                    token_scores.append((clean_tok, float(score)))

            return {
                "method": "Captum Integrated Gradients",
                "token_scores": token_scores,
                "convergence_delta": float(delta.squeeze(0).cpu().detach().numpy()) if hasattr(delta, "squeeze") else 0.0
            }
        except Exception as e:
            logger.warning(f"Integrated Gradients calculation failed: {e}")
            return {"error": f"Integrated Gradients calculation unavailable: {str(e)}"}

