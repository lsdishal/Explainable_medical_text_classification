import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from typing import Dict, Any, List, Tuple

class CaptumExplainer:
    """Captum Integrated Gradients wrapper targeting transformer word embeddings."""

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Target embedding layer (supports BERT/BioBERT and DistilBERT)
        if hasattr(model, "bert"):
            self.embedding_layer = model.bert.embeddings
        elif hasattr(model, "distilbert"):
            self.embedding_layer = model.distilbert.embeddings
        elif hasattr(model, "roberta"):
            self.embedding_layer = model.roberta.embeddings
        else:
            # Fallback to base model embedding layer
            self.embedding_layer = list(model.children())[0].embeddings

        self.lig = LayerIntegratedGradients(self._forward_func, self.embedding_layer)

    def _forward_func(self, inputs_embeds, attention_mask=None):
        """Forward function for Captum receiving embedding tensor."""
        if attention_mask is not None:
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds)
        return outputs.logits

    def explain(self, text: str, target_class: int, n_steps: int = 25) -> List[Tuple[str, float]]:
        """Calculates Integrated Gradients attributions for tokens in input text."""
        self.model.eval()
        self.model.zero_grad()

        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Compute Integrated Gradients
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            target=target_class,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True
        )

        # Sum over embedding dimensions
        attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        # Normalize score
        norm = np.linalg.norm(attributions)
        if norm > 0:
            attributions = attributions / norm

        token_scores = []
        for tok, score in zip(tokens, attributions):
            if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                clean_tok = tok.replace("##", "")
                token_scores.append((clean_tok, float(score)))

        return token_scores
