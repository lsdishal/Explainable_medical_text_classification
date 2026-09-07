import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger

def get_attention_signal(model, tokenizer, text: str, device: torch.device) -> Dict[str, Any]:
    """Extracts Transformer multi-head self-attention weights for input symptom text.
    
    Explicitly labeled as 'Model Attention Signal' with non-causal disclaimer.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "output_attentions": True}
        if "token_type_ids" in inputs:
            kwargs["token_type_ids"] = inputs["token_type_ids"].to(device)

        model.eval()
        old_impl = getattr(model.config, "_attn_implementation", None)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

        with torch.no_grad():
            outputs = model(**kwargs)

        if hasattr(model.config, "_attn_implementation") and old_impl is not None:
            model.config._attn_implementation = old_impl

        if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
            return {"error": "Model architecture does not expose self-attention weights."}

        attentions = outputs.attentions # tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
        num_layers = len(attentions)

        # Average attention weights across all heads in the final layer
        last_layer_attentions = attentions[-1][0].cpu().numpy() # shape: (num_heads, seq_len, seq_len)
        avg_last_layer_att = np.mean(last_layer_attentions, axis=0) # shape: (seq_len, seq_len)

        # Focus on attention from [CLS] token (index 0) to all tokens
        cls_attention_scores = np.array(avg_last_layer_att[0]).flatten() # shape: (seq_len,)

        # Normalize attention scores (0.0 to 1.0)
        min_s, max_s = float(cls_attention_scores.min()), float(cls_attention_scores.max())
        if max_s > min_s:
            norm_scores = (cls_attention_scores - min_s) / (max_s - min_s)
        else:
            norm_scores = cls_attention_scores

        token_scores = []
        for tok, score in zip(tokens, norm_scores.tolist()):
            if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                clean_tok = tok.replace("##", "")
                token_scores.append((clean_tok, float(score)))

        disclaimer = (
            "NOTE: Model Attention Signal represents internal Transformer self-attention weights during inference. "
            "Attention reflects contextual alignment and feature interaction, but IS NOT automatically a causal explanation."
        )

        return {
            "section_title": "Model Attention Signal",
            "tokens": tokens,
            "token_scores": token_scores,
            "num_layers": num_layers,
            "attention_disclaimer": disclaimer
        }
    except Exception as e:
        logger.error(f"Attention extraction failed: {e}")
        return {"error": f"Failed to compute attention weights: {str(e)}"}

