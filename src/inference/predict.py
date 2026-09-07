import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.logger import logger
from src.utils.config import get_config
from src.preprocessing.cleaner import clean_medical_text
from src.training.model_loader import get_compute_device
from src.inference.confidence import ConfidenceEstimator

class MedicalSymptomClassifier:
    """Production-quality inference engine for medical text classification."""

    def __init__(self, model_dir: str = "models/best_model", config_path: str = "config.yaml"):
        self.cfg = get_config(config_path)
        self.config_path = config_path
        self.model_dir = Path(model_dir)
        self.device = get_compute_device(getattr(self.cfg.training, "device", "auto"))

        # Fallback resolution if local trained checkpoint directory does not exist
        if not self.model_dir.exists():
            fallback_dir = Path("models/biobert")
            if fallback_dir.exists():
                self.model_dir = fallback_dir
            else:
                logger.info(f"Local checkpoint '{self.model_dir}' not found. Using pretrained HF model fallback '{self.cfg.models.primary.hf_model_id}'...")
                self.model_dir = Path(self.cfg.models.primary.hf_model_id)

        # Check for tokenizer directory
        tokenizer_dir = Path(getattr(self.cfg.models, "tokenizer_dir", "models/tokenizer"))
        tok_path = tokenizer_dir if tokenizer_dir.exists() else self.model_dir

        logger.info(f"Initializing inference engine from model '{self.model_dir}' and tokenizer '{tok_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        # Load label mapping
        label_file = Path("data/processed/label_mapping.json")
        if self.model_dir.exists() and (self.model_dir / "id2label.json").exists():
            label_file = self.model_dir / "id2label.json"

        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                raw_id2label = json.load(f)
                if "id2label" in raw_id2label:
                    self.id2label = {int(k): v for k, v in raw_id2label["id2label"].items()}
                    self.label2id = raw_id2label["label2id"]
                else:
                    self.id2label = {int(k): v for k, v in raw_id2label.items()}
                    self.label2id = {v: k for k, v in self.id2label.items()}
            num_labels = len(self.id2label)
        else:
            self.id2label = {0: "Psoriasis", 1: "Varicose Veins"}
            self.label2id = {"Psoriasis": 0, "Varicose Veins": 1}
            num_labels = 2

        self.label_list = [self.id2label[i] for i in sorted(self.id2label.keys())]

        # Load Sequence Classification Model
        if self.model_dir.exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )

        self.model.to(self.device)
        self.model.eval()

        self.model_name = str(self.model_dir.name)

        # Initialize Confidence Estimator
        self.confidence_estimator = ConfidenceEstimator(config_path=config_path)

    def predict(self, symptom_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Runs production inference on arbitrary natural language medical symptom text.
        
        Guarantees 100% JSON-serializable output structure and never crashes on unusual user input.
        """
        try:
            # 1. Clean & Normalize Text (handles HTML, strange unicode, unusual spaces)
            cleaned_text = clean_medical_text(symptom_text)
            
            # Handle Empty Input
            if not cleaned_text:
                return {
                    "error": "Empty input text. Please provide a valid natural-language symptom description.",
                    "status": "invalid_input",
                    "predicted_disease": None,
                    "confidence": 0.0,
                    "low_confidence_warning": True
                }

            # 2. Tokenize with Truncation (handles extremely long inputs gracefully)
            max_len = int(getattr(self.cfg.training, "max_length", 128))
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=False
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Convert input IDs to readable token string list for output JSON
            tokenized_input = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            # 3. Model Forward Pass & Softmax Probabilities
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in inputs:
                kwargs["token_type_ids"] = inputs["token_type_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(**kwargs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # 4. Sort Top-K Predictions
            top_k_val = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k_val]

            top_predictions = []
            all_probabilities = {}

            for i in range(len(probs)):
                disease_name = self.id2label.get(i, f"Class_{i}")
                prob_val = float(probs[i])
                all_probabilities[disease_name] = prob_val

            for idx in top_indices:
                disease_name = self.id2label.get(int(idx), f"Class_{idx}")
                confidence = float(probs[idx])
                top_predictions.append({
                    "class_id": int(idx),
                    "disease": disease_name,
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence * 100:.2f}%"
                })

            primary_pred = top_predictions[0]
            primary_disease = primary_pred["disease"]
            primary_confidence = primary_pred["confidence"]

            # 5. Evaluate Confidence Category & Warning Flags
            conf_info = self.confidence_estimator.evaluate_confidence(primary_confidence)

            # Assemble JSON-serializable Output Structure
            result = {
                "status": "success",
                "input_text": cleaned_text,
                "predicted_disease": primary_disease,
                "confidence": primary_confidence,
                "confidence_percentage": f"{primary_confidence * 100:.2f}%",
                "confidence_category": conf_info["confidence_category"],
                "low_confidence_warning": conf_info["low_confidence_warning"],
                "warning_message": conf_info["warning_message"],
                "top_3_predictions": top_predictions,
                "all_probabilities": all_probabilities,
                "tokenized_input": tokenized_input,
                "model_name": self.model_name,
                "medical_disclaimer": conf_info["medical_disclaimer"]
            }

            # Verify JSON serializability before returning
            json.dumps(result)
            return result

        except Exception as e:
            logger.error(f"Inference error on input text: {e}")
            return {
                "error": f"An error occurred during prediction: {str(e)}",
                "status": "execution_error",
                "predicted_disease": None,
                "confidence": 0.0,
                "low_confidence_warning": True
            }

if __name__ == "__main__":
    classifier = MedicalSymptomClassifier()
    sample_res = classifier.predict("I have red scaly rashes on my arms with dry skin patches.")
    print("Inference Result:\n", json.dumps(sample_res, indent=2))
