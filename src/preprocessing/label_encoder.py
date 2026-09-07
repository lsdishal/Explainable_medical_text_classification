import json
from pathlib import Path
from typing import Dict, List, Union, Any
from src.utils.logger import logger

class DiseaseLabelEncoder:
    """Encoder class to map medical disease categories to contiguous integer IDs."""

    def __init__(self):
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.classes_: List[str] = []

    def fit(self, labels: List[str]) -> "DiseaseLabelEncoder":
        """Fits encoder on list of string labels."""
        unique_labels = sorted(list(set(str(lbl).strip() for lbl in labels if str(lbl).strip())))
        self.classes_ = unique_labels
        self.label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(unique_labels)}
        logger.info(f"Fitted LabelEncoder on {len(unique_labels)} unique disease classes.")
        return self

    def transform(self, labels: List[str]) -> List[int]:
        """Transforms string labels to integer IDs."""
        if not self.label2id:
            raise ValueError("DiseaseLabelEncoder must be fitted before transform.")
        return [self.label2id[str(lbl).strip()] for lbl in labels]

    def fit_transform(self, labels: List[str]) -> List[int]:
        """Fits on labels and transforms to integer IDs."""
        return self.fit(labels).transform(labels)

    def inverse_transform(self, ids: List[int]) -> List[str]:
        """Transforms integer IDs back to original string disease names."""
        if not self.id2label:
            raise ValueError("DiseaseLabelEncoder must be fitted before inverse_transform.")
        return [self.id2label[i] for i in ids]

    def save(self, filepath: Union[str, Path] = "data/processed/label_mapping.json"):
        """Saves label mapping dictionary to JSON file."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_classes": len(self.classes_),
            "classes": self.classes_,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()}
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved label mapping to '{save_path}'")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "data/processed/label_mapping.json") -> "DiseaseLabelEncoder":
        """Loads DiseaseLabelEncoder instance from saved JSON file."""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Label mapping file not found at '{load_path}'")

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encoder = cls()
        encoder.classes_ = data["classes"]
        encoder.label2id = data["label2id"]
        encoder.id2label = {int(k): v for k, v in data["id2label"].items()}
        return encoder
