import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.config import get_config

def download_symptom2disease(config_path: str = "config.yaml") -> pd.DataFrame:
    """Downloads or loads the Symptom2Disease dataset."""
    cfg = get_config(config_path)
    raw_path = Path(cfg.dataset.raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info(f"Loading existing raw dataset from {raw_path}")
        df = pd.read_csv(raw_path)
        # Normalize column names if needed
        if "label" not in df.columns or "text" not in df.columns:
            if "label" in df.columns and "text" not in df.columns:
                text_col = [c for c in df.columns if c != "label"][0]
                df = df.rename(columns={text_col: "text"})
            elif "label" not in df.columns:
                # Symptom2Disease often has unnamed index, label, text
                if len(df.columns) >= 3:
                    df.columns = ["idx", "label", "text"]
                elif len(df.columns) == 2:
                    df.columns = ["label", "text"]
        return df

    logger.info("Attempting to download Symptom2Disease dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(cfg.dataset.hf_name)
        df = pd.DataFrame(dataset["train"])
        if "text" not in df.columns and "symptom" in df.columns:
            df = df.rename(columns={"symptom": "text"})
        df.to_csv(raw_path, index=False)
        logger.info(f"Successfully downloaded and saved dataset to {raw_path} ({len(df)} samples)")
        return df
    except Exception as e:
        logger.warning(f"Failed to load dataset via Hugging Face: {e}. Trying direct HTTPS fallback...")
        fallback_url = "https://raw.githubusercontent.com/Anish-Karthik/Symptom2Disease/main/Symptom2Disease.csv"
        try:
            df = pd.read_csv(fallback_url)
            if "label" not in df.columns or "text" not in df.columns:
                if len(df.columns) >= 3:
                    df = df.iloc[:, 1:]
                    df.columns = ["label", "text"]
            df.to_csv(raw_path, index=False)
            logger.info(f"Successfully downloaded fallback CSV to {raw_path} ({len(df)} samples)")
            return df
        except Exception as e_fallback:
            logger.error(f"Fallback download also failed: {e_fallback}")
            raise RuntimeError("Could not download dataset. Please check internet connection or place symptom2disease.csv in data/raw/")

def prepare_splits(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Loads raw dataset, creates stratified train/val/test splits, and saves them in processed_dir."""
    cfg = get_config(config_path)
    df = download_symptom2disease(config_path)
    
    # Ensure text and label columns exist
    df = df[["label", "text"]].dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Create label mapping dictionaries
    unique_labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    df["label_id"] = df["label"].map(label2id)

    # Train / Val / Test split
    test_size = cfg.dataset.test_split
    val_size = cfg.dataset.val_split / (cfg.dataset.train_split + cfg.dataset.val_split)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=cfg.project.seed, stratify=df["label_id"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=cfg.project.seed, stratify=train_val_df["label_id"]
    )

    processed_dir = Path(cfg.dataset.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    logger.info(f"Dataset splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} across {len(unique_labels)} disease classes.")
    return train_df, val_df, test_df, label2id, id2label

if __name__ == "__main__":
    prepare_splits()
