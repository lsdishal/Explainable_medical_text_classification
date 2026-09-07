import os
import ssl
import json
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from src.utils.logger import logger
from src.utils.config import get_config

def fetch_symptom2disease_hf_api() -> pd.DataFrame:
    """Fetches Symptom2Disease dataset via Hugging Face REST API with pagination."""
    rows = []
    offset = 0
    limit = 100
    ctx = ssl._create_unverified_context()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    logger.info("Fetching Symptom2Disease via Hugging Face REST API...")
    headers = {"User-Agent": "Mozilla/5.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    while True:
        url = f"https://datasets-server.huggingface.co/rows?dataset=NeuronZero%2FSymptom2Disease&config=default&split=train&offset={offset}&limit={limit}"
        req = urllib.request.Request(url, headers=headers)
        try:
            res = urllib.request.urlopen(req, context=ctx)
            data = json.loads(res.read().decode("utf-8"))
            batch = [item["row"] for item in data["rows"]]
            if not batch:
                break
            rows.extend(batch)
            offset += limit
            if len(batch) < limit:
                break
        except Exception as e:
            logger.warning(f"HF API Pagination ended at offset {offset}: {e}")
            break

    if not rows:
        raise RuntimeError("Failed to fetch rows from Hugging Face REST API.")

    df = pd.DataFrame(rows)
    logger.info(f"Successfully fetched {len(df)} rows from Hugging Face REST API across {df['label'].nunique()} classes.")
    return df

def load_raw_dataset(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads raw Symptom2Disease dataset, validates columns, removes index columns, and audits data quality."""
    cfg = get_config(config_path)
    raw_path = Path(cfg.dataset.raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    df = None
    if raw_path.exists():
        logger.info(f"Loading raw dataset from local file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        logger.info("Local dataset not found. Fetching Symptom2Disease...")
        try:
            from datasets import load_dataset
            kwargs = {"token": hf_token} if hf_token else {}
            ds = load_dataset(cfg.dataset.hf_name, **kwargs)
            df = pd.DataFrame(ds["train"])
            logger.info(f"Successfully loaded dataset via Hugging Face datasets package '{cfg.dataset.hf_name}'")
        except Exception as e:
            logger.warning(f"Hugging Face datasets package load failed ({e}). Using REST API fallback...")
            df = fetch_symptom2disease_hf_api()

        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw dataset to '{raw_path}'")

    audit_meta = {}
    audit_meta["original_row_count"] = len(df)
    audit_meta["original_columns"] = list(df.columns)

    # Remove unnecessary index columns (e.g. 'Unnamed: 0', 'idx', etc.)
    index_cols = [c for c in df.columns if c.startswith("Unnamed") or c.lower() in ["idx", "index", "id"]]
    if index_cols:
        logger.info(f"Removing index columns: {index_cols}")
        df = df.drop(columns=index_cols)

    # Normalize column names to lower case if needed
    col_mapping = {}
    for col in df.columns:
        if col.lower() in ["label", "disease", "category", "target"]:
            col_mapping[col] = "label"
        elif col.lower() in ["text", "symptom", "symptoms", "description"]:
            col_mapping[col] = "text"

    if col_mapping:
        df = df.rename(columns=col_mapping)

    # Validate required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset missing required columns ('text', 'label'). Found columns: {list(df.columns)}")

    # Audit missing text and label values
    missing_text = df["text"].isna().sum()
    missing_label = df["label"].isna().sum()
    total_missing = int(df[["text", "label"]].isna().any(axis=1).sum())

    audit_meta["missing_text_count"] = int(missing_text)
    audit_meta["missing_label_count"] = int(missing_label)
    audit_meta["total_missing_rows"] = total_missing

    # Audit duplicates
    duplicate_rows = int(df.duplicated(subset=["label", "text"]).sum())
    audit_meta["duplicate_rows_count"] = duplicate_rows

    logger.info(
        f"Dataset loaded successfully. Total Rows: {len(df)} | Missing Rows: {total_missing} | Duplicate Rows: {duplicate_rows}"
    )

    return df, audit_meta

if __name__ == "__main__":
    df, audit = load_raw_dataset()
    print("Dataset audit:", audit)
    print(df.head())
