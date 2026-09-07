# Dataset Directory

This directory stores datasets for the Explainable Medical Text Classification project.

- `raw/`: Raw Symptom2Disease dataset files downloaded from Hugging Face (`NeuronZero/Symptom2Disease`) or Kaggle.
- `processed/`: Preprocessed and split dataset files (`train.csv`, `val.csv`, `test.csv`).

## Symptom2Disease Dataset Overview
- **Task**: Medical text classification of natural language patient symptom descriptions into disease categories.
- **Categories**: 24 distinct medical conditions (e.g., Psoriasis, Varicose Veins, Typhoid, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, etc.).
- **Features**: `text` (symptom text description), `label` (target disease category).
