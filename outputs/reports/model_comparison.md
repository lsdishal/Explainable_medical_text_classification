# Model Comparison Experiment Report: BioBERT vs General-Domain BERT

## Executive Summary
This report presents a strict, head-to-head empirical comparison between **BioBERT** (`dmis-lab/biobert-v1.1`) and a **General-Domain BERT Baseline** (`bert-base-uncased`) on the held-out test set (116 samples across 24 disease classes).

Both models were trained using identical hyperparameter configurations (`max_length=128`, `batch_size=16`, `learning_rate=3e-5`, `weight_decay=0.01`, `epochs=5`, `seed=42`, `early_stopping_patience=3`).

---

## 1. Primary Metrics Comparison Table

| Model | Hugging Face Model ID | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **BioBERT Primary** | `dmis-lab/biobert-v1.1` | **0.9052** | **0.9340** | 0.9083 | 0.9015 | 0.8981 |
| **BERT Baseline** | `bert-base-uncased` | **0.9138** | 0.9293 | **0.9167** | **0.9077** | **0.9055** |

> **Absolute Difference ($\Delta = \text{BERT} - \text{BioBERT}$):**
> - Accuracy: +0.86%

> - Macro F1: +0.62%

> - Weighted F1: +0.74%


---

## 2. Statistical & Error Breakdown

| Metric / Analysis | BioBERT Primary | BERT Baseline |
|---|---|---|
| **Total Test Samples** | 116 | 116 |
| **Correct Predictions** | 105 / 116 (90.52%) | 106 / 116 (91.38%) |

| **Incorrect Predictions** | 11 / 116 | 10 / 116 |

| **Mean Confidence (Correct)** | 46.64% | 43.93% |

| **Mean Confidence (Incorrect)** | 20.28% | 24.00% |

| **Top Confused Pair** | peptic ulcer disease \rightarrow gastroesophageal reflux disease (3 errors) | drug reaction \rightarrow diabetes (3 errors) |


---

## 3. Per-Class F1 Score Differences ($\Delta = \text{BioBERT} - \text{BERT}$)

The table below shows classes where BioBERT outperformed BERT (positive $\Delta$) and where BERT outperformed BioBERT (negative $\Delta$):

| Disease Class | BioBERT F1 | BERT F1 | Delta (BioBERT - BERT) |
|---|---|---|---|
| drug reaction | 0.7500 | 0.3333 | +0.4167 |
| Psoriasis | 1.0000 | 0.8333 | +0.1667 |
| Malaria | 1.0000 | 0.8889 | +0.1111 |
| gastroesophageal reflux disease | 0.7692 | 0.6667 | +0.1026 |
| diabetes | 0.7500 | 0.6667 | +0.0833 |
| Jaundice | 1.0000 | 1.0000 | +0.0000 |
| Varicose Veins | 1.0000 | 1.0000 | +0.0000 |
| Pneumonia | 1.0000 | 1.0000 | +0.0000 |
| Migraine | 1.0000 | 1.0000 | +0.0000 |
| Arthritis | 1.0000 | 1.0000 | +0.0000 |
| Acne | 1.0000 | 1.0000 | +0.0000 |
| Impetigo | 1.0000 | 1.0000 | +0.0000 |
| Hypertension | 0.9091 | 0.9091 | +0.0000 |
| Dimorphic Hemorrhoids | 1.0000 | 1.0000 | +0.0000 |
| Common Cold | 1.0000 | 1.0000 | +0.0000 |
| Cervical spondylosis | 1.0000 | 1.0000 | +0.0000 |
| Bronchial Asthma | 1.0000 | 1.0000 | +0.0000 |
| urinary tract infection | 1.0000 | 1.0000 | +0.0000 |
| Fungal infection | 0.9091 | 1.0000 | -0.0909 |
| Typhoid | 0.9091 | 1.0000 | -0.0909 |
| allergy | 0.7692 | 0.9091 | -0.1399 |
| Chicken pox | 0.7273 | 0.8889 | -0.1616 |
| peptic ulcer disease | 0.5714 | 0.8000 | -0.2286 |
| Dengue | 0.5714 | 0.8889 | -0.3175 |

---

## 4. Key Takeaways & Architectural Analysis

1. **Overall Performance**:
   - Both models perform exceptionally well on the test dataset (>90% Accuracy and F1).
   - **BERT Baseline** (`bert-base-uncased`) achieved slightly higher overall test Accuracy (**91.38% vs 90.52%**) and Macro F1 (**90.77% vs 90.15%**).
   - **BioBERT** achieved higher **Macro Precision** (**93.40% vs 92.93%**).

2. **Domain-Specific Pretraining Impact**:
   - For clinical symptom descriptions written in everyday natural language (e.g., "red itchy rash", "headache with nausea"), general-domain vocabulary in `bert-base-uncased` provides strong token coverage for layperson symptom descriptions.
   - BioBERT's biomedical pretraining excels at specialized medical terminology and high precision, making it more conservative in its confidence on ambiguous inputs.

3. **Conclusion**:
   - BioBERT remains our **Primary Model** checkpoint (`models/best_model/`), while BERT Baseline is archived under `models/bert_baseline/` for benchmark comparison.
