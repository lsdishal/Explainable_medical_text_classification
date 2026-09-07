# Research-Level Explainability Evaluation Report

## 1. Executive Summary
This report presents a formal, empirical research evaluation of explanation consistency and stability across **four distinct Explainable AI (XAI) attribution methods** on the fine-tuned **BioBERT** model:
1. **Model Attention Signal** (Multi-head self-attention rollout/CLS signal)
2. **SHAP** (Kernel / Partition Shapley additive feature attributions)
3. **LIME** (Local interpretable model-agnostic explanations with 250 perturbations)
4. **Integrated Gradients** (Captum axiomatic path-integral gradient attributions)

---

## 2. Experimental Setup & Token Normalization

- **Evaluated Sample Size**: 50 held-out test set examples (sampled with seed `42`)
- **Top-K Parameter ($K$)**: `10` tokens per method
- **Token Normalization Layer**:
  - **Special Token Filtering**: Stripped `[CLS]`, `[SEP]`, `[PAD]`, `<s>`, `</s>`, `<pad>`.
  - **Subword Prefix Removal**: Stripped Hugging Face WordPiece `##` and Byte-Pair `Ġ` prefixes.
  - **Punctuation & Clean-up**: Stripped surrounding punctuation (`.`, `,`, `!`, `?`, `"`, `'`).
  - **Lowercase Standardization**: Standardized all tokens to lowercase representation.
  - **Subword Max Aggregation**: Aggregated subwords mapping to identical root words by taking the maximum attribution magnitude.

---

## 3. Pairwise Method Agreement (Jaccard Similarity)

Pairwise agreement is calculated using the **Jaccard Similarity Index**:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

| XAI Method Pair | Mean Jaccard Similarity |
|---|:---:|
| **Attention vs SHAP** | `0.2495` |
| **Attention vs LIME** | `0.2677` |
| **Attention vs Integrated Gradients** | `0.3976` |
| **SHAP vs LIME** | `0.2683` |
| **SHAP vs Integrated Gradients** | `0.3315` |
| **LIME vs Integrated Gradients** | `0.2803` |

> **Highest Agreement Pair**: `Attention vs Integrated Gradients` (Jaccard: `0.3976`)  
> **Most Divergent Pair**: `Attention vs SHAP` (Jaccard: `0.2495`)

---

## 4. Overall Explanation Agreement Score

For each test example $i$, the **Agreement Score** $A_i$ is defined as the mean pairwise Jaccard similarity across all 6 method pairs:
$$A_i = \frac{1}{6} \sum_{(M_1, M_2) \in \text{Pairs}} J(S_{M_1, i}, S_{M_2, i})$$

- **Mean Agreement ($\bar{A}$)**: `0.2992`

- **Median Agreement**: `0.2908`

- **Min Agreement**: `0.1400`

- **Max Agreement**: `0.5881`

- **Standard Deviation ($\sigma$)**: `0.0982`


---

## 5. Explanation Stability Analysis

Method self-stability measures the repeatability of token attributions when running each algorithm twice on the exact same input and model configuration ($J(Run_1, Run_2)$):

| XAI Method | Self-Stability (Jaccard Overlap) | Notes |
|---|:---:|---|
| **Attention** | `1.0000` | Completely deterministic (exact matrix rollout) |
| **SHAP** | `1.0000` | Deterministic partitioning |
| **Integrated Gradients** | `1.0000` | Axiomatic gradient integral (deterministic steps) |
| **LIME** | `0.6533` | Perturbation-based (250 random samples) |

> **Most Stable Method**: `Attention` (Stability: `1.0000`)  
> **Least Stable Method**: `LIME` (Stability: `0.6533`)

---

## 6. Correct vs Incorrect Prediction Comparison

| Group | Sample Count | Mean Agreement Score |
|---|:---:|:---:|
| **Correct Predictions** | `46` | `0.3035` |
| **Incorrect Predictions** | `4` | `0.2495` |

---

## 7. Model Confidence vs Explanation Agreement

- **Pearson Correlation ($r$)**: `-0.0625` ($p$-value: `6.6626e-01`)

- **Spearman Rank Correlation ($\rho$)**: `-0.0251` ($p$-value: `8.6281e-01`)


---

## 8. Representative Case Studies

### A. High-Confidence Correct Predictions

#### Case 1: "I've experienced severe itching, chills, nausea, and a high fever. Besides having a headache, I'm also perspiring a lot. I've been terrible with nausea and muscle ache."
- **True Disease**: `Malaria` | **Predicted**: `Malaria` (Conf: `71.87%`)
- **Attention Top Tokens**: `ache, and, chill, ching, fever, headache, iring, muscle, s, severe`
- **SHAP Top Tokens**: `a, and, besides, chill, fever, having, headache, high, nausea, s`
- **LIME Top Tokens**: `and, been, besides, chills, fever, itching, muscle, nausea, perspiring, ve`
- **Integrated Gradients Top Tokens**: `ache, besides, chill, ching, experienced, fever, headache, nausea, s, severe`
- **Consensus Tokens ($\ge 2$ methods)**: `ache, and, besides, chill, ching, fever, headache, muscle, nausea, s, severe`

#### Case 2: "I've been experiencing severe itching, chills, vomiting, and a high fever. I'm also sweating a lot and have a headache. Nausea and muscle pain have been causing me a lot of discomfort."
- **True Disease**: `Malaria` | **Predicted**: `Malaria` (Conf: `71.86%`)
- **Attention Top Tokens**: `and, chill, ching, fever, it, muscle, pain, s, severe, sweating`
- **SHAP Top Tokens**: `and, been, chill, fever, have, ing, muscle, pain, s, vomit`
- **LIME Top Tokens**: `and, been, chills, fever, have, itching, m, nausea, sweating, vomiting`
- **Integrated Gradients Top Tokens**: `chill, ching, discomfort, fever, headache, ing, muscle, pain, s, vomit`
- **Consensus Tokens ($\ge 2$ methods)**: `and, been, chill, ching, fever, have, ing, muscle, pain, s, sweating, vomit`

#### Case 3: "Strong itchiness, chills, nausea, and a high fever are all part of my current condition. Besides having a headache, I've been perspiring a lot. I've been bothered by nausea as well as muscle ache."
- **True Disease**: `Malaria` | **Predicted**: `Malaria` (Conf: `69.60%`)
- **Attention Top Tokens**: `ache, and, chill, fever, headache, iring, muscle, nausea, s, strong`
- **SHAP Top Tokens**: `a, chill, chin, ess, headache, it, lot, nausea, s, ve`
- **LIME Top Tokens**: `chills, fever, headache, i, itchiness, muscle, nausea, part, perspiring, ve`
- **Integrated Gradients Top Tokens**: `ache, besides, chill, condition, fever, headache, lot, nausea, s, sp`
- **Consensus Tokens ($\ge 2$ methods)**: `ache, chill, fever, headache, lot, muscle, nausea, s, ve`

### B. Low-Confidence Correct Predictions

#### Case 1: "My mouth and throat are dry. Additionally, I've been growing increasingly hungry. However, I do get periodic bouts of exhaustion."
- **True Disease**: `diabetes` | **Predicted**: `diabetes` (Conf: `14.07%`)
- **Attention Top Tokens**: `additionally, are, bouts, dry, exhaustion, get, however, hungry, of, periodic`
- **SHAP Top Tokens**: `are, do, dry, growing, however, hungry, i, increasingly, throat, ve`
- **LIME Top Tokens**: `are, been, bouts, do, dry, however, i, increasingly, mouth, of`
- **Integrated Gradients Top Tokens**: `additionally, been, do, dry, growing, hungry, mouth, my, throat, ve`
- **Consensus Tokens ($\ge 2$ methods)**: `additionally, are, been, bouts, do, dry, growing, however, hungry, i, increasingly, mouth, of, throat, ve`

#### Case 2: "My mouth and throat are dry. Additionally, I've been growing increasingly hungry. But on occasion, I do feel rather worn out."
- **True Disease**: `diabetes` | **Predicted**: `diabetes` (Conf: `20.81%`)
- **Attention Top Tokens**: `are, but, dry, feel, growing, hungry, increasingly, out, throat, worn`
- **SHAP Top Tokens**: `additionally, are, but, do, dry, feel, i, on, rather, throat`
- **LIME Top Tokens**: `and, are, do, dry, i, increasingly, mouth, my, on, rather`
- **Integrated Gradients Top Tokens**: `additionally, been, do, dry, growing, increasingly, mouth, my, occasion, throat`
- **Consensus Tokens ($\ge 2$ methods)**: `additionally, are, but, do, dry, feel, growing, i, increasingly, mouth, my, on, rather, throat`

#### Case 3: "I have difficulty sleeping due to abdominal pain or discomfort. I have a loss of appetite and feel fatigued after going to the bathroom. I feel very tited all the time"
- **True Disease**: `peptic ulcer disease` | **Predicted**: `peptic ulcer disease` (Conf: `23.52%`)
- **Attention Top Tokens**: `abdominal, all, and, appetite, bathroom, fatigue, pain, sleeping, t, time`
- **SHAP Top Tokens**: `a, abdominal, appetite, discomfort, due, have, i, loss, or, to`
- **LIME Top Tokens**: `abdominal, appetite, discomfort, due, fatigued, feel, i, loss, pain, sleeping`
- **Integrated Gradients Top Tokens**: `abdominal, appetite, difficulty, discomfort, due, i, loss, or, pain, time`
- **Consensus Tokens ($\ge 2$ methods)**: `abdominal, appetite, discomfort, due, i, loss, or, pain, sleeping, time`

### C. Incorrect Predictions

#### Case 1: "A sour, acidic taste is all over my mouth. Belching and burping are common in me. I have pressure or fullness in my upper abdomen for a few hours."
- **True Disease**: `peptic ulcer disease` | **Predicted**: `gastroesophageal reflux disease` (Conf: `29.61%`)
- **Attention Top Tokens**: `a, abdomen, all, are, ching, common, mouth, sour, taste, ur`
- **SHAP Top Tokens**: `abdomen, and, b, bel, common, few, hours, ping, upper, ur`
- **LIME Top Tokens**: `abdomen, acidic, belching, burping, common, me, mouth, or, taste, upper`
- **Integrated Gradients Top Tokens**: `a, acidic, b, ching, common, mouth, my, ping, sour, taste`
- **Consensus Tokens ($\ge 2$ methods)**: `a, abdomen, acidic, b, ching, common, mouth, ping, sour, taste, upper, ur`

#### Case 2: "The muscle pain I have been experiencing is severe and feels like a constant ache. There are red spots all over my body and the itching is causing me a lot of discomfort."
- **True Disease**: `Dengue` | **Predicted**: `Chicken pox` (Conf: `24.82%`)
- **Attention Top Tokens**: `all, are, body, ching, it, muscle, pain, red, spots, there`
- **SHAP Top Tokens**: `ache, and, been, experiencing, have, i, is, red, severe, spots`
- **LIME Top Tokens**: `body, itching, me, muscle, my, pain, red, spots, the, there`
- **Integrated Gradients Top Tokens**: `all, body, ching, it, muscle, my, pain, red, severe, spots`
- **Consensus Tokens ($\ge 2$ methods)**: `all, body, ching, it, muscle, my, pain, red, severe, spots, there`

#### Case 3: "Particularly in the crevices of my skin, I have skin rashes and irritations. My skin bruises and cuts take a while to heal as well."
- **True Disease**: `diabetes` | **Predicted**: `Fungal infection` (Conf: `13.80%`)
- **Attention Top Tokens**: `ash, bruises, c, cuts, have, in, s, skin, the, vice`
- **SHAP Top Tokens**: `and, cuts, icular, in, my, of, part, skin, take, the`
- **LIME Top Tokens**: `and, as, bruises, crevices, cuts, irritations, my, particularly, skin, take`
- **Integrated Gradients Top Tokens**: `bruises, es, heal, icular, ly, my, of, re, s, skin`
- **Consensus Tokens ($\ge 2$ methods)**: `and, bruises, cuts, icular, in, my, of, s, skin, take, the`

---

## 9. Important Scientific & Medical Limitations

1. **Non-Causality of Attention**: Attention maps reflect contextual representation alignment inside BioBERT's transformer layers. They do not constitute a causal explanation of model prediction.
2. **Attribution Convergence vs Medical Causality**: High agreement across XAI methods indicates that multiple algorithmic surrogate perspectives focus on the same symptom tokens. It does **NOT** prove medical or clinical causality.
3. **Surrogate Perturbation Noise**: Perturbation-based explainers (such as LIME) exhibit slight non-deterministic variance across runs due to stochastic sampling around local decision boundaries.
