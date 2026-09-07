# MedXAI — Model Quality & Performance Investigation Report

**Model Architecture**: `BertForSequenceClassification` (`dmis-lab/biobert-v1.1`)  
**Dataset**: `Symptom2Disease` (24 classes)  
**Evaluation Target**: Held-Out Test Set (116 samples)  
**Date**: September 7, 2026  

---

## 1. Overall Model Performance

Primary metrics calculated on the 116 held-out test set samples:

| Metric | Value |
| :--- | :---: |
| **Accuracy** | **90.52%** |
| **Macro Precision** | **93.40%** |
| **Macro Recall** | **90.83%** |
| **Macro F1** | **90.15%** |
| **Weighted F1** | **89.81%** |

The model achieves strong overall classification performance on standard test set sentences. However, investigation below reveals significant domain shift sensitivity when handling shorthand or natural conversational user prompts.

---

## 2. Per-Class Performance Breakdown

| Disease | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Acne** | 100.0% | 100.0% | **100.0%** | 4 |
| **Impetigo** | 100.0% | 100.0% | **100.0%** | 5 |
| **Varicose Veins** | 100.0% | 100.0% | **100.0%** | 5 |
| **Psoriasis** | 100.0% | 100.0% | **100.0%** | 5 |
| **Pneumonia** | 100.0% | 100.0% | **100.0%** | 5 |
| **Migraine** | 100.0% | 100.0% | **100.0%** | 5 |
| **Arthritis** | 100.0% | 100.0% | **100.0%** | 5 |
| **Jaundice** | 100.0% | 100.0% | **100.0%** | 4 |
| **Malaria** | 100.0% | 100.0% | **100.0%** | 4 |
| **Dimorphic Hemorrhoids** | 100.0% | 100.0% | **100.0%** | 4 |
| **Common Cold** | 100.0% | 100.0% | **100.0%** | 5 |
| **Cervical spondylosis** | 100.0% | 100.0% | **100.0%** | 5 |
| **Bronchial Asthma** | 100.0% | 100.0% | **100.0%** | 5 |
| **urinary tract infection** | 100.0% | 100.0% | **100.0%** | 5 |
| **Hypertension** | 83.3% | 100.0% | **90.9%** | 5 |
| **Fungal infection** | 83.3% | 100.0% | **90.9%** | 5 |
| **Typhoid** | 83.3% | 100.0% | **90.9%** | 5 |
| **allergy** | 62.5% | 100.0% | **76.9%** | 5 |
| **gastroesophageal reflux disease** | 62.5% | 100.0% | **76.9%** | 5 |
| **diabetes** | 100.0% | 60.0% | **75.0%** | 5 |
| **drug reaction** | 100.0% | 60.0% | **75.0%** | 5 |
| **Chicken pox** | 66.7% | 80.0% | **72.7%** | 5 |
| **Dengue** | 100.0% | 40.0% | **57.1%** | 5 |
| **peptic ulcer disease** | 100.0% | 40.0% | **57.1%** | 5 |

### Key Class Performance Insights:
- **Best Performing Classes (F1 = 100%)**: Acne, Arthritis, Bronchial Asthma, Cervical spondylosis, Common Cold, Dimorphic Hemorrhoids, Impetigo, Jaundice, Malaria, Migraine, Pneumonia, Psoriasis, Varicose Veins, Urinary Tract Infection (14 out of 24 classes achieve perfect performance).
- **Lowest Performing Classes (F1 < 75%)**:
  1. `Dengue`: F1 = **57.1%** (Recall = 40.0%, 3 samples misclassified as Chicken pox or Typhoid due to overlapping fever/rash symptoms).
  2. `peptic ulcer disease`: F1 = **57.1%** (Recall = 40.0%, 3 samples misclassified as GERD due to identical upper abdominal pain descriptions).
  3. `Chicken pox`: F1 = **72.7%** (Precision = 66.7%).
  4. `diabetes`: F1 = **75.0%** (Recall = 60.0%).
  5. `drug reaction`: F1 = **75.0%** (Recall = 60.0%).

---

## 3. Confusion & Error Analysis

Top confused disease pairs on the test set:

| Ground Truth Disease | Predicted Disease | Misclassification Count | Clinical Explanation |
| :--- | :--- | :---: | :--- |
| `peptic ulcer disease` | `gastroesophageal reflux disease` | **3** | Overlapping upper GI symptoms (heartburn, stomach pain, burning sensation) |
| `Dengue` | `Chicken pox` | **2** | Overlapping acute systemic symptoms (high fever, severe body pain, skin rash) |
| `drug reaction` | `allergy` | **2** | Overlapping skin allergy symptoms (hives, itching, redness, inflammation) |
| `Dengue` | `Typhoid` | **1** | Shared acute fever and malaise symptoms |
| `diabetes` | `allergy` | **1** | Non-specific symptom overlap |
| `Chicken pox` | `Hypertension` | **1** | Non-specific symptom overlap |
| `diabetes` | `Fungal infection` | **1** | Skin irritation overlap |

---

## 4. Data Split & Leakage Verification

- **Train Set**: 921 samples (80.0%)
- **Validation Set**: 116 samples (10.0%)
- **Test Set**: 116 samples (10.0%)
- **Stratification**: Confirmed applied. Each validation and test class contains exactly 4 or 5 samples across all 24 classes.
- **Exact Overlap**: **0 exact duplicate sentences** across train, validation, and test splits.
- **Near-Duplicate Check**: **0 near-duplicate sentences** (Jaccard token similarity > 0.85) across splits.
- **Data Leakage Conclusion**: **NO DATA LEAKAGE DETECTED.** Test metrics reflect genuine generalization on holdout test set sentences.

---

## 5. Manual Natural-Language Test Input Investigation

Running the 5 manually entered test prompts:

| # | Input Text | BioBERT Prediction | Conf % | Top-3 Predictions | Influential XAI Tokens |
|---|---|---|:---:|---|---|
| **Ex 1** | *"I have a severe headache with nausea and sensitivity to light."* | `Dengue` | **13.12%** | Dengue (13.1%), Malaria (10.4%), Hypertension (10.1%) | `headache`, `severe`, `light` |
| **Ex 2** | *"I have a red itchy rash with dry scaly patches."* | `Fungal infection` | **24.71%** | Fungal infection (24.7%), Psoriasis (14.3%), Chicken pox (10.4%) | `patches`, `dry`, `red`, `scaly`, `rash`, `itchy` |
| **Ex 3** | *"I have a cough, sore throat and runny nose."* | `allergy` | **37.24%** | allergy (37.2%), GERD (12.0%), Peptic Ulcer (6.1%) | `throat`, `runny`, `nose`, `cough`, `sore` |
| **Ex 4** | *"no fever but persistent cough"* | `urinary tract infection` | **10.55%** | UTI (10.6%), Typhoid (7.7%), Drug Reaction (5.9%) | `persistent`, `cough` |
| **Ex 5** | *"fever"* | `urinary tract infection` | **8.65%** | UTI (8.6%), Typhoid (6.9%), Hypertension (6.5%) | `fever` |

---

## 6. Training Data Style & Out-of-Distribution (OOD) Analysis

Inspection of actual `processed_train.csv` text samples for key classes:

### `Migraine` (Training Samples):
- *"Along with excessive appetite, a stiff neck, depression, impatience, and visual disturbance, I've also been having indigestion, headaches, blurred vision, and acid reflux. These symptoms have been bothering me and interfering with my usual activities."*
- *"Along with impaired seeing, excessive eating, a sore neck, melancholy, impatience, and constant headaches, I've also been fighting heartburn, acid reflux, and indigestion."*
- *"Stomach acid, indigestion, frequent headaches, hazy and distorted vision, increased appetite, stiff neck, depression, irritation, and visual distortion have all been bothering me."*

### `Dengue` (Training Samples):
- *"My shivers and shivering have been accompanied with a sensation of coldness and a very high fever. Rashes on my arms and red patches on my neck have also appeared."*
- *"I am experiencing very high fever and chills every night. It is really concerning me. Moreover, I don’t feel like eating anything and my back, arms, legs pain a lot. There is a strange pain behing my eyes. I can’t do any physical activities."*
- *"I have been vomiting frequently and have lost my appetite as a result. There are rashes on my skin and my eyes pain, because of which I cannot sleep properly."*

### `Fungal infection` (Training Samples):
- *"My skin frequently develops itchy bumps that can occasionally turn red or rough. Aside from that, my skin occasionally develops strange pimples that resemble small balls and some odd spots that are a different hue from the rest of my skin."*
- *"All over my body has been itching like crazy, and now there are red areas all over. Additionally, some of the patches have a different tone than my natural skin. And on my skin, there are these lumps or pimples that have developed."*

### `Psoriasis` (Training Samples):
- *"There is a silver like dusting on my skin, especially on my lower back and scalp. This dusting is made up of small scales that flake off easily when I scratch them."*
- *"I've had trouble sleeping because of the itching and pain produced by the rash. My nails have little dents, which is really alarming. Moreover, my joints pain everyday and I have no idea what is causing it."*

### `allergy` (Training Samples):
- *"I have itchy and red skin. Sometimes these result in flaking. My face and lips puff up and causes a lot of inconvinience. I get watery eyes sometimes and headches when the puffing gets severe"*
- *"I have nasal congestion and a blocked nose. On occasion, during the night, I have stiffness and chest pain. Additionally, I've lost my senses of taste and smell."*

### `urinary tract infection` (Training Samples):
- *"I been having very low temperatues and a very foul smelling pee. The area near my kidneys hurt a lot and I cant seem to hold my pee. I often get these uncontrollable urges"*
- *"I have to go to the bathroom a lot, but generally only a few drops of urine come out. I have severe stomach pains and an overwhelming want to urinate."*

### `Typhoid` (Training Samples):
- *"I've been experiencing chills, fever, and extreme stomach discomfort. I've been generally unhappy, and I can't seem to get rid of these symptoms."*

### `Hypertension` (Training Samples):
- *"Headache, chest discomfort, dizziness, lack of balance, and difficulty concentrating are some of the symptoms I've been having."*

---

### Out-of-Distribution (OOD) Findings:

1. **Syntactic & Lexical Style Difference**:
   - **Training Set Writing Style**: Highly detailed, multi-sentence narrative paragraphs describing 4–6 co-occurring clinical symptoms in formal language (e.g., *"I have been suffering from a continuous cough, high grade fever, shivering, and severe chest pressure for 4 days."*).
   - **User Input Writing Style**: Short, conversational phrases (1 sentence or single words like *"fever"* or *"no fever but persistent cough"*).

2. **Softmax Output Dispersion**:
   - Because short input phrases lack the full co-occurring symptom context present in the training set, BioBERT spreads softmax probability mass across multiple candidate classes.
   - For example, *"fever"* is a non-specific symptom shared by 18 out of 24 disease classes in `Symptom2Disease`, leading to a low max confidence score (~8.65%).

3. **Calibration & Uncertainty**:
   - **Low Confidence Is Working Correctly**: Rather than being overconfident, the model correctly outputs low confidence (< 30%) when given ambiguous or short inputs.
   - The safety guardrail (70% threshold) appropriately flags these predictions for human review.

---

## 7. Confidence & Calibration Analysis

| Metric | Correct Predictions | Incorrect Predictions |
| :--- | :---: | :---: |
| **Mean Confidence** | **46.64%** | **20.28%** |
| **Median Confidence** | **46.93%** | **18.51%** |

- **Separation**: The mean confidence for correct test set predictions (**46.64%**) is more than **2x higher** than for incorrect predictions (**20.28%**).
- **All 11 Test Errors**: Had confidence scores below **30%**.
- **Conclusion**: The model exhibits well-separated confidence behavior and is **not overconfident** on ambiguous or misclassified examples.

---

## 8. Potential Causes of Unexpected Predictions

1. **Dataset Syntactic Homogeneity**:
   - `Symptom2Disease` contains ~1,153 long narrative paragraphs written in a specific clinical template.
   - Short user queries act as Out-of-Distribution (OOD) inputs relative to training paragraph length.

2. **Non-Specific Symptom Terms**:
   - Symptoms like *"fever"*, *"headache"*, and *"cough"* are shared across many diseases (e.g., Dengue, Malaria, Common Cold, Typhoid, Pneumonia).
   - Without secondary symptoms (e.g., *"retro-orbital pain"* for Dengue vs *"stiff neck"* for Migraine), the multi-class softmax cannot resolve a single high-probability class.

---

## 9. Recommended Future Improvements

1. **Data Augmentation for Short Queries**:
   - Generate synthetic short/conversational symptom variations during data preprocessing to bridge the OOD gap between long paragraphs and brief user inputs.
2. **Multi-Label Classification**:
   - Transition from single-label 24-class classification to multi-label classification to support overlapping symptom profiles and co-morbidities.
3. **Out-of-Distribution (OOD) Detector**:
   - Add explicit entropy-based or Mahalanobis-distance OOD detection to instantly inform users when input text is too short or ambiguous for reliable inference.

---
*Report auto-generated by MedXAI Diagnostic Engine.*