# MedXAI: Project Report Notes & Research Methodology

---

## 1. Research Methodology & Clinical Problem Statement
Deep learning models—specifically Transformer sequence classifiers like BioBERT—achieve state-of-the-art performance in processing natural-language clinical symptom text. However, their opaque black-box operation impedes adoption in high-stakes healthcare environments.

**MedXAI** addresses this bottleneck by coupling high-accuracy Transformer models with a multi-method Explainable AI (XAI) engine and Responsible AI calibration guardrails.

---

## 2. System Architecture & Component Breakdown

```
[ Patient Symptom Text ]
           │
           ▼
[ Medical Text Cleaner ] ──► (Preserves negations: "no", "not", "without"; strips HTML)
           │
           ▼
[ BioBERT / Transformer Tokenizer ]
           │
           ▼
[ PyTorch Sequence Classifier ] ──► [ Top-3 Disease Predictions + Softmax Probabilities ]
           │
           ▼
[ Unified XAI Engine ] ──► Attention + SHAP + LIME + Captum Integrated Gradients
           │
           ▼
[ Explanation Agreement ] ──► Highlights "Consistently influential features"
           │
           ▼
[ Streamlit Healthcare Web Platform ]
```

### Key Modules:
- `src/preprocessing/`: Sanitize clinical text, handle Hugging Face REST API pagination fallbacks, and compute 80/10/10 reproducible stratified splits.
- `src/training/`: Fine-tune sequence classification models with AdamW optimizer, validation Macro F1 best-model checkpointing, early stopping (patience=3), and seed setting.
- `src/evaluation/`: Test evaluation metrics, error analysis by confidence descending, confusion matrix generation, and comparative model benchmark.
- `src/explainability/`: Attention signal extraction, SHAP Shapley values, LIME local linear surrogates, Captum LayerIntegratedGradients attributions, MD5 caching, and explanation agreement analysis.
- `src/reliability/`: Confidence Calibration (ECE, MCE, Reliability Diagrams) and Safety Guardrails with exact low-confidence warning flags ($< 0.70$).
- `app/`: Multi-page Streamlit application with 7 dedicated pages (Home, Prediction, Explainability, Model Evaluation, Dataset Analysis, Error Analysis, About).

---

## 3. Experiment Design & Preprocessing Protocol

### Dataset Specification
- **Dataset**: `Symptom2Disease` (Kaggle / Hugging Face `NeuronZero/Symptom2Disease`).
- **Raw Samples**: 1,200 symptom descriptions across 24 disease classes.
- **Deduplication**: 47 duplicate records removed $\rightarrow$ 1,153 clean records.
- **Stratified Split**: 80% Training (921 samples), 10% Validation (116 samples), 10% Test (116 samples).

### Medical Text Cleaning Rules
1. **HTML & Formatting**: Unescape HTML entities (`&amp;` $\rightarrow$ `&`) and strip HTML tags.
2. **Whitespace Normalization**: Convert multiple spaces/newlines to single space.
3. **Negation Word Preservation**: Strictly preserve medical negation cues (`no`, `not`, `without`, `never`, `neither`, `nor`, `absent`, `lack`).
4. **No Aggressive Stemming/Lemmatization**: Retain full contextual word tokens for Transformer tokenization.

---

## 4. Evaluation Methodology

### Metrics Calculated:
- **Accuracy**: Overall prediction correctness on held-out test set.
- **Macro Precision, Recall, and F1**: Unweighted arithmetic mean across all 24 classes, ensuring equal weighting for all disease categories regardless of class size.
- **Weighted F1**: Class-frequency weighted F1 score.

### Error Analysis Protocol:
- Categorize predictions into correct vs incorrect.
- Sort test errors by prediction confidence descending to investigate high-confidence failures.
- Calculate top confused disease class pairs.

---

## 5. Explainability (XAI) Methodology

MedXAI incorporates four complementary interpretability methods:

1. **Model Attention Signal**:
   - Aggregates multi-head self-attention weights from the `[CLS]` token across layers.
   - Labeled explicitly as **"Model Attention Signal"** with non-causal disclaimer.
2. **SHAP (Shapley Additive exPlanations)**:
   - Game-theoretic marginal contributions measuring feature impact on target disease logit.
3. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Perturbs text tokens and fits a local linear surrogate model around the prediction instance.
4. **Captum Integrated Gradients**:
   - Integrates model gradients along a straight path from a zero embedding baseline to token embeddings.

### Explanation Agreement:
When $\ge 2$ independent XAI methods highlight the same symptom word, the system categorizes it as a **"Consistently influential feature"**.

---

## 6. Responsible AI, Calibration & Limitations

### Calibration Analysis
- Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across binned confidence scores.
- Generates Reliability Diagrams, Confidence Histograms, and Accuracy vs Confidence threshold trade-off curves.

### System Limitations
1. **Clinical Bounds**: Academic decision-support prototype; NOT a clinical diagnostic tool.
2. **Sequence Length**: Input text truncated to 128 max tokens.
3. **Text-Only Scope**: Lacks patient clinical vitals, lab results, demographics, or imaging.
4. **Correlational XAI**: Feature attributions reflect model correlation, not clinical causality.

---

## 7. Future Work & Extensions

1. **Multimodal EHR Integration**: Incorporating structured tabular EHR vitals and lab values alongside unstructured text.
2. **Multi-Label Co-Morbidity Support**: Extending classification from single-label to multi-label disease predictions.
3. **Interactive Physician Feedback**: Active learning pipeline incorporating clinician validation to refine model attributions.
