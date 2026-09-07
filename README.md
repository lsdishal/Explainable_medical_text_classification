# MedXAI: Explainable Medical Text Classification using Transformer-Based NLP Models

---

## 1. Project Title
**MedXAI: Explainable Medical Text Classification using Transformer-Based NLP Models**

---

## 2. Problem Statement
Deep learning models—especially domain-specific Transformer architectures like BioBERT—achieve remarkable diagnostic classification accuracy on natural-language medical symptom text. However, their black-box nature hinders adoption in clinical environments where practitioners cannot trust predictions without understanding **why** a specific disease candidate was output by the system.

---

## 3. Project Objectives
1. **Clinical Symptom Classification**: Classify natural-language symptom descriptions into 24 disease categories with high accuracy and macro F1 scores.
2. **Multi-Method Explainability**: Illuminate model reasoning using four complementary interpretability techniques: Model Attention Signal, SHAP, LIME, and Captum Integrated Gradients.
3. **Explanation Agreement Analysis**: Identify recurring symptom words highlighted across multiple XAI methods and categorize them as *Consistently influential features*.
4. **Responsible AI & Reliability**: Integrate confidence calibration (ECE/MCE), low-confidence safety warning flags ($< 70\%$), and strict non-clinical decision-support disclaimers.

---

## 4. System Architecture

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
[ MedXAI Streamlit Web Platform ]
```

---

## 5. Dataset Source
- **Dataset**: `Symptom2Disease`
- **Source**: Kaggle (`niyarrbarman/symptom2disease`) & Hugging Face mirror (`NeuronZero/Symptom2Disease`).
- **Scope**: 1,200 raw symptom text samples across 24 disease classes.

---

## 6. Dataset Preprocessing Pipeline
1. **Validation & Index Cleanup**: Validated required `text` and `label` columns; removed unneeded CSV index columns (`Unnamed: 0`).
2. **Deduplication**: Detected and removed 47 duplicate records (1,200 raw $\rightarrow$ 1,153 cleaned records).
3. **Clinical Cleaning**: Stripped HTML tags/unescaped markup and normalized whitespace.
4. **Negation Preservation**: Strictly preserved medical negation cues (`no`, `not`, `without`, `never`, `neither`, `nor`, `absent`, `lack`).
5. **Contextual Token Integrity**: Avoided aggressive stemming or lemmatization to preserve full sentence context for Transformer tokenization.
6. **Stratified Split**: 80% Train (921 samples), 10% Validation (116 samples), 10% Test (116 samples).

---

## 7. Models
1. **BioBERT (Primary Model)**: `dmis-lab/biobert-v1.1` fine-tuned on PubMed biomedical corpora.
2. **BERT (Baseline Model)**: `bert-base-uncased`.
3. **DistilBERT (Lightweight Baseline)**: `distilbert-base-uncased`.

---

## 8. Training Pipeline
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: AdamW (learning rate: $3 \times 10^{-5}$, weight decay: $0.01$).
- **Checkpointing**: Saves best model checkpoint based on validation Macro F1 score (`models/best_model`).
- **Early Stopping**: Patience of 3 epochs.
- **Reproducibility**: Set seed = 42 across PyTorch, NumPy, and random.
- **Hardware Acceleration**: Auto-detects CUDA GPU, Apple Silicon MPS GPU, or CPU.

---

## 9. Evaluation Methodology
- **Metrics**: Accuracy, Precision, Recall, Macro F1, Weighted F1 on held-out test set.
- **Error Analysis**: Misclassifications sorted by confidence score descending to investigate high-confidence errors.
- **Visualizations**: Confusion matrix heatmaps, confidence histograms, and model comparison charts in `outputs/figures/`.

---

## 10. Explainability (XAI) Methods
1. **Model Attention Signal**: Multi-head self-attention weights extracted from `[CLS]` token across layers (labeled with non-causal disclaimer).
2. **SHAP**: Shapley additive exPlanations measuring marginal feature contributions.
3. **LIME**: Local linear surrogate model feature weights.
4. **Captum Integrated Gradients**: Axiomatic gradient attributions integrated along path from zero baseline to token embeddings.
5. **Explanation Agreement**: Highlights symptom words highlighted by $\ge 2$ XAI methods and labels them **"Consistently influential feature"**.

---

## 11. Application (MedXAI Streamlit Platform)
A white & soft-green healthcare-themed Streamlit platform (`app/app.py`) featuring 7 navigation pages:
1. `🏠 1. Home`: Project overview, capabilities, and architecture flowchart.
2. `🔬 2. Prediction`: Symptom text input area, preset example buttons, Prediction Card, Top 3 probability bars, low-confidence warning banner.
3. `💡 3. Explainability`: Interactive 4-method XAI token heatmap viewer and Explanation Agreement section.
4. `📊 4. Model Evaluation`: Empirical test metrics for BioBERT, BERT, DistilBERT loaded directly from saved reports (`outputs/reports/model_comparison.csv`).
5. `📈 5. Dataset Analysis`: Real Symptom2Disease stats loaded from `data/processed/dataset_statistics.json`.
6. `🔍 6. Error Analysis`: Held-out test set misclassifications table with interactive filtering.
7. `ℹ️ 7. About`: Project documentation, objectives, tech stack, ethics, and future scope.

---

## 12. Installation Instructions

```bash
# 1. Clone repository & enter workspace
git clone <repository_url>
cd "NLP PROJECT"

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 13. Usage & Execution Commands

### Run Master Automated Pipeline
```bash
python3 run_pipeline.py --model primary
```

### Launch MedXAI Streamlit Web Application
```bash
streamlit run app/app.py
```

### Run Individual Modules
```bash
# Preprocessing only
python3 -m src.preprocessing.dataset_preparation

# Training only
python3 -m src.training.train --model primary

# Evaluation only
python3 -m src.evaluation.evaluate --model primary

# Confidence Calibration only
python3 -m src.reliability.calibration

# Run all unit tests
python3 -m pytest tests/
```

---

## 14. Project Structure

```
medical-text-xai/
├── config.yaml                     # Central YAML configuration
├── requirements.txt                # Locked Python dependencies
├── setup_env.sh                    # Environment setup script
├── run_pipeline.py                 # Master CLI pipeline runner
├── README.md                       # Documentation & execution guide
├── PROJECT_REPORT_NOTES.md         # In-depth research report notes
│
├── data/                           # Raw & processed dataset splits
├── models/                         # Trained checkpoints & saved tokenizers
├── notebooks/                      # Exploratory Jupyter Notebooks
├── src/                            # Modular Python package source code
│   ├── data/                       # Data loader & HF REST API fallback
│   ├── preprocessing/              # Preprocessing, cleaner & label encoder
│   ├── training/                   # Model loader, PyTorch trainer & metrics
│   ├── evaluation/                 # Evaluator & error analysis engine
│   ├── explainability/             # Attention, SHAP, LIME, Captum & Manager
│   ├── inference/                  # Prediction engine & confidence estimator
│   ├── reliability/                # Safety guardrails & calibration engine
│   └── utils/                      # Logger, Config, Metrics
├── app/                            # Streamlit Web Application (MedXAI)
│   ├── app.py                      # Multi-page navigation router
│   ├── styles/                     # White & soft-green healthcare CSS
│   ├── components/                 # Header, Sidebar, Prediction Card, Tabs, Footer
│   └── pages/                      # 7 Dedicated Application Pages
├── outputs/                        # Plots, figures & reports
└── tests/                          # Automated Pytest test suites
```

---

## 15. System Limitations
1. **Academic Decision Support**: Prototype system designed for academic and research evaluation; **NOT** a clinical diagnostic tool.
2. **Sequence Truncation**: Inputs truncated to a maximum of 128 tokens.
3. **Text-Only Scope**: Lacks patient clinical vitals, lab results, demographics, or imaging.
4. **Correlational XAI**: Feature attributions indicate statistical correlation, **NOT clinical causality**.

---

## 16. Ethical Considerations & Responsible AI
- **No Autonomous Diagnosis**: Does NOT provide clinical diagnoses, prescribe medication, or recommend treatment plans.
- **Human-in-the-Loop**: All predictions must be independently evaluated by a licensed physician.
- **Low-Confidence Warning**: Predictions with confidence score $< 0.70$ automatically trigger low-confidence safety warning flags.

---

## 17. Future Scope
- Integration of Electronic Health Record (EHR) tabular data alongside text.
- Expansion to multi-label disease co-morbidity classification.
- Continuous active learning from clinician validation feedback.

---

## 18. Team Information
- **Project Title**: Explainable Medical Text Classification using Transformer-Based NLP Models
- **Institution / Lab**: NLP & Healthcare AI Research Lab
- **Contact / Maintainer**: Advanced Agentic Coding Pair Programmer & AI Team
