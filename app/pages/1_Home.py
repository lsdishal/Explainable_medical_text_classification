"""MedXAI — 1_Home page (sidebar navigation alias)."""
import sys
import streamlit as st
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.theme import apply_theme
from app.components.header import render_header
from app.components.sidebar import render_sidebar
from app.components.footer import render_footer
from app.components.cards import feature_card

st.set_page_config(page_title="MedXAI — Home", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
apply_theme()
render_header()
render_sidebar()

st.markdown("""
<div class="mxai-hero">
    <div class="mxai-hero-title">MedXAI</div>
    <div style="font-size:1.25rem; color:#3d5140; font-weight:500; margin-bottom:0.5rem;">
        Explainable Medical Text Classification
    </div>
    <div class="mxai-hero-sub">
        Transformer-based medical text classification with transparent,<br>
        human-readable model explanations.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### What MedXAI Does")
f1, f2, f3, f4 = st.columns(4)
cards = [
    ("🧠", "Prediction",      "BioBERT classifies symptom descriptions into 24 disease categories with top-3 probability output."),
    ("💡", "Explainability",  "Attention · SHAP · LIME · Integrated Gradients reveal which symptom tokens drove each prediction."),
    ("🛡️", "Confidence",      "Calibration analysis and configurable threshold automatically flag low-confidence predictions."),
    ("📈", "Model Analytics", "Confusion matrix · per-class F1 · training curves · error analysis on 116 held-out test samples."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], cards):
    with col:
        st.markdown(feature_card(icon, title, desc), unsafe_allow_html=True)

st.markdown("---")
st.markdown("### System Architecture")
st.code("""
 [ Patient Symptom Text ]
          │
          ▼
 [ Clinical Text Cleaner ]  ──►  [ BioBERT Tokenizer ]
          │
          ▼
 [ BertForSequenceClassification  ·  dmis-lab/biobert-v1.1 ]
          │
          ├───────────────────────►  [ Top-3 Disease Probabilities ]
          │
          ▼
 [ ExplanationManager (4-method XAI) ]
      ├── 1. Model Attention Signal  (non-causal)
      ├── 2. SHAP  (Shapley additive explanations)
      ├── 3. LIME  (local linear surrogate)
      └── 4. Captum Integrated Gradients  (axiomatic)
          │
          ▼
 [ Explanation Agreement  →  "Consistently influential features" ]
""", language="text")

st.markdown("---")
st.markdown("### At a Glance")
s1, s2, s3, s4, s5 = st.columns(5)
for col, (val, lbl) in zip([s1, s2, s3, s4, s5], [
    ("24","Disease Classes"),("1,153","Cleaned Samples"),
    ("91.4%","Best Val Acc"),("90.5%","Test Accuracy"),("4","XAI Methods")
]):
    with col:
        st.markdown(f"""
        <div class="mxai-card" style="text-align:center;padding:1.1rem 0.8rem;">
            <div style="font-size:1.7rem;font-weight:750;color:#1b5e20;">{val}</div>
            <div style="font-size:0.75rem;color:#6b7f6d;font-weight:500;margin-top:3px;">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

render_footer()
