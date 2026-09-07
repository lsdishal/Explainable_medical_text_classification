"""MedXAI — 7_About page (redesigned v2)."""
import sys
import streamlit as st
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.theme import apply_theme
from app.components.header import render_header
from app.components.sidebar import render_sidebar
from app.components.cards import section_tag
from app.components.footer import render_footer

st.set_page_config(page_title="MedXAI — About", page_icon="ℹ️", layout="wide", initial_sidebar_state="expanded")
apply_theme()
render_header()
render_sidebar()

st.markdown(f"""
<div style="margin-bottom:1.4rem;">
    {section_tag("Documentation & Ethics", "ℹ️")}
    <h2 style="margin:4px 0 4px; font-size:1.6rem;">About MedXAI</h2>
    <p style="color:#6b7f6d; font-size:0.9rem; margin:0;">
        Explainable Medical Text Classification using Transformer-Based NLP Models.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("""
    <div class="mxai-card">
        <div class="mxai-card-title">🎯 1. Problem Statement</div>
        <p style="font-size:0.88rem; color:#3d5140; line-height:1.6;">
            Deep learning Transformer models achieve high accuracy in medical text processing, but their black-box nature creates a barrier to clinical adoption. Healthcare practitioners cannot rely on uninterpretable predictions without understanding <strong>why</strong> a model reached a specific output.
        </p>
    </div>

    <div class="mxai-card">
        <div class="mxai-card-title">🚩 2. Project Objectives</div>
        <ul style="font-size:0.86rem; color:#3d5140; padding-left:1.2rem; margin:0; line-height:1.65;">
            <li><strong>Accurate Classification</strong>: Categorize symptom descriptions into 24 disease classes.</li>
            <li><strong>Multi-Method XAI</strong>: Feature attributions via Attention, SHAP, LIME, and Integrated Gradients.</li>
            <li><strong>Explanation Agreement</strong>: Identify symptom tokens consistently highlighted across algorithms.</li>
            <li><strong>Responsible AI</strong>: ECE calibration, low-confidence warnings (70%), and clinical disclaimers.</li>
        </ul>
    </div>

    <div class="mxai-card">
        <div class="mxai-card-title">📂 3. Dataset & Models</div>
        <p style="font-size:0.86rem; color:#3d5140; margin-bottom:8px;">
            <strong>Dataset:</strong> <code>Symptom2Disease</code> (1,153 cleaned samples, 24 disease classes)
        </p>
        <p style="font-size:0.86rem; color:#3d5140; margin-bottom:4px;">
            <strong>Models:</strong>
        </p>
        <ul style="font-size:0.84rem; color:#3d5140; padding-left:1.2rem; margin:0; line-height:1.6;">
            <li><code>dmis-lab/biobert-v1.1</code> (Primary Model — PubMed pre-trained)</li>
            <li><code>bert-base-uncased</code> (General NLP Baseline)</li>
            <li><code>distilbert-base-uncased</code> (Distilled Lightweight Baseline)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="mxai-card">
        <div class="mxai-card-title">💡 4. Explainability Framework</div>
        <div style="font-size:0.84rem; color:#3d5140; line-height:1.6;">
            <div style="margin-bottom:8px;">
                <strong style="color:#1b5e20;">Model Attention Signal:</strong><br>
                Multi-head self-attention from [CLS] token (labeled as non-causal contextual signal).
            </div>
            <div style="margin-bottom:8px;">
                <strong style="color:#1b5e20;">SHAP (Shapley Additive Explanations):</strong><br>
                Game-theoretic partition explainer for exact fair token attribution.
            </div>
            <div style="margin-bottom:8px;">
                <strong style="color:#1b5e20;">LIME (Local Linear Surrogate):</strong><br>
                Word-level importance scores from 250 local perturbation samples.
            </div>
            <div>
                <strong style="color:#1b5e20;">Captum Integrated Gradients:</strong><br>
                Axiomatic path-integral gradient attributions from zero-embedding baseline.
            </div>
        </div>
    </div>

    <div class="mxai-card">
        <div class="mxai-card-title">🛠️ 5. Technology Stack</div>
        <div style="margin-top:6px;">
            <span class="mxai-tech-pill">Python 3.12</span>
            <span class="mxai-tech-pill">PyTorch 2.x</span>
            <span class="mxai-tech-pill">BioBERT</span>
            <span class="mxai-tech-pill">HuggingFace</span>
            <span class="mxai-tech-pill">SHAP</span>
            <span class="mxai-tech-pill">LIME</span>
            <span class="mxai-tech-pill">Captum</span>
            <span class="mxai-tech-pill">scikit-learn</span>
            <span class="mxai-tech-pill">Plotly</span>
            <span class="mxai-tech-pill">Streamlit</span>
        </div>
    </div>

    <div class="mxai-card">
        <div class="mxai-card-title">🛡️ 6. Responsible AI & Ethics</div>
        <div class="mxai-warning" style="margin:4px 0 0; font-size:0.82rem;">
            Academic decision-support prototype only. Does not provide medical diagnoses, prescribe treatment, or replace clinician judgment.
        </div>
    </div>
    """, unsafe_allow_html=True)

render_footer()
