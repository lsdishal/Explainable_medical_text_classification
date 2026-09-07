"""MedXAI Sidebar component — redesigned v2."""
import streamlit as st
from pathlib import Path
from src.utils.config import get_config


def render_sidebar(config_path: str = "config.yaml"):
    """Renders the redesigned MedXAI sidebar with system controls."""
    cfg = get_config(config_path)

    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1.2rem 0.5rem 0.8rem;">
        <div style="font-size:1.8rem; margin-bottom:4px;">🌿</div>
        <div style="font-size:1.05rem; font-weight:700; color:#1b5e20; letter-spacing:-0.02em;">MedXAI</div>
        <div style="font-size:0.72rem; color:#6b7f6d; margin-top:2px;">Explainable Medical AI</div>
    </div>
    <hr style="border-color:rgba(168,213,170,0.4); margin: 0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Model")

    selected_model_name = st.sidebar.selectbox(
        "Architecture",
        options=["BioBERT (Primary Model)", "BERT (Baseline)", "DistilBERT (Baseline)"],
        index=0,
        help="Select fine-tuned Transformer model checkpoint.",
        key="sb_model_select"
    )

    model_key_map = {
        "BioBERT (Primary Model)":   ("primary",             cfg.models.primary.save_dir),
        "BERT (Baseline)":           ("baseline_bert",       cfg.models.baseline_bert.save_dir),
        "DistilBERT (Baseline)":     ("baseline_distilbert", cfg.models.baseline_distilbert.save_dir),
    }
    model_key, save_dir = model_key_map[selected_model_name]
    model_exists = Path(save_dir).exists()

    if model_exists:
        st.sidebar.markdown(f"""
        <div style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:8px;
                    padding:8px 12px; font-size:0.78rem; color:#2e7d32; margin-top:6px;">
            ✅ Checkpoint ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="background:#fff8f0; border:1px solid #ffd6a5; border-radius:8px;
                    padding:8px 12px; font-size:0.78rem; color:#c05a00; margin-top:6px;">
            ⚠️ Using pretrained fallback
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🎚️ Parameters")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.50, max_value=0.95, value=0.70, step=0.05,
        help="Safety warning threshold — predictions below this flag as low-confidence.",
        key="sb_conf_threshold"
    )

    max_text_length = st.sidebar.slider(
        "Max Token Length",
        min_value=32, max_value=512,
        value=int(getattr(cfg.training, "max_length", 128)),
        step=32,
        help="Maximum input sequence length accepted by the tokenizer.",
        key="sb_max_len"
    )

    st.sidebar.markdown("""
    <hr style="border-color:rgba(168,213,170,0.35); margin: 1rem 0 0.8rem;">
    <div style="font-size:0.72rem; color:#8fa391; line-height:1.65; padding:0 0.2rem 1rem;">
        <strong style="color:#3d5140;">Dataset</strong><br>
        Symptom2Disease · 24 classes<br>
        921 train · 116 val · 116 test<br><br>
        <strong style="color:#3d5140;">Model</strong><br>
        dmis-lab/biobert-v1.1<br>
        Fine-tuned · Apple MPS GPU
    </div>
    """, unsafe_allow_html=True)

    return selected_model_name, model_key, save_dir, confidence_threshold, max_text_length
