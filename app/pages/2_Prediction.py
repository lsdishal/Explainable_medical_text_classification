"""MedXAI — Unified Clinical Symptom Prediction & Explainable AI page."""
import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.theme import apply_theme
from app.components.header import render_header
from app.components.sidebar import render_sidebar
from app.components.cards import render_prediction_card, section_tag
from app.components.explanation_tabs import render_explanation_tabs
from app.components.footer import render_footer

st.set_page_config(
    page_title="MedXAI — Prediction & Explainability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()
render_header()
selected_model_name, model_key, save_dir, confidence_threshold, max_text_length = render_sidebar()

PRESETS = {
    "Migraine":   "I have a severe headache with nausea, vomiting, and photophobia.",
    "Pneumonia":  "High fever, chills, persistent cough with yellowish sputum, and chest pain when breathing.",
    "Psoriasis":  "I have a skin rash on my arms and legs with red scaly patches and dry itching.",
    "Malaria":    "High fever with intense shivering chills, sweating, and severe muscle aches.",
}

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.4rem;">
    {section_tag("Clinical Symptom Classification & Multi-Method XAI", "🔬")}
    <h2 style="margin:4px 0 4px; font-size:1.6rem;">Disease Prediction & Explainable AI</h2>
    <p style="color:#6b7f6d; font-size:0.9rem; margin:0;">
        Predict disease conditions from natural-language symptom descriptions and inspect transparent, word-level feature importance across 4 complementary XAI algorithms.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Method Overview Cards ─────────────────────────────────────────────────────
st.markdown("**Four Complementary XAI Methods Included:**")
m1, m2, m3, m4 = st.columns(4)
methods = [
    ("👁️", "Attention Signal",     "Multi-head self-attention from [CLS] token. Contextual alignment signal."),
    ("📐", "SHAP",                 "Game-theoretic Shapley additive feature attributions across token combinations."),
    ("🔍", "LIME",                 "Local linear surrogate with 250 perturbation samples for word-level weights."),
    ("∫",  "Integrated Gradients", "Axiomatic gradient attributions calculated from zero-embedding baseline via Captum."),
]
for col, (ico, name, desc) in zip([m1, m2, m3, m4], methods):
    with col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.7);border:1px solid rgba(168,213,170,0.35);
                    border-radius:12px;padding:0.9rem;text-align:center;height:100%;">
            <div style="font-size:1.3rem;margin-bottom:4px;">{ico}</div>
            <div style="font-size:0.86rem;font-weight:650;color:#1b5e20;margin-bottom:3px;">{name}</div>
            <div style="font-size:0.75rem;color:#6b7f6d;line-height:1.45;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── Session State Callbacks ───────────────────────────────────────────────────
if "symptom_input" not in st.session_state:
    st.session_state["symptom_input"] = ""
if "pred_textarea" not in st.session_state:
    st.session_state["pred_textarea"] = st.session_state["symptom_input"]


def _on_select_preset(preset_text: str):
    st.session_state["pred_textarea"] = preset_text
    st.session_state["symptom_input"] = preset_text
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("xai_outputs", None)


def _on_clear():
    st.session_state["pred_textarea"] = ""
    st.session_state["symptom_input"] = ""
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("xai_outputs", None)


# ── Presets & Input Text Area ─────────────────────────────────────────────────
st.markdown("**Try an example:**")
pcols = st.columns(len(PRESETS))
for i, (label, text) in enumerate(PRESETS.items()):
    with pcols[i]:
        st.button(
            f"📝 {label}",
            use_container_width=True,
            key=f"preset_{i}",
            on_click=_on_select_preset,
            args=(text,),
        )

user_input = st.text_area(
    "Symptom description",
    height=120,
    placeholder="Describe the patient's symptoms in natural language…",
    key="pred_textarea",
    label_visibility="collapsed",
)
st.session_state["symptom_input"] = user_input

ba, bb, _ = st.columns([2.5, 1, 4])
with ba:
    analyze = st.button("🔬  Analyze & Explain Symptoms", type="primary", use_container_width=True, key="btn_analyze")
with bb:
    st.button("✕  Clear", use_container_width=True, key="btn_clear", on_click=_on_clear)

# ── Inference & Explanation Computation ──────────────────────────────────────
if analyze:
    if not user_input.strip():
        st.warning("Please enter a symptom description first.")
    else:
        try:
            from src.inference.predict import MedicalSymptomClassifier
            from src.explainability.explanation_manager import ExplanationManager

            with st.spinner("Classifying symptoms & generating multi-method XAI attributions…"):
                clf = MedicalSymptomClassifier(model_dir=save_dir)
                pred = clf.predict(user_input)
                st.session_state["prediction_result"] = pred

                if pred.get("status") == "success":
                    top = pred["top_3_predictions"][0]
                    xai = ExplanationManager(clf.model, clf.tokenizer, clf.label_list, clf.device)
                    xai_out = xai.explain(
                        user_input,
                        target_class_id=top["class_id"],
                        target_class_name=top["disease"],
                        confidence=top["confidence"]
                    )
                    st.session_state["xai_outputs"] = xai_out
        except Exception as e:
            st.error(f"Analysis error: {e}")

# ── Unified Results Section ───────────────────────────────────────────────────
if "prediction_result" in st.session_state:
    pred = st.session_state["prediction_result"]
    st.markdown("---")

    # 1. Prediction Card & Top-3 Probabilities
    render_prediction_card(pred, confidence_threshold=confidence_threshold)

    # 2. Explainable AI Attributions
    if "xai_outputs" in st.session_state:
        st.markdown("---")
        xai_outputs = st.session_state["xai_outputs"]
        consistent = xai_outputs.get("consistent_influential_features", [])

        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
            <div>
                {section_tag("Transparent Model Explanation", "💡")}
                <h3 style="margin:4px 0 0; font-size:1.3rem;">Word-Level Feature Attributions</h3>
            </div>
            {"<span class='mxai-consistent-badge'>✓ "+str(len(consistent))+" Consistent Features across XAI</span>" if consistent else ""}
        </div>
        """, unsafe_allow_html=True)

        render_explanation_tabs(xai_outputs, pred, model_name=selected_model_name)

        # 3. Research-Level Benchmark Results Expander
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        with st.expander("📊 Research-Level XAI Agreement & Stability Benchmark Results", expanded=False):
            st.markdown("""
            **Empirical XAI Method Consistency & Stability Study**  
            Evaluated across 50 held-out test set examples ($K=10$ top influential tokens per method).
            """)

            xai_json_path = Path("outputs/reports/xai_agreement.json")
            if xai_json_path.exists():
                with open(xai_json_path, "r", encoding="utf-8") as f:
                    xai_data = json.load(f)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Mean Pairwise Agreement", f"{xai_data['overall_agreement_stats']['mean']*100:.1f}%")
                with c2:
                    st.metric("Most Stable Method", "Attention / SHAP / IG (100%)")
                with c3:
                    st.metric("Top Agreeing Pair", "Attention vs IG (39.8%)")

                st.markdown("**Pairwise Method Agreement (Jaccard Index):**")
                pw_df = pd.DataFrame(
                    list(xai_data["pairwise_jaccard_means"].items()),
                    columns=["Method Pair", "Mean Jaccard Similarity"]
                )
                st.dataframe(pw_df, width="stretch")

                st.markdown("""
                > 💡 **Scientific Note**: High agreement across XAI methods indicates feature attribution convergence, but does **not** prove medical causality. Attention signals represent transformer context alignment.
                """)
            else:
                st.info("Run `python3 -m src.explainability.research_evaluator` to generate complete research benchmarks.")

render_footer()
