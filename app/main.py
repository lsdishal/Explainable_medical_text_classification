import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import get_config
from src.inference.predict import MedicalSymptomClassifier
from src.explainability.token_importance import XAIEngine
from src.evaluation.evaluator import generate_model_comparison
from src.reliability.safety import SafetyGuardrail, LOW_CONFIDENCE_WARNING_TEXT
from app.components.sidebar import render_sidebar
from app.components.visuals import (
    render_confidence_chart,
    render_token_highlight_html,
    render_xai_comparison_chart
)

st.set_page_config(
    page_title="Explainable Medical Text Classification (BioBERT XAI)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .disclaimer-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .low-conf-box {
        background-color: rgba(220, 53, 69, 0.15);
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 4px;
        margin-top: 15px;
        color: #ff8080;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_SYMPTOMS = [
    "Select a sample symptom description...",
    "I have been feeling severe itching on my skin with red scaly rashes and joint pain.",
    "High fever, intense headache, muscle aches, shivering with severe chills, and sweating.",
    "Persistent cough with yellowish phlegm, high fever, shortness of breath, and chest pain when breathing.",
    "Frequent urination, excessive thirst, extreme fatigue, sudden weight loss, and blurry vision.",
    "Severe headache, stiff neck, nausea, vomiting, and high sensitivity to bright light."
]

def main():
    selected_model_name, model_key, save_dir = render_sidebar()

    st.title("🩺 Explainable Medical Text Classification")
    st.markdown("##### *Academic AI-Assisted Decision Support with BioBERT & Multi-Method XAI*")

    safety = SafetyGuardrail()

    # Safety Disclaimer Header
    st.markdown(f"""
    <div class="disclaimer-box">
        <strong>⚠️ RESPONSIBLE AI & ACADEMIC DISCLAIMER:</strong> {safety.get_medical_disclaimer()}
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🔬 Disease Classifier",
        "💡 Explainable AI (XAI)",
        "📊 Model Performance & Comparison",
        "🛡️ Responsible AI, Safety & Calibration"
    ])

    # Instantiate Classifier
    classifier = None
    if Path(save_dir).exists():
        try:
            classifier = MedicalSymptomClassifier(model_dir=save_dir)
        except Exception as e:
            st.error(f"Error loading model from '{save_dir}': {e}")

    # ==================== TAB 1: CLASSIFIER ====================
    with tabs[0]:
        st.subheader("Patient Symptom Description Input")

        col1, col2 = st.columns([3, 2])
        with col1:
            preset = st.selectbox("Quick Preset Symptom Examples:", SAMPLE_SYMPTOMS)
            default_text = preset if preset != SAMPLE_SYMPTOMS[0] else ""
            user_input = st.text_area(
                "Describe patient symptoms in natural language:",
                value=default_text,
                height=140,
                placeholder="Example: Patient presents with persistent fever, severe body aches, shivering chills, and loss of appetite..."
            )
            classify_btn = st.button("🔍 Classify Disease & Predict Probabilities", type="primary", use_container_width=True)

        with col2:
            st.markdown("### Diagnosis Summary")
            if classify_btn or (user_input and preset != SAMPLE_SYMPTOMS[0]):
                if not classifier:
                    st.warning("Please train the selected model checkpoint first before running inference.")
                elif not user_input.strip():
                    st.warning("Please enter a valid symptom text description.")
                else:
                    with st.spinner("Analyzing symptoms via BioBERT classifier..."):
                        result = classifier.predict(user_input)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["last_result"] = result
                        st.session_state["last_input"] = user_input

                        primary_disease = result["primary_prediction"]
                        primary_conf = result["primary_confidence"]
                        is_low_conf = (primary_conf < safety.confidence_threshold)

                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color:#64b5f6; margin:0;">Top Predicted Disease</h4>
                            <h2 style="color:#ffffff; margin:5px 0;">{primary_disease}</h2>
                            <h4 style="color:{'#81c784' if not is_low_conf else '#e57373'}; margin:0;">
                                Softmax Probability: {primary_conf * 100:.2f}% ({'High/Moderate' if not is_low_conf else 'Low'})
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)

                        if is_low_conf:
                            st.markdown(f"""
                            <div class="low-conf-box">
                                <strong>🚨 LOW CONFIDENCE WARNING:</strong><br>
                                {LOW_CONFIDENCE_WARNING_TEXT}
                            </div>
                            """, unsafe_allow_html=True)

        if "last_result" in st.session_state and st.session_state.get("last_input") == user_input:
            result = st.session_state["last_result"]
            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(render_confidence_chart(result["top_k_predictions"]), use_container_width=True)
            with c2:
                st.markdown("#### Top-3 Predictions Detail")
                pred_df = pd.DataFrame(result["top_k_predictions"])[["disease", "confidence_percentage"]]
                pred_df.columns = ["Disease Category", "Confidence Score"]
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # ==================== TAB 2: EXPLAINABLE AI (XAI) ====================
    with tabs[1]:
        st.subheader("Model Interpretability & Feature Attribution")
        st.markdown("Understand *why* the Transformer model assigned the prediction using Attention, SHAP, LIME, and Integrated Gradients.")

        if "last_result" not in st.session_state or not classifier:
            st.info("👈 Please enter a symptom description in Tab 1 and click 'Classify Disease' to run XAI explanations.")
        else:
            result = st.session_state["last_result"]
            input_text = result["input_text"]
            top_class_id = result["top_k_predictions"][0]["class_id"]
            predicted_disease = result["primary_prediction"]

            st.write(f"Generating explanations for: **{predicted_disease}**")
            
            run_xai_btn = st.button("⚡ Generate Multi-Method XAI Explanations", type="secondary")

            if run_xai_btn or "xai_outputs" not in st.session_state:
                with st.spinner("Computing Attention, SHAP, LIME, and Integrated Gradients attributions..."):
                    xai_engine = XAIEngine(
                        model=classifier.model,
                        tokenizer=classifier.tokenizer,
                        label_list=classifier.label_list,
                        device=classifier.device
                    )
                    xai_outputs = xai_engine.explain_all(input_text, target_class_id=top_class_id)
                    st.session_state["xai_outputs"] = xai_outputs

            xai_outputs = st.session_state["xai_outputs"]

            st.markdown("---")
            xai_subtabs = st.tabs(["1. Attention Map", "2. SHAP Explanation", "3. LIME Explanation", "4. Integrated Gradients (Captum)", "5. XAI Comparison"])

            with xai_subtabs[0]:
                st.markdown("#### Model Attention Signal")
                st.caption("Transformer self-attention weights from [CLS] token to symptom words. NOTE: Attention signals reflect internal contextual alignment, NOT clinical causality.")
                html_att = render_token_highlight_html(xai_outputs["attention_scores"], "Model Attention Signal")
                st.markdown(html_att, unsafe_allow_html=True)

            with xai_subtabs[1]:
                st.markdown("#### SHAP (Shapley Additive exPlanations)")
                st.caption("Game-theoretic token attributions measuring contribution to predicted disease score.")
                html_shap = render_token_highlight_html(xai_outputs["shap_scores"], "SHAP Word Attributions")
                st.markdown(html_shap, unsafe_allow_html=True)

            with xai_subtabs[2]:
                st.markdown("#### LIME (Local Interpretable Model-agnostic Explanations)")
                st.caption("Local linear surrogate attributions around the current symptom input.")
                html_lime = render_token_highlight_html(xai_outputs["lime_scores"], "LIME Word Attributions")
                st.markdown(html_lime, unsafe_allow_html=True)

            with xai_subtabs[3]:
                st.markdown("#### Captum / Integrated Gradients")
                st.caption("Axiomatic gradient attributions integrated from zero baseline to token embeddings.")
                html_captum = render_token_highlight_html(xai_outputs["captum_scores"], "Integrated Gradients Attributions")
                st.markdown(html_captum, unsafe_allow_html=True)

            with xai_subtabs[4]:
                st.markdown("#### Side-by-Side Multi-Method Token Attribution")
                st.plotly_chart(render_xai_comparison_chart(xai_outputs["summary_df"]), use_container_width=True)
                st.dataframe(xai_outputs["summary_df"], use_container_width=True)

    # ==================== TAB 3: MODEL COMPARISON & EVALUATION ====================
    with tabs[2]:
        st.subheader("Model Performance Evaluation & Comparison")

        comp_df = generate_model_comparison()
        if comp_df.empty:
            st.info("No saved model metrics found. Train models using `python src/training/trainer.py` to populate performance comparisons.")
        else:
            st.markdown("#### Performance Metrics Summary (Test Set)")
            st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Confusion Matrix Visualizations")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        cfg = get_config()
        fig_dir = Path(cfg.paths.figures_dir)
        
        with col_m1:
            st.markdown("##### BioBERT")
            cm_bio = fig_dir / "confusion_matrix_biobert.png"
            if cm_bio.exists():
                st.image(str(cm_bio), use_container_width=True)
            else:
                st.caption("Run evaluation to generate confusion matrix.")

        with col_m2:
            st.markdown("##### BERT Baseline")
            cm_bert = fig_dir / "confusion_matrix_bert.png"
            if cm_bert.exists():
                st.image(str(cm_bert), use_container_width=True)
            else:
                st.caption("Run evaluation to generate confusion matrix.")

        with col_m3:
            st.markdown("##### DistilBERT Baseline")
            cm_distil = fig_dir / "confusion_matrix_distilbert.png"
            if cm_distil.exists():
                st.image(str(cm_distil), use_container_width=True)
            else:
                st.caption("Run evaluation to generate confusion matrix.")

    # ==================== TAB 4: RESPONSIBLE AI & SAFETY ====================
    with tabs[3]:
        st.subheader("🛡️ Responsible AI, Safety & System Boundaries")
        
        c_rel1, c_rel2 = st.columns(2)

        with c_rel1:
            st.markdown("#### 1. Medical Disclaimer & Boundaries")
            st.info(safety.get_medical_disclaimer())

            st.markdown("#### 2. Human-Review Recommendation")
            st.warning(safety.get_human_review_recommendation())

            st.markdown("#### 3. Model Limitations")
            for lim in safety.get_model_limitations():
                st.markdown(f"- {lim}")

        with c_rel2:
            st.markdown("#### 4. Dataset Limitations")
            for lim in safety.get_dataset_limitations():
                st.markdown(f"- {lim}")

            st.markdown("#### 5. Explanation (XAI) Limitations")
            for lim in safety.get_explanation_limitations():
                st.markdown(f"- {lim}")

        st.markdown("---")
        st.markdown("#### Model Confidence Calibration Plots")

        rel_fig = fig_dir / "reliability_diagram.png"
        conf_fig = fig_dir / "confidence_histogram.png"
        acc_fig = fig_dir / "accuracy_vs_confidence.png"

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("##### Reliability Diagram")
            if rel_fig.exists():
                st.image(str(rel_fig), use_container_width=True)
            else:
                st.caption("Run calibration script to generate plot.")

        with col_p2:
            st.markdown("##### Confidence Histogram")
            if conf_fig.exists():
                st.image(str(conf_fig), use_container_width=True)
            else:
                st.caption("Run calibration script to generate plot.")

        with col_p3:
            st.markdown("##### Accuracy vs Confidence")
            if acc_fig.exists():
                st.image(str(acc_fig), use_container_width=True)
            else:
                st.caption("Run calibration script to generate plot.")

if __name__ == "__main__":
    main()
