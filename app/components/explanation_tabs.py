import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Tuple

def render_token_highlight_with_legend(token_scores: List[Tuple[str, float]], title: str = "Feature Importance Heatmap"):
    """Renders HTML token/word heatmap with legend explaining highlight intensity."""
    if not token_scores:
        st.info("No feature attributions available.")
        return

    scores = [abs(s) for _, s in token_scores]
    max_s = max(scores) if scores else 1.0
    if max_s == 0:
        max_s = 1.0

    html_spans = []
    for token, score in token_scores:
        norm_score = score / max_s
        if norm_score >= 0:
            # Positive contribution (Soft Green)
            alpha = min(abs(norm_score), 1.0) * 0.75
            bg_color = f"rgba(76, 175, 80, {alpha:.2f})"
        else:
            # Negative contribution (Soft Coral)
            alpha = min(abs(norm_score), 1.0) * 0.75
            bg_color = f"rgba(239, 83, 80, {alpha:.2f})"

        span = f'<span style="background-color: {bg_color}; padding: 4px 8px; margin: 3px; border-radius: 4px; display: inline-block; font-weight: 600;" title="Attribution: {score:.4f}">{token}</span>'
        html_spans.append(span)

    # Legend & Heatmap HTML
    st.markdown(f"""
    <div class="legend-box" style="margin-bottom:10px;">
        <strong>🎨 Highlight Legend:</strong> &nbsp;
        <span style="background-color: rgba(76, 175, 80, 0.6); padding: 2px 8px; border-radius: 3px; font-weight: 600;">Green</span> Increases Disease Probability &nbsp;|&nbsp; 
        <span style="background-color: rgba(239, 83, 80, 0.6); padding: 2px 8px; border-radius: 3px; font-weight: 600;">Red/Coral</span> Decreases Probability
    </div>
    <div class="token-container">
        <h4 style="margin-top: 0; color: #2e7d32;">{title}</h4>
        <div style="line-height: 2.2;">{" ".join(html_spans)}</div>
    </div>
    """, unsafe_allow_html=True)

def render_raw_token_research_mode(raw_scores: List[Tuple[str, float]], method_name: str):
    """Renders Research Mode expander with raw subword Transformer tokens and attributions."""
    with st.expander(f"🔬 Technical Token-Level Details ({method_name} - Research Mode)", expanded=False):
        st.caption("Raw subword tokens (WordPiece `##` fragments and Transformer special tokens) prior to human-readable aggregation.")
        if raw_scores:
            df_raw = pd.DataFrame(raw_scores, columns=["Raw Token", "Raw Attribution Score"])
            st.dataframe(df_raw, width="stretch", hide_index=True)
        else:
            st.info("No raw token attributions available.")

def render_explanation_tabs(xai_outputs: Dict[str, Any], prediction_result: Dict[str, Any], model_name: str):
    """Renders interactive human-readable explanation tabs across Attention, SHAP, LIME, and Integrated Gradients."""
    st.markdown("### 🔍 Model Interpretability & Feature Attribution")

    tabs = st.tabs([
        "📋 Overview & Agreement",
        "👁️ Attention",
        "📊 SHAP",
        "📈 LIME",
        "⚡ Integrated Gradients",
        "ℹ️ Model Information"
    ])

    summary_records = xai_outputs.get("summary_table", [])
    consistent_list = xai_outputs.get("consistent_influential_features", [])

    # ==================== TAB 1: OVERVIEW & AGREEMENT ====================
    with tabs[0]:
        st.markdown("#### Human-Readable Symptom Feature Overview")
        
        # 1. Primary Highlighted Words Callout
        top_words = [f['word'].upper() for f in consistent_list[:6]]
        if not top_words and summary_records:
            # Fallback to top words by attention/SHAP
            sorted_recs = sorted(summary_records, key=lambda x: abs(x.get("shap_score", 0)) + abs(x.get("lime_score", 0)) + abs(x.get("integrated_gradients_score", 0)), reverse=True)
            top_words = [r['word'].upper() for r in sorted_recs[:4]]

        if top_words:
            pills_html = " ".join([f'<span style="background:#e8f5e9; border:1px solid #a5d6a7; color:#1b5e20; padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.9rem; margin-right:6px;">{w}</span>' for w in top_words])
            st.markdown(f"""
            <div class="mxai-card" style="padding:1.2rem; border-left:4px solid #2e7d32; margin-bottom:1.2rem;">
                <div style="font-size:0.78rem; font-weight:700; color:#6b7f6d; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                    🎯 Key Medical Features Driving Prediction
                </div>
                <div style="font-size:1.05rem; font-weight:600; color:#1b5e20; margin-bottom:10px;">
                    The model focused on:
                </div>
                <div>{pills_html}</div>
            </div>
            """, unsafe_allow_html=True)

        col_o1, col_o2 = st.columns([1, 1])

        with col_o1:
            st.markdown(f"**Predicted Disease:** `{xai_outputs.get('prediction', 'Unknown')}`")
            st.markdown(f"**Confidence:** `{prediction_result.get('confidence_percentage', '0.0%')}`")
            st.markdown(f"**Confidence Category:** `{prediction_result.get('confidence_category', 'Low')}`")

            if prediction_result.get("low_confidence_warning"):
                st.warning("⚠️ **Low Confidence Flag**: Prediction confidence is below threshold.")

        with col_o2:
            st.markdown("##### 🌟 Multi-Method Explanation Agreement")
            if consistent_list:
                st.caption("Human-readable symptom words flagged by $\ge 2$ XAI attribution methods:")
                for feat in consistent_list:
                    st.markdown(f"- **{feat['word'].upper()}** *(Agreement: {feat.get('agreement_ratio', str(feat['highlight_count'])+'/4')} methods)*")
            else:
                st.info("No single feature met multi-method threshold criteria.")

        st.markdown("---")
        st.markdown("#### 🤝 Human-Readable Feature Agreement Table")
        st.caption("Normalized word attributions across all four XAI algorithms.")

        if summary_records:
            table_rows = []
            for r in summary_records:
                table_rows.append({
                    "Feature": r["word"],
                    "Attention": "✓" if r.get("has_attention", r.get("attention_score", 0) > 0.15) else "—",
                    "SHAP": "✓" if r.get("has_shap", abs(r.get("shap_score", 0)) > 0.005) else "—",
                    "LIME": "✓" if r.get("has_lime", abs(r.get("lime_score", 0)) > 0.005) else "—",
                    "Integrated Gradients": "✓" if r.get("has_ig", abs(r.get("integrated_gradients_score", 0)) > 0.02) else "—",
                    "Agreement": r.get("agreement_ratio", f"{r.get('highlight_count', 0)}/4")
                })
            df_agree = pd.DataFrame(table_rows)
            # Filter rows with at least 1 hit
            df_agree_filtered = df_agree[df_agree["Agreement"] != "0/4"]
            if not df_agree_filtered.empty:
                st.dataframe(df_agree_filtered, width="stretch", hide_index=True)
            else:
                st.dataframe(df_agree.head(10), width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### Sentence Highlight (Human-Readable Words)")
        att_words = xai_outputs.get("attention_importance", {}).get("word_attributions", [])
        render_token_highlight_with_legend(att_words, "Input Text Word Importance")

    # ==================== TAB 2: ATTENTION ====================
    with tabs[1]:
        st.markdown("#### Model Attention Signal")
        st.info("📌 **Methodological Note**: Attention represents an internal Transformer signal and should not automatically be interpreted as a causal explanation.")
        
        att_info = xai_outputs.get("attention_importance", {})
        att_words = att_info.get("word_attributions", att_info.get("token_scores", []))
        att_raw = att_info.get("raw_token_scores", att_info.get("token_scores", []))

        render_token_highlight_with_legend(att_words, "Human-Readable Attention Heatmap")

        if att_words:
            df_att = pd.DataFrame(att_words, columns=["Word", "Attention Score"])
            fig = px.bar(df_att, x="Word", y="Attention Score", color="Attention Score", color_continuous_scale="Greens", title="Attention Weight Distribution (Human-Readable Words)")
            fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, use_container_width=True)

        render_raw_token_research_mode(att_raw, "Attention")

    # ==================== TAB 3: SHAP ====================
    with tabs[2]:
        st.markdown("#### SHAP (Shapley Additive exPlanations)")
        st.info("📌 **Methodological Note**: SHAP estimates feature contribution relative to the model's prediction.")

        shap_info = xai_outputs.get("shap_importance", {})
        shap_words = shap_info.get("word_attributions", shap_info.get("token_scores", []))
        shap_raw = shap_info.get("raw_token_scores", shap_info.get("token_scores", []))

        render_token_highlight_with_legend(shap_words, "Human-Readable SHAP Heatmap")

        if shap_words:
            df_shap = pd.DataFrame(shap_words, columns=["Word", "SHAP Score"])
            fig = px.bar(df_shap, x="Word", y="SHAP Score", color="SHAP Score", color_continuous_scale="Greens", title="SHAP Contribution Values (Human-Readable Words)")
            fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, use_container_width=True)

        render_raw_token_research_mode(shap_raw, "SHAP")

    # ==================== TAB 4: LIME ====================
    with tabs[3]:
        st.markdown("#### LIME (Local Interpretable Model-agnostic Explanations)")
        st.info("📌 **Methodological Note**: LIME approximates the model locally using perturbed input samples.")

        lime_info = xai_outputs.get("lime_importance", {})
        lime_words = lime_info.get("word_attributions", lime_info.get("token_scores", []))
        lime_raw = lime_info.get("raw_token_scores", lime_info.get("token_scores", []))

        render_token_highlight_with_legend(lime_words, "Human-Readable LIME Heatmap")

        if lime_words:
            df_lime = pd.DataFrame(lime_words, columns=["Word", "LIME Score"])
            fig = px.bar(df_lime, x="Word", y="LIME Score", color="LIME Score", color_continuous_scale="Greens", title="LIME Local Surrogate Feature Weights (Human-Readable Words)")
            fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, use_container_width=True)

        render_raw_token_research_mode(lime_raw, "LIME")

    # ==================== TAB 5: INTEGRATED GRADIENTS ====================
    with tabs[4]:
        st.markdown("#### Captum / Integrated Gradients")
        st.info("📌 **Methodological Note**: Integrated Gradients attributes prediction differences to input features along a path from a baseline.")

        ig_info = xai_outputs.get("integrated_gradients_importance", {})
        ig_words = ig_info.get("word_attributions", ig_info.get("token_scores", []))
        ig_raw = ig_info.get("raw_token_scores", ig_info.get("token_scores", []))

        render_token_highlight_with_legend(ig_words, "Human-Readable Integrated Gradients Heatmap")

        if ig_words:
            df_cap = pd.DataFrame(ig_words, columns=["Word", "Integrated Gradients Score"])
            fig = px.bar(df_cap, x="Word", y="Integrated Gradients Score", color="Integrated Gradients Score", color_continuous_scale="Greens", title="Integrated Gradients Attributions (Human-Readable Words)")
            fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, use_container_width=True)

        render_raw_token_research_mode(ig_raw, "Integrated Gradients")

    # ==================== TAB 6: MODEL INFORMATION ====================
    with tabs[5]:
        st.markdown("#### ℹ️ Transformer Model Specifications & Metadata")

        st.markdown(f"""
        - **Model Name**: `{model_name}`
        - **Model Architecture**: PyTorch `AutoModelForSequenceClassification`
        - **Training Dataset**: `Symptom2Disease` (Hugging Face mirror: `NeuronZero/Symptom2Disease`)
        - **Number of Disease Classes**: `24 Categories`
        - **Sequence Length**: `128 max tokens`
        - **Optimizer**: `AdamW` (learning rate: $3 \times 10^{-5}$, weight decay: $0.01$)
        - **Evaluation Metrics**: Test Set Accuracy, Precision, Recall, Macro F1, Weighted F1
        """)
