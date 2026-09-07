import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, Any
from src.reliability.safety import LOW_CONFIDENCE_WARNING_TEXT

def render_prediction_card(prediction_result: Dict[str, Any], confidence_threshold: float = 0.70):
    """Renders Prediction Card, Top-3 disease probability chart, and safety warning alerts."""
    if "error" in prediction_result:
        st.error(prediction_result["error"])
        return

    disease = prediction_result["predicted_disease"]
    confidence = prediction_result["confidence"]
    conf_pct = prediction_result["confidence_percentage"]
    category = prediction_result["confidence_category"]
    top_3 = prediction_result["top_3_predictions"]
    is_low_conf = (confidence < confidence_threshold)

    badge_class = "confidence-badge-low" if is_low_conf else "confidence-badge-high"

    # Prediction Card HTML
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-title">Primary Predicted Disease Category</div>
        <div class="prediction-name">{disease}</div>
        <div style="margin-top: 10px;">
            <span class="{badge_class}">
                Confidence: {conf_pct} ({category} Confidence)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Low-Confidence Warning Alert
    if is_low_conf:
        st.markdown(f"""
        <div class="warning-box">
            <strong>🚨 LOW MODEL CONFIDENCE WARNING:</strong><br>
            {LOW_CONFIDENCE_WARNING_TEXT}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Top 3 Disease Predictions")

    # Plotly Bar Chart for Top 3 Predictions
    df = pd.DataFrame(top_3)
    df["confidence_pct"] = df["confidence"] * 100

    fig = px.bar(
        df,
        x="confidence_pct",
        y="disease",
        orientation="h",
        text=df["confidence_pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"confidence_pct": "Probability (%)", "disease": "Disease Category"},
        color="confidence_pct",
        color_continuous_scale=["#a5d6a7", "#4caf50", "#2e7d32"]
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
        height=240,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    st.plotly_chart(fig, use_container_width=True)
