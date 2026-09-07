"""MedXAI Cards component — reusable card HTML helpers."""
import streamlit as st
from typing import Optional


def metric_card(label: str, value: str, sub: str = "", icon: str = "") -> str:
    """Returns HTML for a white metric card."""
    return f"""
    <div class="mxai-card" style="text-align:center;">
        <div style="font-size:1.5rem; margin-bottom:6px;">{icon}</div>
        <div class="mxai-card-title">{label}</div>
        <div class="mxai-card-value">{value}</div>
        {"<div style='font-size:0.78rem;color:#6b7f6d;margin-top:4px;'>"+sub+"</div>" if sub else ""}
    </div>
    """


def feature_card(icon: str, title: str, desc: str) -> str:
    """Returns HTML for a feature highlight card (Home page)."""
    return f"""
    <div class="mxai-feature">
        <span class="mxai-feature-icon">{icon}</span>
        <div class="mxai-feature-title">{title}</div>
        <div class="mxai-feature-desc">{desc}</div>
    </div>
    """


def section_tag(text: str, icon: str = "") -> str:
    """Returns a small green pill label for section headers."""
    return f'<div class="mxai-section-tag">{icon + " " if icon else ""}{text}</div>'


def render_probability_bars(predictions: list):
    """Render top-N disease probability bars using clean HTML."""
    bars_html = ""
    for pred in predictions:
        pct = pred["confidence"] * 100
        bars_html += f"""
        <div class="mxai-prob-row">
            <div class="mxai-prob-label">{pred["disease"]}</div>
            <div class="mxai-prob-track">
                <div class="mxai-prob-fill" style="width:{min(pct, 100):.1f}%;"></div>
            </div>
            <div class="mxai-prob-pct">{pct:.1f}%</div>
        </div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)


def render_prediction_card(pred_result: dict, confidence_threshold: float = 0.70):
    """Render the main disease prediction result card."""
    if pred_result.get("status") != "success":
        st.error(f"Prediction failed: {pred_result.get('error', 'Unknown error')}")
        return

    top = pred_result["top_3_predictions"][0]
    disease   = top["disease"]
    conf      = top["confidence"]
    conf_cat  = pred_result.get("confidence_category", "")
    conf_pct  = conf * 100

    if conf >= confidence_threshold:
        badge = f'<span class="mxai-conf-high">● {conf_pct:.1f}%&nbsp;·&nbsp;{conf_cat}</span>'
    elif conf >= confidence_threshold * 0.75:
        badge = f'<span class="mxai-conf-med">◐ {conf_pct:.1f}%&nbsp;·&nbsp;{conf_cat}</span>'
    else:
        badge = f'<span class="mxai-conf-low">○ {conf_pct:.1f}%&nbsp;·&nbsp;{conf_cat}</span>'

    st.markdown(f"""
    <div class="mxai-prediction">
        <div class="mxai-pred-label">🔬 Predicted Condition</div>
        <div class="mxai-pred-disease">{disease}</div>
        {badge}
    </div>
    """, unsafe_allow_html=True)

    if conf < confidence_threshold:
        st.markdown("""
        <div class="mxai-warning">
            ⚠️ <strong>Low model confidence.</strong> The prediction may be unreliable.
            Review alternative predictions below and consult a qualified healthcare professional.
            This tool does not provide clinical diagnoses or treatment recommendations.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.2rem;">
        {section_tag("Top Predictions", "📊")}
    </div>
    """, unsafe_allow_html=True)
    render_probability_bars(pred_result["top_3_predictions"])
