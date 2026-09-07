"""MedXAI Header component — redesigned v2."""
import streamlit as st


def render_header():
    """Renders the MedXAI glassmorphic header bar."""
    st.markdown("""
    <div class="mxai-header">
        <div class="mxai-brand">
            <div class="mxai-logo">🌿</div>
            <div>
                <div class="mxai-title">MedXAI</div>
                <div class="mxai-sub">Explainable Medical Text Classification</div>
            </div>
        </div>
        <div class="mxai-badge">🎓 Academic Prototype v1.0</div>
    </div>
    """, unsafe_allow_html=True)
