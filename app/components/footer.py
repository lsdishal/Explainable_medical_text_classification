"""MedXAI Footer component — redesigned v2."""
import streamlit as st


def render_footer():
    """Renders the academic disclaimer footer."""
    st.markdown("""
    <div class="mxai-footer">
        <strong>⚕️ Academic &amp; Research Disclaimer</strong><br>
        This application is an AI-assisted academic text classification prototype.
        It is <strong>not</strong> a clinical diagnostic tool, does not prescribe treatment, and does not replace 
        evaluation by a qualified healthcare professional. All predictions must be reviewed by a licensed clinician.
    </div>
    """, unsafe_allow_html=True)
