"""
MedXAI Theme Loader — inject the unified CSS design system into every page.
Import and call apply_theme() at the top of every page file.
"""
import streamlit as st
from pathlib import Path


def apply_theme():
    """Inject the MedXAI unified design system CSS into the current page."""
    css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Minimal inline fallback
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background: #f7fbf7; }
        .block-container { max-width: 1180px !important; margin: 0 auto !important; }
        </style>
        """, unsafe_allow_html=True)
