"""MedXAI — 5_Dataset_Analysis page (redesigned v2)."""
import sys, json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.theme import apply_theme
from app.components.header import render_header
from app.components.sidebar import render_sidebar
from app.components.cards import section_tag
from app.components.footer import render_footer
from src.utils.config import get_config

PLOT_LAYOUT = dict(plot_bgcolor="rgba(255,255,255,0.0)", paper_bgcolor="rgba(255,255,255,0.0)",
                    font=dict(family="Inter, sans-serif", color="#3d5140"),
                    margin=dict(l=12,r=12,t=36,b=12))

st.set_page_config(page_title="MedXAI — Dataset Analysis", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
apply_theme()
render_header()
render_sidebar()

cfg = get_config()
stats_file = Path(cfg.dataset.processed_dir) / "dataset_statistics.json"

st.markdown(f"""
<div style="margin-bottom:1.4rem;">
    {section_tag("Data Explorer", "📈")}
    <h2 style="margin:4px 0 4px; font-size:1.6rem;">Dataset Analysis</h2>
    <p style="color:#6b7f6d; font-size:0.9rem; margin:0;">
        Exploratory analysis of the Symptom2Disease dataset — class distributions, text statistics, and split sizes.
    </p>
</div>
""", unsafe_allow_html=True)

if not stats_file.exists():
    st.warning("Dataset statistics not found. Run `python3 -m src.preprocessing.dataset_preparation` first.")
    render_footer()
    st.stop()

with open(stats_file) as f:
    stats = json.load(f)

# ── Overview metrics ──────────────────────────────────────────────────────────
m1,m2,m3,m4,m5 = st.columns(5)
sp = stats.get("splits", {})
for col, (lbl, val, icon) in zip([m1,m2,m3,m4,m5],[
    ("Total Samples",   f"{stats.get('total_cleaned_samples',0):,}", "🗂️"),
    ("Disease Classes", str(stats.get("num_classes", 0)),            "🏷️"),
    ("Train Samples",   f"{sp.get('train_samples',0):,}",            "🚂"),
    ("Val Samples",     f"{sp.get('val_samples',0):,}",              "🔎"),
    ("Test Samples",    f"{sp.get('test_samples',0):,}",             "🧪"),
]):
    with col:
        st.markdown(f"""
        <div class="mxai-card" style="text-align:center;">
            <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
            <div class="mxai-card-title">{lbl}</div>
            <div class="mxai-card-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Class distribution chart ──────────────────────────────────────────────────
st.markdown("#### Disease Category Distribution")
cls_counts = stats.get("samples_per_class", {})
if cls_counts:
    df_cls = pd.DataFrame(list(cls_counts.items()), columns=["Disease", "Samples"])
    df_cls = df_cls.sort_values("Samples", ascending=True)
    fig = px.bar(df_cls, x="Samples", y="Disease", orientation="h",
                 text="Samples", color="Samples",
                 color_continuous_scale=["#c8e6c9","#43a047","#1b5e20"])
    fig.update_layout(**PLOT_LAYOUT, height=560,
                      yaxis=dict(autorange="reversed", title=""),
                      xaxis=dict(title="Sample Count"),
                      coloraxis_showscale=False)
    fig.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Text stats ─────────────────────────────────────────────────────────────────
txt = stats.get("text_statistics", {})
if txt:
    st.markdown("#### Text Length Statistics")
    ts1, ts2 = st.columns(2)
    with ts1:
        wc = txt.get("word_count", {})
        st.markdown(f"""
        <div class="mxai-card">
            <div class="mxai-card-title">Word Count per Sample</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                <div class="mxai-stat-row"><span class="mxai-stat-label">Min</span><span class="mxai-stat-value">{wc.get('min',0)}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Max</span><span class="mxai-stat-value">{wc.get('max',0)}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Mean</span><span class="mxai-stat-value">{wc.get('mean',0):.1f}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Median</span><span class="mxai-stat-value">{wc.get('median',0):.1f}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Std Dev</span><span class="mxai-stat-value">{wc.get('std',0):.1f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ts2:
        cl = txt.get("char_length", {})
        st.markdown(f"""
        <div class="mxai-card">
            <div class="mxai-card-title">Character Length per Sample</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                <div class="mxai-stat-row"><span class="mxai-stat-label">Min</span><span class="mxai-stat-value">{cl.get('min',0)}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Max</span><span class="mxai-stat-value">{cl.get('max',0)}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Mean</span><span class="mxai-stat-value">{cl.get('mean',0):.1f}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Median</span><span class="mxai-stat-value">{cl.get('median',0):.1f}</span></div>
                <div class="mxai-stat-row"><span class="mxai-stat-label">Std Dev</span><span class="mxai-stat-value">{cl.get('std',0):.1f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

render_footer()
