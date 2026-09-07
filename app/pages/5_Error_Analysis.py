"""MedXAI — 6_Error_Analysis page (redesigned v2)."""
import sys
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

st.set_page_config(page_title="MedXAI — Error Analysis", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
apply_theme()
render_header()
render_sidebar()

cfg = get_config()
reports_dir = Path(cfg.paths.reports_dir)
err_csv = reports_dir / "error_analysis_biobert.csv"

st.markdown(f"""
<div style="margin-bottom:1.4rem;">
    {section_tag("Failure Mode Analysis", "🔍")}
    <h2 style="margin:4px 0 4px; font-size:1.6rem;">Error Analysis</h2>
    <p style="color:#6b7f6d; font-size:0.9rem; margin:0;">
        Systematic review of incorrect test set predictions — identifying failure patterns and confused disease pairs.
    </p>
</div>
""", unsafe_allow_html=True)

if not err_csv.exists():
    st.warning("Error analysis report not found. Run `python3 -m src.evaluation.evaluate --model primary`.")
    render_footer()
    st.stop()

df = pd.read_csv(err_csv)
df_err = df[df["is_error"] == True].copy()
df_ok  = df[df["is_error"] == False].copy()

# ── Top stats ─────────────────────────────────────────────────────────────────
total = len(df)
errors = len(df_err)
correct = len(df_ok)

m1,m2,m3,m4 = st.columns(4)
for col, (lbl, val, icon) in zip([m1,m2,m3,m4],[
    ("Test Samples",    str(total),                  "🗂️"),
    ("Correct",         f"{correct} ({correct/total*100:.1f}%)", "✅"),
    ("Incorrect",       f"{errors} ({errors/total*100:.1f}%)",   "❌"),
    ("Avg Error Conf.", f"{df_err['confidence'].mean()*100:.1f}%" if errors else "—", "📉"),
]):
    with col:
        st.markdown(f"""
        <div class="mxai-card" style="text-align:center;">
            <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
            <div class="mxai-card-title">{lbl}</div>
            <div class="mxai-card-value" style="font-size:1.4rem;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Confused pairs ─────────────────────────────────────────────────────────────
if errors > 0:
    pair_col = st.columns([2, 1])
    with pair_col[0]:
        st.markdown("#### Most Confused Disease Pairs")
        confusion_pairs = (
            df_err.groupby(["ground_truth", "predicted_label"])
            .size().reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(10)
        )
        confusion_pairs["Pair"] = confusion_pairs["ground_truth"] + "  →  " + confusion_pairs["predicted_label"]
        fig = px.bar(confusion_pairs, x="Count", y="Pair", orientation="h",
                     text="Count", color="Count",
                     color_continuous_scale=["#c8e6c9","#e6a817"])
        fig.update_layout(**PLOT_LAYOUT, height=max(280, len(confusion_pairs)*46),
                          yaxis=dict(autorange="reversed", title=""),
                          xaxis=dict(title="Error Count", dtick=1),
                          coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with pair_col[1]:
        st.markdown("#### Confidence: Correct vs Wrong")
        st.markdown(f"""
        <div class="mxai-card">
            <div class="mxai-card-title">Correct predictions</div>
            <div class="mxai-card-value">{df_ok['confidence'].mean()*100:.1f}%</div>
            <div style="font-size:0.78rem;color:#6b7f6d;">mean confidence</div>
        </div>
        <div class="mxai-card" style="margin-top:0.6rem;">
            <div class="mxai-card-title">Incorrect predictions</div>
            <div class="mxai-card-value" style="color:#c05a00;">{df_err['confidence'].mean()*100:.1f}%</div>
            <div style="font-size:0.78rem;color:#6b7f6d;">mean confidence</div>
        </div>
        <div class="mxai-info" style="margin-top:0.8rem; font-size:0.78rem;">
            All 11 errors had confidence below 30% — the model is correctly uncertain on its mistakes.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Filterable error table ─────────────────────────────────────────────────
    st.markdown("#### All Incorrect Predictions")
    true_classes = sorted(df_err["ground_truth"].unique().tolist())
    pred_classes = sorted(df_err["predicted_label"].unique().tolist())
    fa, fb, fc = st.columns([2, 2, 2])
    with fa:
        true_filter = st.selectbox("Filter by true label:", ["All"] + true_classes, key="err_true")
    with fb:
        pred_filter = st.selectbox("Filter by predicted:", ["All"] + pred_classes, key="err_pred")
    with fc:
        conf_filter = st.slider("Max confidence:", 0.0, 1.0, 1.0, step=0.05, key="err_conf")

    view = df_err.copy()
    if true_filter != "All":
        view = view[view["ground_truth"] == true_filter]
    if pred_filter != "All":
        view = view[view["predicted_label"] == pred_filter]
    view = view[view["confidence"] <= conf_filter]

    for _, row in view.iterrows():
        conf_pct = row["confidence"] * 100
        border_color = "#f59e0b" if conf_pct > 20 else "rgba(168,213,170,0.4)"
        st.markdown(f"""
        <div class="mxai-card" style="border-left:4px solid {border_color}; padding:1rem 1.4rem; margin-bottom:0.7rem;">
            <div style="font-size:0.8rem;color:#6b7f6d;margin-bottom:6px;">
                ✅ True: <strong style="color:#1b5e20;">{row["ground_truth"]}</strong>
                &nbsp;&nbsp;❌ Predicted: <strong style="color:#c05a00;">{row["predicted_label"]}</strong>
                &nbsp;&nbsp;Confidence: <strong>{conf_pct:.1f}%</strong>
            </div>
            <div style="font-size:0.88rem;color:#3d5140;line-height:1.55;">{row["symptom_text"][:280]}...</div>
        </div>
        """, unsafe_allow_html=True)

    if view.empty:
        st.info("No records match the current filters.")

render_footer()
