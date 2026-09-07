"""MedXAI — 4_Model_Evaluation page (redesigned v2)."""
import sys, json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

CHART_COLORS = ["#2e7d32","#66bb6a","#a5d6a7","#1b5e20","#43a047"]
PLOT_LAYOUT  = dict(plot_bgcolor="rgba(255,255,255,0.0)", paper_bgcolor="rgba(255,255,255,0.0)",
                     font=dict(family="Inter, sans-serif", color="#3d5140"),
                     margin=dict(l=12,r=12,t=36,b=12))

st.set_page_config(page_title="MedXAI — Model Evaluation", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
apply_theme()
render_header()
render_sidebar()

cfg = get_config()
reports_dir = Path(cfg.paths.reports_dir)

st.markdown(f"""
<div style="margin-bottom:1.4rem;">
    {section_tag("Research Dashboard", "📊")}
    <h2 style="margin:4px 0 4px; font-size:1.6rem;">Model Evaluation</h2>
    <p style="color:#6b7f6d; font-size:0.9rem; margin:0;">
        Empirical test set metrics for BioBERT, BERT, and DistilBERT on the held-out Symptom2Disease test split.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top metric cards ──────────────────────────────────────────────────────────
ev_file = reports_dir / "test_evaluation_biobert.json"
if ev_file.exists():
    with open(ev_file) as f:
        ev = json.load(f)
    m1,m2,m3,m4 = st.columns(4)
    for col, (lbl, key, icon) in zip([m1,m2,m3,m4],[
        ("Accuracy",    "accuracy",         "🎯"),
        ("Macro F1",    "macro_f1",         "📐"),
        ("Precision",   "macro_precision",  "🔍"),
        ("Recall",      "macro_recall",     "📡"),
    ]):
        with col:
            val = ev.get(key, 0)
            st.markdown(f"""
            <div class="mxai-card" style="text-align:center;">
                <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
                <div class="mxai-card-title">{lbl}</div>
                <div class="mxai-card-value">{val*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Model comparison chart ────────────────────────────────────────────────────
comp_file = reports_dir / "model_comparison.csv"
if comp_file.exists():
    comp_df = pd.read_csv(comp_file)
    st.markdown(f"#### Model Comparison — Test Set")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    metrics_cols = [c for c in ["Accuracy","Precision","Recall","Macro F1","Weighted F1"] if c in comp_df.columns]
    if metrics_cols and "Model" in comp_df.columns:
        melted = comp_df.melt(id_vars=["Model"], value_vars=metrics_cols, var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                     text=melted["Score"].apply(lambda x: f"{x*100:.1f}%"),
                     color_discrete_sequence=CHART_COLORS)
        fig.update_layout(**PLOT_LAYOUT, title="", yaxis=dict(range=[0,1.05], title="Score", tickformat=".0%"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No model comparison report found. Run `python3 -m src.evaluation.evaluate --model primary` first.")

st.markdown("---")

# ── Per-class classification report ──────────────────────────────────────────
if ev_file.exists():
    rep = ev.get("classification_report", {})
    class_rows = {k: v for k, v in rep.items()
                  if isinstance(v, dict) and k not in ["accuracy","macro avg","weighted avg"]}
    if class_rows:
        st.markdown("#### Per-Class Metrics — BioBERT")
        df_rep = pd.DataFrame([
            {"Class": k, "Precision": f"{v['precision']*100:.1f}%",
             "Recall": f"{v['recall']*100:.1f}%",
             "F1": f"{v['f1-score']*100:.1f}%", "Support": int(v["support"])}
            for k, v in class_rows.items()
        ])
        st.dataframe(df_rep, use_container_width=True, hide_index=True)

        # Per-class F1 bar
        df_chart = pd.DataFrame([
            {"Class": k, "F1": round(v["f1-score"]*100, 1)} for k, v in class_rows.items()
        ]).sort_values("F1", ascending=True)
        fig2 = px.bar(df_chart, x="F1", y="Class", orientation="h",
                      color="F1", color_continuous_scale=["#c8e6c9","#2e7d32"],
                      text="F1")
        fig2.update_layout(**PLOT_LAYOUT, height=520,
                           xaxis=dict(range=[0,105], title="F1 Score (%)"),
                           coloraxis_showscale=False)
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

# ── Training curves ────────────────────────────────────────────────────────────
hist_path = reports_dir / "training_history_biobert.json"
if hist_path.exists():
    with open(hist_path) as f:
        hist = json.load(f)
    st.markdown("---")
    st.markdown("#### Training Curves — BioBERT")
    epochs     = hist.get("epochs", [])
    train_loss = hist.get("train_loss", [])
    val_loss   = hist.get("val_loss", [])
    val_acc    = hist.get("val_accuracy", [])
    val_f1     = hist.get("val_macro_f1", [])

    tc1, tc2 = st.columns(2)
    with tc1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines+markers",
                                   line=dict(color="#2e7d32", width=2), marker=dict(size=7)))
        fig3.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", mode="lines+markers",
                                   line=dict(color="#a5d6a7", width=2, dash="dash"), marker=dict(size=7)))
        fig3.update_layout(**PLOT_LAYOUT, title="Loss per Epoch",
                           xaxis=dict(title="Epoch", tickvals=epochs),
                           yaxis=dict(title="Cross-Entropy Loss"),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

    with tc2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=epochs, y=[v*100 for v in val_acc], name="Val Accuracy",
                                   mode="lines+markers", line=dict(color="#2e7d32", width=2),
                                   marker=dict(size=7)))
        fig4.add_trace(go.Scatter(x=epochs, y=[v*100 for v in val_f1], name="Val Macro F1",
                                   mode="lines+markers", line=dict(color="#66bb6a", width=2, dash="dash"),
                                   marker=dict(size=7)))
        fig4.update_layout(**PLOT_LAYOUT, title="Validation Metrics per Epoch",
                           xaxis=dict(title="Epoch", tickvals=epochs),
                           yaxis=dict(title="Score (%)", range=[0,105]),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig4, use_container_width=True)

render_footer()
