import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_confidence_chart(top_k_predictions):
    """Renders horizontal bar chart of Top-K prediction probabilities using Plotly."""
    df = pd.DataFrame(top_k_predictions)
    df["confidence_pct"] = df["confidence"] * 100

    fig = px.bar(
        df,
        x="confidence_pct",
        y="disease",
        orientation="h",
        text=df["confidence_pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"confidence_pct": "Confidence (%)", "disease": "Disease Category"},
        color="confidence_pct",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        title="Top-3 Disease Category Probabilities",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def render_token_highlight_html(token_scores, title="Token Feature Attribution Heatmap"):
    """Generates HTML snippet displaying tokens highlighted according to attribution score."""
    if not token_scores:
        return "<p>No attributions available.</p>"

    scores = [s for _, s in token_scores]
    max_s = max(abs(min(scores)), abs(max(scores))) if scores else 1.0
    if max_s == 0:
        max_s = 1.0

    html_spans = []
    for token, score in token_scores:
        norm_score = score / max_s
        if norm_score > 0:
            # Positive contribution (Red/Orange highlight)
            alpha = min(abs(norm_score), 1.0) * 0.75
            bg_color = f"rgba(255, 99, 71, {alpha:.2f})"
        else:
            # Negative contribution (Blue highlight)
            alpha = min(abs(norm_score), 1.0) * 0.75
            bg_color = f"rgba(70, 130, 180, {alpha:.2f})"

        span = f'<span style="background-color: {bg_color}; padding: 3px 6px; margin: 2px; border-radius: 4px; display: inline-block; font-weight: 500;" title="Score: {score:.4f}">{token}</span>'
        html_spans.append(span)

    full_html = f"""
    <div style="background-color: #1e1e2f; color: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #33334d; font-family: monospace; line-height: 2.0;">
        <h5 style="margin-top: 0; color: #64b5f6;">{title}</h5>
        <div>{" ".join(html_spans)}</div>
        <div style="font-size: 11px; margin-top: 12px; color: #aaa;">
            <span style="background-color: rgba(255, 99, 71, 0.6); padding: 2px 6px; border-radius: 3px;">Red</span> = Increases Disease Probability | 
            <span style="background-color: rgba(70, 130, 180, 0.6); padding: 2px 6px; border-radius: 3px;">Blue</span> = Decreases Probability
        </div>
    </div>
    """
    return full_html

def render_xai_comparison_chart(summary_df):
    """Renders grouped bar chart comparing Attention, SHAP, LIME, and Integrated Gradients."""
    if summary_df.empty:
        return None

    words = summary_df["Word"].tolist()
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name="Attention", x=words, y=summary_df["Attention Score"], marker_color="#42a5f5"))
    fig.add_trace(go.Bar(name="SHAP", x=words, y=summary_df["SHAP Score"], marker_color="#ff7043"))
    fig.add_trace(go.Bar(name="LIME", x=words, y=summary_df["LIME Score"], marker_color="#66bb6a"))
    fig.add_trace(go.Bar(name="Integrated Gradients", x=words, y=summary_df["Integrated Gradients"], marker_color="#ab47bc"))

    fig.update_layout(
        title="Multi-Method Feature Importance Comparison per Word",
        barmode="group",
        xaxis_title="Tokens / Words",
        yaxis_title="Normalized Importance Score",
        height=400,
        margin=dict(l=20, r=20, t=40, b=40)
    )
    return fig
