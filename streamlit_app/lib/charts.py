"""
charts.py — Plotly figure builders: RAGAS-style radar, trust-distribution
donut, compliance/gauge indicator. Chosen over Altair because Vega-Lite has
no native radar or gauge chart type; Plotly ships all three directly.
"""

import plotly.graph_objects as go

from lib.ui import TRUST_COLORS

SCORECARD_METRICS = [
    ("faithfulness_post", "Faithfulness (post)"),
    ("context_relevance", "Context Relevance"),
    ("context_diversity", "Context Diversity"),
    ("citation_precision", "Citation Precision"),
    ("answer_relevancy", "Answer Relevancy"),
    ("context_utilization", "Context Utilization"),
    ("paraphrase_stability", "Paraphrase Stability"),
]


def scorecard_radar(scorecard: dict, faithfulness_post: float | None = None, title: str = "Trust Scorecard") -> go.Figure:
    """8-spoke radar: the 7 pure-math scorecard metrics (faithfulness swapped
    for the post-mitigation value when available, since that's what matters
    to the end user — the raw pre-filter number is shown separately)."""
    merged = dict(scorecard)
    if faithfulness_post is not None:
        merged["faithfulness_post"] = faithfulness_post
    elif "faithfulness_post" not in merged:
        merged["faithfulness_post"] = merged.get("faithfulness", 0.0)

    labels = [label for key, label in SCORECARD_METRICS]
    values = [float(merged.get(key, 0.0) or 0.0) for key, label in SCORECARD_METRICS]
    # close the loop
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name=title,
        line_color="#6366f1",
        fillcolor="rgba(99,102,241,0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
    )
    return fig


def trust_distribution_donut(counts: dict) -> go.Figure:
    """counts: {"Safe": n, "Needs_Human_Review": n, "Non_Compliant": n}"""
    labels = ["Safe", "Needs_Human_Review", "Non_Compliant"]
    values = [counts.get(l, 0) for l in labels]
    colors = [TRUST_COLORS[l] for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=["Safe", "Needs Review", "Non-Compliant"],
        values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo="value",
    )])
    total = sum(values)
    fig.update_layout(
        annotations=[dict(text=str(total), x=0.5, y=0.5, font_size=28, showarrow=False)],
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def score_gauge(value_pct: float, title: str = "Compliance Score") -> go.Figure:
    """value_pct: 0-100 scale."""
    color = "#10b981" if value_pct >= 70 else ("#f59e0b" if value_pct >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            # rgba (not 8-digit hex) — plotly gauge steps reject #rrggbbaa.
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.13)"},
                {"range": [50, 70], "color": "rgba(245,158,11,0.13)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.13)"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def eval_trend_line(runs: list[dict], metric_key: str, title: str, threshold: float | None = None,
                     threshold_label: str = "target") -> go.Figure:
    """runs: list of {"started_at": str, "metrics": {...}} sorted oldest->newest."""
    x = [r.get("started_at", r.get("run_label", str(i))) for i, r in enumerate(runs)]
    y = [(r.get("metrics") or {}).get(metric_key) for r in runs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=title, line=dict(color="#6366f1")))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text=threshold_label, annotation_position="top left")
    fig.update_layout(title=title, height=280, margin=dict(l=30, r=20, t=50, b=30))
    return fig
