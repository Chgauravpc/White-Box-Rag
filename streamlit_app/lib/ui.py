"""
ui.py — shared visual language: trust-status colors, badges, KPI cards, CSS.

Colors verified from the original React app's frontend/src/index.css (not an
approximation) so the Streamlit rebuild matches exactly:
  Safe=#10b981, Needs_Human_Review=#f59e0b, Non_Compliant=#ef4444
"""

import streamlit as st

TRUST_COLORS = {
    "Safe": "#10b981",
    "Needs_Human_Review": "#f59e0b",
    "Non_Compliant": "#ef4444",
    "Unknown": "#6b7280",
}

TRUST_LABELS = {
    "Safe": "Safe",
    "Needs_Human_Review": "Needs Review",
    "Non_Compliant": "Non-Compliant",
    "Unknown": "Unknown",
}

TRUST_ICONS = {
    "Safe": "✅",
    "Needs_Human_Review": "⚠️",
    "Non_Compliant": "❌",
    "Unknown": "❓",
}


def trust_color(status: str) -> str:
    return TRUST_COLORS.get(status, TRUST_COLORS["Unknown"])


def trust_badge_html(status: str, score: float | None = None) -> str:
    color = trust_color(status)
    label = TRUST_LABELS.get(status, status or "Unknown")
    icon = TRUST_ICONS.get(status, TRUST_ICONS["Unknown"])
    score_text = f" &middot; {score * 100:.0f}%" if score is not None else ""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{color}22;color:{color};border:1px solid {color}55;'
        f'border-radius:999px;padding:4px 12px;font-weight:600;font-size:0.9rem;">'
        f'{icon} {label}{score_text}</span>'
    )


def trust_badge(status: str, score: float | None = None) -> None:
    st.markdown(trust_badge_html(status, score), unsafe_allow_html=True)


def score_pill_html(label: str, value: float, higher_is_better: bool = True, na: bool = False) -> str:
    if na:
        return (
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;color:#6b7280;">'
            f'<span>{label}</span><span>N/A</span></div>'
        )
    pct = value * 100
    if higher_is_better:
        color = "#10b981" if pct >= 80 else ("#f59e0b" if pct >= 50 else "#ef4444")
    else:
        color = "#10b981" if pct <= 20 else ("#f59e0b" if pct <= 50 else "#ef4444")
    return (
        f'<div style="margin:6px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;">'
        f'<span>{label}</span><span style="color:{color};font-weight:600;">{pct:.1f}%</span></div>'
        f'<div style="background:#e5e7eb22;border-radius:999px;height:6px;margin-top:3px;">'
        f'<div style="background:{color};width:{min(100, max(0, pct))}%;height:6px;border-radius:999px;"></div>'
        f'</div></div>'
    )


def score_pill(label: str, value: float, higher_is_better: bool = True, na: bool = False) -> None:
    st.markdown(score_pill_html(label, value, higher_is_better, na), unsafe_allow_html=True)


def kpi_card(label: str, value: str, color: str | None = None, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def integrity_chip_html(intact: bool, detail: str = "") -> str:
    """Pill for tamper-evident audit-chain status (reuses the trust-badge look)."""
    color = TRUST_COLORS["Safe"] if intact else TRUST_COLORS["Non_Compliant"]
    icon = "🔒" if intact else "⛓️‍💥"
    label = "Chain Verified" if intact else "Chain Broken"
    suffix = f" &middot; {detail}" if detail else ""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{color}22;color:{color};border:1px solid {color}55;'
        f'border-radius:999px;padding:4px 12px;font-weight:600;font-size:0.9rem;">'
        f'{icon} {label}{suffix}</span>'
    )


def integrity_chip(intact: bool, detail: str = "") -> None:
    st.markdown(integrity_chip_html(intact, detail), unsafe_allow_html=True)


def render_global_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_claim_card(claim: dict) -> None:
    """Render a single claim with its retained/stripped state and citation."""
    retained = claim.get("retained", True)
    reason = claim.get("filter_reason", "")
    text = claim.get("text", "")
    source = " &middot; ".join(
        filter(None, [claim.get("source_publication"), claim.get("source_edition"), claim.get("source_section_id")])
    )

    if not retained:
        st.markdown(
            f'<div style="opacity:0.55;text-decoration:line-through;padding:6px 0;">'
            f'{text}<br/><span style="font-size:0.8rem;color:#ef4444;">removed — {reason}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        flag = f' <span style="font-size:0.75rem;color:#f59e0b;">({reason})</span>' if reason else ""
        cite = f' <span style="font-size:0.8rem;color:#6b7280;">[{source}]</span>' if source else ""
        st.markdown(f'<div style="padding:6px 0;">&bull; {text}{cite}{flag}</div>', unsafe_allow_html=True)
