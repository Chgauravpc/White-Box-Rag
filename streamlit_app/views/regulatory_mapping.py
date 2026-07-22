"""Regulatory Mapping — how this system maps to EU AI Act / NIST AI RMF controls."""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import score_gauge

st.title("⚖️ Regulatory Mapping")
st.caption(
    "How White Box RAG's capabilities map to real AI-governance controls. Status is honest: "
    "**satisfied** where a feature fully addresses a control, **partial** where it addresses "
    "part of the obligation. Static reference — no per-query cost."
)

_STATUS_STYLE = {
    "satisfied": ("#10b981", "✅ Satisfied"),
    "partial": ("#f59e0b", "🟡 Partial"),
    "planned": ("#6b7280", "⚪ Planned"),
}


def _status_chip(status: str) -> str:
    color, label = _STATUS_STYLE.get(status, ("#6b7280", status))
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:600;">{label}</span>'
    )


try:
    resp = api_client.get_frameworks()
    frameworks = resp.get("data", []) if isinstance(resp, dict) else []
except ApiError as e:
    frameworks = []
    st.error(f"Could not load framework mapping: {e}")

for fw in frameworks:
    st.divider()
    header, gauge = st.columns([2, 1])
    with header:
        st.subheader(fw["framework"])
        st.caption(fw.get("reference", ""))
        counts = fw.get("status_counts", {})
        st.markdown(
            f"{counts.get('satisfied', 0)} satisfied &middot; {counts.get('partial', 0)} partial "
            f"&middot; {counts.get('planned', 0)} planned &nbsp;·&nbsp; {fw.get('num_controls', 0)} controls",
            unsafe_allow_html=True,
        )
    with gauge:
        st.plotly_chart(
            score_gauge(round(fw.get("coverage", 0.0) * 100, 1), title="Coverage"),
            width="stretch",
        )

    for c in fw.get("controls", []):
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**{c['control_id']} — {c['title']}**")
                st.caption(c.get("requirement", ""))
            with top[1]:
                st.markdown(_status_chip(c.get("status", "planned")), unsafe_allow_html=True)

            st.markdown("**Satisfied by:** " + ", ".join(c.get("satisfied_by", [])))
            for ev in c.get("evidence", []):
                st.markdown(f"- {ev}")
            page = c.get("page")
            if page:
                if st.button("View evidence →", key=f"ev_{fw['framework']}_{c['control_id']}"):
                    st.switch_page(page)

if not frameworks:
    st.info("No framework data available.")
