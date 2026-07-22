"""Dashboard — system overview: ingested collections, audit trail summary, trust distribution."""

import streamlit as st
import pandas as pd

from lib import api_client
from lib.api_client import ApiError
from lib.charts import trust_distribution_donut
from lib.ui import trust_badge_html

st.title("📊 Dashboard")
st.caption("Overview of ingested knowledge, query volume, and system-wide trust distribution.")

try:
    documents = api_client.list_documents()
except ApiError as e:
    documents = []
    st.error(f"Could not load documents: {e}")

try:
    audit_logs = api_client.list_audit_logs()
except ApiError as e:
    audit_logs = []
    st.error(f"Could not load audit logs: {e}")

try:
    _queue = api_client.list_review_queue()
    pending_review = len(_queue.get("data", [])) if isinstance(_queue, dict) else 0
except ApiError:
    pending_review = 0

total_chunks = sum(d.get("chunk_count", 0) for d in documents)
total_queries = len(audit_logs)


def _is_status(log: dict, needle: str) -> bool:
    return needle.lower() in str(log.get("risk_level", "")).lower()


safe_count = sum(1 for l in audit_logs if _is_status(l, "safe"))
review_count = sum(1 for l in audit_logs if _is_status(l, "review"))
noncompliant_count = sum(1 for l in audit_logs if _is_status(l, "non_compliant") or _is_status(l, "noncompliant"))
risk_flags = review_count + noncompliant_count
safety_pct = round(100 * safe_count / total_queries, 1) if total_queries else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Collections Ingested", len(documents), help=f"{total_chunks} chunks total")
with col2:
    st.metric("Queries Logged", total_queries)
with col3:
    st.metric("Risk Flags", risk_flags, help="Needs-review + non-compliant audit logs")
with col4:
    st.metric("Pending Review", pending_review, help="Flagged audits awaiting human resolution")
with col5:
    st.metric("Safety Rate", f"{safety_pct}%", help="Fraction of logged queries with a Safe trust status")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Recent Audits")
    if not audit_logs:
        st.info("No queries logged yet — try the RAG Query page.")
    else:
        recent = audit_logs[:8]
        for log in recent:
            c1, c2, c3 = st.columns([5, 2, 1])
            with c1:
                st.write(f"**{(log.get('query') or '')[:70]}**")
            with c2:
                st.markdown(trust_badge_html(log.get("risk_level", "Unknown")), unsafe_allow_html=True)
            with c3:
                if st.button("View", key=f"view_{log['id']}"):
                    st.query_params["audit_id"] = str(log["id"])
                    st.switch_page("views/audit_trail.py")

with right:
    st.subheader("Trust Distribution")
    if total_queries:
        fig = trust_distribution_donut({
            "Safe": safe_count,
            "Needs_Human_Review": review_count,
            "Non_Compliant": noncompliant_count,
        })
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data yet.")

st.divider()
st.subheader("Ingested Collections")
if documents:
    df = pd.DataFrame(documents)[["publication_name", "edition_date", "chunk_count", "structured", "ingested_at"]]
    df.columns = ["Collection", "Version", "Chunks", "Structured", "Ingested At"]
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No documents ingested yet — head to the Ingest Documents page.")
