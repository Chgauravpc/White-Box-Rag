"""BRD Validator — validate a requirements document against the ingested knowledge base."""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import score_gauge
from lib.ui import trust_badge

st.title("📋 Requirement Validator")
st.caption("Upload a requirements document (BRD) and check each requirement's alignment against the ingested knowledge base.")


def _status(risk_level: str) -> str:
    rl = (risk_level or "").lower()
    if "low" in rl or "safe" in rl:
        return "COMPLIANT"
    if "medium" in rl or "review" in rl:
        return "REVIEW"
    return "VIOLATION"


_STATUS_COLOR = {"COMPLIANT": "#10b981", "REVIEW": "#f59e0b", "VIOLATION": "#ef4444"}

if "brd_requirements" not in st.session_state:
    st.session_state["brd_requirements"] = None
if "brd_results" not in st.session_state:
    st.session_state["brd_results"] = None
if "brd_filename" not in st.session_state:
    st.session_state["brd_filename"] = None

with st.sidebar:
    st.markdown("### Validation History")
    try:
        history = api_client.list_brd_runs()
    except ApiError:
        history = []
    if not history:
        st.caption("No past runs yet.")
    for run in history[:10]:
        label = f"#{run['id']} · {run.get('source_filename') or 'sample'} · {run.get('overall_score', 0):.0f}%"
        if st.button(label, key=f"hist_{run['id']}", width="stretch"):
            detail = api_client.get_brd_run(run["id"])
            st.session_state["brd_results"] = detail.get("results", [])
            st.session_state["brd_filename"] = detail.get("source_filename")
            st.rerun()

if not st.session_state["brd_results"]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload a Requirements Document")
        uploaded = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
        if uploaded and st.button("Parse Document", type="primary"):
            with st.spinner("Extracting requirements..."):
                try:
                    result = api_client.upload_brd(uploaded.name, uploaded.getvalue())
                    st.session_state["brd_requirements"] = result.get("data", [])
                    st.session_state["brd_filename"] = uploaded.name
                    st.rerun()
                except ApiError as e:
                    st.error(f"Upload failed: {e}")
    with col2:
        st.subheader("Or Try a Sample")
        if st.button("Load Sample BRD"):
            try:
                sample = api_client.get_sample_brd()
                txt = sample.get("data", {}).get("txt", "")
                reqs = [{"id": f"REQ-{i+1:03d}", "text": line.strip()} for i, line in enumerate(txt.splitlines()) if line.strip()]
                st.session_state["brd_requirements"] = reqs
                st.session_state["brd_filename"] = "sample_brd.txt"
                st.rerun()
            except ApiError as e:
                st.error(f"Could not load sample: {e}")

    if st.session_state["brd_requirements"]:
        st.divider()
        st.subheader(f"Extracted Requirements ({len(st.session_state['brd_requirements'])})")
        for r in st.session_state["brd_requirements"]:
            st.write(f"- **{r.get('id', '')}**: {r.get('text', '')}")

        if st.button("Validate Against Knowledge Base", type="primary"):
            texts = [r.get("text", "") for r in st.session_state["brd_requirements"] if r.get("text", "").strip()]
            with st.spinner("Mapping requirements to sources and scoring alignment..."):
                try:
                    result = api_client.validate_brd(texts, source_filename=st.session_state["brd_filename"])
                    st.session_state["brd_results"] = result.get("data", [])
                    st.rerun()
                except ApiError as e:
                    st.error(f"Validation failed: {e}")

else:
    results = st.session_state["brd_results"]
    statuses = [_status(r.get("risk_level", "")) for r in results]

    if st.button("← New Scan"):
        st.session_state["brd_requirements"] = None
        st.session_state["brd_results"] = None
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requirements", len(results))
    c2.metric("Compliant", statuses.count("COMPLIANT"))
    c3.metric("Needs Review", statuses.count("REVIEW"))
    c4.metric("Violations", statuses.count("VIOLATION"))

    scores = [r.get("overall_compliance_score", r.get("alignment_score", 0)) for r in results]
    overall = sum(scores) / len(scores) if scores else 0

    gcol, tcol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(score_gauge(overall, "Overall Compliance"), width="stretch")

    with tcol:
        st.markdown("#### Requirements")
        for i, (r, status) in enumerate(zip(results, statuses)):
            color = _STATUS_COLOR[status]
            with st.expander(f"REQ-{i+1:03d} · {status} · {r.get('requirement', '')[:70]}"):
                st.markdown(f'<span style="color:{color};font-weight:600;">{status}</span>', unsafe_allow_html=True)
                st.write(r.get("requirement", ""))
                if r.get("gaps"):
                    st.markdown("**Identified Gaps**")
                    for g in r["gaps"]:
                        st.markdown(f"- 🟠 {g}")
                if r.get("violations"):
                    st.markdown("**Compliance Violations**")
                    for v in r["violations"]:
                        st.markdown(f"- 🔴 {v}")
                if r.get("remediation_suggestions"):
                    st.markdown("**Recommended Fix**")
                    for s in r["remediation_suggestions"]:
                        st.markdown(f"- ✅ {s}")

    st.divider()
    st.markdown("#### Ask a follow-up question against the knowledge base")
    followup = st.text_input("Query", key="brd_followup_query")
    if st.button("Run Query", key="brd_followup_run") and followup.strip():
        with st.spinner("Running the RAG pipeline..."):
            try:
                report = api_client.query_rag(followup.strip())
                trust_badge(report.get("trust_gate", {}).get("status", "Unknown"), report.get("trust_gate", {}).get("overall_score"))
                for c in report.get("claims", []):
                    if c.get("retained", True):
                        st.markdown(f"- {c.get('text', '')}")
            except ApiError as e:
                st.error(f"Query failed: {e}")
