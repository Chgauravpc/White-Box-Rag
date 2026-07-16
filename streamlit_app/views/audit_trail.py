"""Audit Trail — list of all logged queries, with a full-detail drill-down view."""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import scorecard_radar
from lib.ui import trust_badge, render_claim_card

st.title("🗂️ Audit Trail")

audit_id = st.query_params.get("audit_id")

if audit_id:
    if st.button("← Back to list"):
        del st.query_params["audit_id"]
        st.rerun()

    try:
        report = api_client.get_audit_report(int(audit_id))
    except ApiError as e:
        report = None
        st.error(f"Could not load audit report {audit_id}: {e}")

    if report:
        trust_gate = report.get("trust_gate") or {}
        scorecard = report.get("scorecard") or {}
        claims = report.get("claims", [])
        edition_conflicts = report.get("edition_conflicts", [])
        latency = report.get("latency_ms", {})

        st.subheader(f"Audit #{report.get('id')}")
        c1, c2 = st.columns([3, 1])
        with c1:
            trust_badge(trust_gate.get("status", "Unknown"), trust_gate.get("overall_score"))
            st.caption(report.get("timestamp", ""))
        with c2:
            st.download_button(
                "⬇ Download Audit (JSON)",
                data=api_client.download_audit(report["id"]) if report.get("id") else b"{}",
                file_name=f"audit_report_{report.get('id')}.json",
                mime="application/json",
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Claims Verified", len(claims))
        m2.metric("Total Latency", f"{sum(latency.values()):.0f}ms" if latency else "N/A")
        m3.metric("Gemini Calls", report.get("gemini_call_count", "N/A"))

        st.divider()
        st.markdown("#### Query")
        st.write(f"*{report.get('query', '')}*")
        st.markdown("#### System Response (raw)")
        st.write(report.get("response", ""))

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Decision Proof Registry")
            for c in claims:
                render_claim_card(c)
                with st.expander("Source passage & verification detail", expanded=False):
                    st.caption(c.get("source_passage", ""))
                    matching = [v for v in report.get("verifications", []) if v.get("claim_text") == c.get("text")]
                    if matching:
                        st.write(matching[0].get("explanation", ""))

        with right:
            st.markdown("#### Trust Score Matrix")
            if scorecard:
                fig = scorecard_radar(scorecard, faithfulness_post=report.get("faithfulness_post"))
                st.plotly_chart(fig, width="stretch")

            if report.get("final_audit_summary"):
                summary = report["final_audit_summary"]
                st.markdown("#### Auditor Summary")
                st.info(summary.get("reasoning", ""))
                if summary.get("key_risks"):
                    st.markdown("**Safety Warnings**")
                    for risk in summary["key_risks"]:
                        st.markdown(f"- ⚠️ {risk}")

        if edition_conflicts:
            st.warning(f"**Edition Conflicts** ({len(edition_conflicts)})")
            for c in edition_conflicts:
                st.caption(f"{c.get('conflict_description')} — superseding: {c.get('superseding_edition')}")

else:
    try:
        logs = api_client.list_audit_logs()
    except ApiError as e:
        logs = []
        st.error(f"Could not load audit logs: {e}")

    def _is(log, needle):
        return needle.lower() in str(log.get("risk_level", "")).lower()

    safe_n = sum(1 for l in logs if _is(l, "safe"))
    review_n = sum(1 for l in logs if _is(l, "review"))
    noncompliant_n = sum(1 for l in logs if _is(l, "non_compliant") or _is(l, "noncompliant"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Logs", len(logs))
    c2.metric("Safe", safe_n)
    c3.metric("Under Review", review_n)
    c4.metric("Non-Compliant", noncompliant_n)

    search = st.text_input("🔍 Search by query text or ID")

    filtered = logs
    if search:
        s = search.lower()
        filtered = [l for l in logs if s in str(l.get("query", "")).lower() or s in str(l.get("id", ""))]

    st.divider()
    if not filtered:
        st.info("No audit logs found.")
    for log in filtered:
        c1, c2, c3, c4 = st.columns([1, 4, 2, 1])
        c1.write(f"`IDX-{log['id']}`")
        c2.write((log.get("query") or "")[:60])
        from lib.ui import trust_badge_html
        with c3:
            st.markdown(trust_badge_html(log.get("risk_level", "Unknown")), unsafe_allow_html=True)
        if c4.button("View", key=f"list_view_{log['id']}"):
            st.query_params["audit_id"] = str(log["id"])
            st.rerun()
