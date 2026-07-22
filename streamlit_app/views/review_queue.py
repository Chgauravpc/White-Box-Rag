"""Review Queue — audits flagged Needs_Human_Review awaiting human resolution."""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.ui import trust_badge_html

st.title("🧑‍⚖️ Review Queue")
st.caption(
    "Audits the Trust Gate flagged **Needs Human Review**. Resolving one appends a "
    "tamper-evident, hash-chained decision — the original audit record is never altered."
)

try:
    resp = api_client.list_review_queue()
    pending = resp.get("data", []) if isinstance(resp, dict) else []
except ApiError as e:
    pending = []
    st.error(f"Could not load the review queue: {e}")

st.metric("Pending human review", len(pending))
st.divider()

if not pending:
    st.success("Nothing awaiting review. 🎉")
else:
    for item in pending:
        c1, c2, c3, c4 = st.columns([1, 4, 2, 1])
        c1.write(f"`IDX-{item['id']}`")
        c2.write((item.get("query") or "")[:70])
        with c3:
            st.markdown(trust_badge_html(item.get("trust_gate_status", "Needs_Human_Review")), unsafe_allow_html=True)
        if c4.button("Review", key=f"review_{item['id']}"):
            # Resolution controls live on the audit detail page (full context there).
            st.query_params["audit_id"] = str(item["id"])
            st.switch_page("views/audit_trail.py")
