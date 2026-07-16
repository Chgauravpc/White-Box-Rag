"""Trust Gating sandbox — evaluate arbitrary claims, or check two edition texts for conflicts."""

import pandas as pd
import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.ui import trust_badge, score_pill

st.title("🛡️ Trust Gating")
st.caption("A sandbox for the verification engine — no retrieval step, so context metrics show as N/A.")

tab1, tab2 = st.tabs(["Evaluate Claims", "Cross-Edition Check"])

with tab1:
    st.markdown("Provide an AI-generated answer and its claims; run the same NLI + Trust Gate pipeline used in production.")

    answer_text = st.text_area("AI Answer Text", height=100, placeholder="The full generated answer...")

    if "sandbox_claims" not in st.session_state:
        st.session_state["sandbox_claims"] = pd.DataFrame([
            {"text": "", "source_publication": "", "source_edition": "", "source_section_id": "", "source_passage": ""},
        ])

    edited = st.data_editor(
        st.session_state["sandbox_claims"],
        num_rows="dynamic",
        width="stretch",
        column_config={
            "text": st.column_config.TextColumn("Claim text", width="large"),
            "source_publication": st.column_config.TextColumn("Publication"),
            "source_edition": st.column_config.TextColumn("Edition"),
            "source_section_id": st.column_config.TextColumn("Section"),
            "source_passage": st.column_config.TextColumn("Source passage", width="large"),
        },
        key="claims_editor",
    )

    if st.button("Run Trust Gate Validation", type="primary"):
        claims = [row.to_dict() for _, row in edited.iterrows() if row.get("text", "").strip()]
        if not claims or not answer_text.strip():
            st.warning("Provide an answer and at least one claim.")
        else:
            with st.spinner("Verifying claims..."):
                try:
                    result = api_client.verify_claims(answer_text, claims)
                except ApiError as e:
                    result = None
                    st.error(f"Verification failed: {e}")

            if result:
                gate = result.get("trust_gate", {})
                scorecard = result.get("scorecard", {})
                st.divider()
                trust_badge(gate.get("status", "Unknown"), gate.get("overall_score"))
                st.write(gate.get("reasoning", ""))

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("#### Per-claim verdicts")
                    for v in result.get("verifications", []):
                        ok = v.get("verdict") in ("SUPPORTED", "ENTAILMENT")
                        icon = "✅" if ok else ("⚠️" if v.get("verdict") in ("NEUTRAL", "NOT_ENOUGH_INFO") else "❌")
                        st.markdown(f"{icon} **{v.get('claim_text')}** — `{v.get('verdict')}` ({v.get('entailment_score', 0):.2f})")
                        st.caption(v.get("explanation", ""))
                with col2:
                    st.markdown("#### Scorecard")
                    if scorecard:
                        score_pill("Faithfulness", scorecard.get("faithfulness", 0))
                        score_pill("Citation Precision", scorecard.get("citation_precision", 0))
                        score_pill("Context Relevance", 0, na=True)
                        score_pill("Context Diversity", 0, na=True)
                        st.caption("Context metrics are N/A in sandbox mode — no retrieval step here.")

with tab2:
    st.markdown("Compare an older and newer edition of a section to detect regulatory/edition drift.")

    with st.form("conflict_form"):
        c1, c2 = st.columns(2)
        publication = c1.text_input("Publication / Collection", placeholder="e.g. FSR")
        topic = c2.text_input("Topic", placeholder="e.g. Capital Adequacy")
        section_id = st.text_input("Section ID", placeholder="e.g. 4.2.3")

        c3, c4 = st.columns(2)
        with c3:
            older_date = st.text_input("Older Edition Label", placeholder="e.g. June 2024")
            older_text = st.text_area("Older Edition Text", height=150)
        with c4:
            newer_date = st.text_input("Newer Edition Label", placeholder="e.g. Dec 2024")
            newer_text = st.text_area("Newer Edition Text", height=150)

        check = st.form_submit_button("Detect Contradictions", type="primary")

    if check:
        if not (older_text.strip() and newer_text.strip() and section_id.strip()):
            st.warning("Provide both edition texts and a section ID.")
        else:
            with st.spinner("Comparing editions via NLI..."):
                try:
                    result = api_client.check_conflicts(publication, topic, older_date, older_text, newer_date, newer_text, section_id)
                except ApiError as e:
                    result = None
                    st.error(f"Conflict check failed: {e}")

            if result:
                st.divider()
                if result.get("has_conflict"):
                    st.error(f"⚠️ **Superseding Conflict Detected** — {result.get('conflict_description', '')}\n\nSystem will favor **{result.get('superseding_edition')}**.")
                else:
                    st.success("✅ Consistency maintained — no contradiction detected between editions.")
                st.json(result)
