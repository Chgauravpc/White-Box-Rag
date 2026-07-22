"""
RAG Query — the real, live pipeline (the React app's Query page never called
the real API at all; this one always does).
"""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import scorecard_radar
from lib.ui import trust_badge, render_claim_card, score_pill, render_counterfactuals

st.title("💬 RAG Query")
st.caption("Ask a question. The answer is filtered to grounded statements only — see 'Show raw answer' for the unfiltered version.")

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

example_prompts = [
    "What are the key requirements described in the ingested documents?",
    "Summarize the main obligations of each party.",
    "What happens if a term is not met?",
]

chip_cols = st.columns(len(example_prompts))
for i, prompt in enumerate(example_prompts):
    if chip_cols[i].button(prompt, key=f"chip_{i}", width="stretch"):
        st.session_state["query_text"] = prompt

query = st.text_area("Your question", value=st.session_state["query_text"], height=80, key="query_input")
run = st.button("Run Query", type="primary")

if run and query.strip():
    with st.spinner("Running retrieval → generation → NLI verification → trust gating (can take up to a minute)..."):
        try:
            report = api_client.query_rag(query.strip())
        except ApiError as e:
            report = None
            st.error(f"Query failed: {e}")

    if report:
        trust_gate = report.get("trust_gate") or {}
        scorecard = report.get("scorecard") or {}
        claims = report.get("claims", [])
        edition_conflicts = report.get("edition_conflicts", [])
        abstained = report.get("abstained", False)
        latency = report.get("latency_ms", {})
        gemini_calls = report.get("gemini_call_count", 0)

        st.divider()

        if edition_conflicts:
            with st.container(border=True):
                st.warning(f"⚠️ **Regulatory/Edition Conflict Detected** — {len(edition_conflicts)} cited section(s) conflict with a different edition in the knowledge base.")
                for c in edition_conflicts:
                    st.caption(f"{c.get('publication')} · §{c.get('section_id')}: {c.get('conflict_description')} — superseding edition: {c.get('superseding_edition')}")

        if abstained:
            st.error(f"🚫 **Abstained — insufficient grounded evidence.**\n\n{report.get('abstention_reason', '')}")
            col1, col2 = st.columns(2)
            col1.metric("Raw Trust Score", f"{trust_gate.get('overall_score', 0) * 100:.0f}%")
            col2.metric("Retained-set Trust Score", f"{report.get('retained_trust_score', 0) * 100:.0f}%")
        else:
            main_col, side_col = st.columns([2, 1])

            with main_col:
                trust_badge(trust_gate.get("status", "Unknown"), trust_gate.get("overall_score"))
                st.markdown("#### Verified Statements")
                retained_claims = [c for c in claims if c.get("retained", True)]
                if retained_claims:
                    for c in retained_claims:
                        render_claim_card(c)
                else:
                    st.info("No individually-verifiable statements were extracted from this answer.")

                fcol1, fcol2 = st.columns(2)
                fcol1.metric("Faithfulness (raw)", f"{report.get('faithfulness_raw', 0) * 100:.0f}%")
                fcol2.metric("Faithfulness (post-filter)", f"{report.get('faithfulness_post', 0) * 100:.0f}%")

                with st.expander(f"Show raw ungrounded answer & full claim breakdown ({len(claims)} claim(s), {len(claims) - len(retained_claims)} removed)"):
                    st.write(report.get("response", ""))
                    st.markdown("---")
                    for c in claims:
                        render_claim_card(c)

                render_counterfactuals(report.get("counterfactuals", []))

                st.caption(f"⏱️ {sum(latency.values()):.0f}ms total · {gemini_calls} Gemini call(s)")

            with side_col:
                st.markdown("#### Trust Scorecard")
                if scorecard:
                    fig = scorecard_radar(scorecard, faithfulness_post=report.get("faithfulness_post"))
                    st.plotly_chart(fig, width="stretch")
                    for key, label in [
                        ("context_relevance", "Context Relevance"),
                        ("context_diversity", "Context Diversity"),
                        ("citation_precision", "Citation Precision"),
                        ("answer_relevancy", "Answer Relevancy"),
                        ("context_utilization", "Context Utilization"),
                        ("paraphrase_stability", "Paraphrase Stability"),
                    ]:
                        score_pill(label, scorecard.get(key, 0.0))

                st.markdown("#### Trust Gate")
                st.write(trust_gate.get("reasoning", ""))

elif run:
    st.warning("Please enter a question.")
