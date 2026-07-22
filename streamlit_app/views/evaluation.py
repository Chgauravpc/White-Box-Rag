"""
Evaluation — offline regression harness. This is the system-wide observability
surface: not "is this one answer good," but "is the pipeline getting better or
worse over time," which per-query scorecards can't show on their own.
"""

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import eval_trend_line

st.title("📈 Evaluation & Observability")
st.caption("Run the golden-dataset harness and track aggregate hallucination/trust metrics across runs over time.")

col1, col2 = st.columns([2, 1])
with col1:
    run_label = st.text_input("Run label (optional)", placeholder="e.g. 'after threshold tuning'")
with col2:
    st.write("")
    st.write("")
    if st.button("▶ Run Evaluation", type="primary"):
        with st.spinner("Running the golden dataset through the live pipeline (bounded concurrency — can take a few minutes)..."):
            try:
                result = api_client.run_eval(run_label=run_label)
                st.success(f"Run #{result.get('id')} complete — {result['metrics']['num_queries']} quer" + ("y" if result['metrics']['num_queries'] == 1 else "ies") + " evaluated.")
            except ApiError as e:
                st.error(f"Eval run failed: {e}")

st.divider()

# ── Conformal abstention calibration ──
st.markdown("#### 🎯 Conformal Abstention Calibration")
st.caption(
    "Replace the hand-tuned abstention ceiling with a threshold calibrated to a statistical "
    "risk target (split conformal). Calibrating at α means at most ~α of truly-answerable "
    "queries are wrongly abstained. Runs the live pipeline over the labelled calibration set."
)

try:
    active = api_client.get_active_calibration()
except ApiError as e:
    active = None
    st.error(f"Could not load calibration status: {e}")

cc1, cc2 = st.columns([2, 1])
with cc1:
    if active and active.get("active"):
        cal = active["calibration"]
        st.success(
            f"**Active threshold: {cal.get('threshold')}** "
            f"(α={cal.get('alpha')}, n={cal.get('n')})  \n{cal.get('coverage_note', '')}"
        )
    elif active is not None:
        st.info(active.get("message", "No calibration set — abstention uses the fixed ceiling (0.25)."))
with cc2:
    alpha = st.slider("α (max wrongful-abstention rate)", 0.01, 0.5, 0.10, 0.01)
    if st.button("Run Calibration"):
        with st.spinner("Running the calibration set through the live pipeline..."):
            try:
                res = api_client.calibrate_conformal(alpha=alpha)
                cal = res.get("calibration", {})
                st.success(f"Calibrated: threshold={cal.get('threshold')} from {res.get('num_answerable')} answerable item(s).")
                st.rerun()
            except ApiError as e:
                st.error(f"Calibration failed: {e}")

st.divider()

try:
    runs = api_client.list_eval_runs()
except ApiError as e:
    runs = []
    st.error(f"Could not load eval runs: {e}")

if not runs:
    st.info("No evaluation runs yet. Click 'Run Evaluation' to establish a baseline.")
else:
    runs_chrono = list(reversed(runs))  # oldest -> newest for trend charts

    st.markdown("#### Trends")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(
            eval_trend_line(runs_chrono, "mean_faithfulness_post", "Mean Faithfulness (post-filter)",
                             threshold=0.8, threshold_label="target ≥ 0.8"),
            width="stretch",
        )
    with r2:
        st.plotly_chart(
            eval_trend_line(runs_chrono, "abstention_rate", "Abstention Rate",
                             threshold=0.2, threshold_label="ceiling ≤ 0.2"),
            width="stretch",
        )

    r3, r4 = st.columns(2)
    with r3:
        st.plotly_chart(
            eval_trend_line(runs_chrono, "mean_context_relevance", "Mean Context Relevance"),
            width="stretch",
        )
    with r4:
        st.plotly_chart(
            eval_trend_line(runs_chrono, "retrieval_hit_rate", "Retrieval Hit Rate"),
            width="stretch",
        )

    st.divider()
    st.markdown("#### Run history")
    for run in runs:
        m = run.get("metrics", {})
        with st.expander(f"Run #{run['id']} · {run.get('run_label') or '(unlabeled)'} · {run.get('started_at', '')[:19]}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Queries", m.get("num_queries", 0))
            c2.metric("Faithfulness (post)", f"{(m.get('mean_faithfulness_post') or 0) * 100:.0f}%")
            c3.metric("Abstention Rate", f"{(m.get('abstention_rate') or 0) * 100:.0f}%")
            c4.metric("Mean Gemini Calls/Query", f"{m.get('mean_gemini_calls_per_query', 0):.1f}")

            if m.get("retrieval_hit_rate") is not None:
                st.caption(f"Retrieval Hit Rate: {m['retrieval_hit_rate']*100:.0f}% · MRR: {m.get('retrieval_mrr', 0):.2f} (over {m.get('num_ground_truth_items', 0)} ground-truth item(s))")
            st.caption(f"Trust status distribution: {m.get('trust_status_distribution', {})}")

            if st.button("View per-query detail", key=f"eval_detail_{run['id']}"):
                detail = api_client.get_eval_run(run["id"])
                for pq in detail.get("per_query", []):
                    status = "❌ error" if pq.get("error") else pq.get("trust_status", "?")
                    st.write(f"**{pq.get('id')}** — {status} — {pq.get('query', '')[:80]}")
