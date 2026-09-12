"""
Evaluation — offline regression harness. This is the system-wide observability
surface: not "is this one answer good," but "is the pipeline getting better or
worse over time," which per-query scorecards can't show on their own.
"""

import math

import streamlit as st

from lib import api_client
from lib.api_client import ApiError
from lib.charts import eval_trend_line


def _min_n_for_alpha(alpha: float) -> int:
    """Mirrors verification/conformal.py::min_n_for_alpha exactly (kept in
    sync manually — Streamlit is a separate process from the backend and
    this is a one-line formula, not worth a round-trip)."""
    return math.ceil(1.0 / alpha) - 1

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
        status = cal.get("status", "UNKNOWN")
        if active.get("effective"):
            st.success(
                f"**Active threshold: {cal.get('threshold')}** — status **{status}** "
                f"(α={cal.get('alpha')}, n={cal.get('n')})  \n{cal.get('coverage_note', '')}"
            )
        else:
            # status != CALIBRATED — a stored INSUFFICIENT_N/NO_DATA calibration
            # (or a legacy file with no status) never overrides the fixed
            # ceiling, no matter what its numeric threshold field says.
            st.warning(
                f"**Stored calibration is NOT active** — status **{status}** "
                f"(α={cal.get('alpha')}, n={cal.get('n')} of {cal.get('min_n', '?')} required)  \n"
                f"Abstention is using the fixed ceiling, not this calibration.  \n{cal.get('coverage_note', '')}"
            )
    elif active is not None:
        st.info(active.get("message", "No calibration set — abstention uses the fixed ceiling (0.25)."))
with cc2:
    alpha = st.slider("α (max wrongful-abstention rate)", 0.01, 0.5, 0.10, 0.01)
    required_n = _min_n_for_alpha(alpha)
    st.caption(f"Needs ≥{required_n} answerable calibration items at this α to avoid INSUFFICIENT_N.")
    force = st.checkbox(
        "Force apply even if insufficient",
        value=False,
        help="Applies the calibration as the active threshold even when status is INSUFFICIENT_N/NO_DATA. "
             "Not recommended — an insufficient calibration provides no real coverage guarantee.",
    )
    if st.button("Run Calibration"):
        with st.spinner("Running the calibration set through the live pipeline..."):
            try:
                res = api_client.calibrate_conformal(alpha=alpha, force=force)
                cal = res.get("calibration", {})
                if res.get("applied"):
                    st.success(
                        f"Calibrated: status={cal.get('status')}, threshold={cal.get('threshold')} "
                        f"from {res.get('num_answerable')} answerable item(s)."
                    )
                else:
                    st.warning(
                        f"Calibration status was **{cal.get('status')}** (n={cal.get('n')} of "
                        f"{cal.get('min_n', '?')} required) — recorded in history but NOT applied as "
                        f"the active threshold. {cal.get('coverage_note', '')}"
                    )
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
            eval_trend_line(runs_chrono, "mean_citation_precision", "Mean Citation Precision"),
            width="stretch",
        )

    st.divider()
    st.markdown("#### Run history")
    _STATUS_ICON = {"complete": "✅", "running": "⏳", "failed": "❌"}
    for run in runs:
        m = run.get("metrics", {})
        status = run.get("status") or ("complete" if m else "unknown")  # pre-W1.2 rows have no status column
        icon = _STATUS_ICON.get(status, "•")
        with st.expander(f"{icon} Run #{run['id']} · {run.get('run_label') or '(unlabeled)'} · {run.get('started_at', '')[:19]}"):
            if status == "failed":
                st.error(f"Run failed: {run.get('error', '(no error recorded)')}")
            elif status == "running":
                st.info("Run is still in progress (or was interrupted mid-run — see the Items tab for what completed so far).")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Queries", m.get("num_queries", 0))
            c2.metric("Faithfulness (post)", f"{(m.get('mean_faithfulness_post') or 0) * 100:.0f}%")
            c3.metric("Abstention Rate", f"{(m.get('abstention_rate') or 0) * 100:.0f}%")
            c4.metric("Mean LLM Calls/Query", f"{m.get('mean_llm_calls_per_query', 0):.1f}")

            # Retrieval quality reads from accuracy.retrieval (graded
            # relevant_chunk_keys). The old top-level retrieval_hit_rate was
            # removed — it compared bare section ids against canonical chunk
            # keys and was structurally always 0.0. Older runs still carry it,
            # so it is deliberately not read back.
            _retrieval = (m.get("accuracy") or {}).get("retrieval") or {}
            _ndcg = (_retrieval.get("ndcg_at_5") or {}).get("value")
            _recall = (_retrieval.get("recall_at_5") or {}).get("value")
            _gt_items = m.get("num_ground_truth_items", 0)
            if _ndcg is not None or _recall is not None:
                st.caption(
                    f"Retrieval — nDCG@5: {_ndcg if _ndcg is None else f'{_ndcg:.3f}'} · "
                    f"recall@5: {_recall if _recall is None else f'{_recall:.3f}'} "
                    f"(over {_gt_items} labeled item(s))"
                )
            else:
                st.caption("Retrieval: no graded ground truth in this dataset — not scored (see docs/BENCHMARK_READINESS.md, Phase 3).")
            st.caption(f"Trust status distribution: {m.get('trust_status_distribution', {})}")
            if m.get("error_breakdown"):
                st.caption(f"⚠️ Error breakdown: {m['error_breakdown']}")
            if run.get("git_commit"):
                st.caption(f"Git commit: `{run['git_commit'][:12]}`")

            if st.button("View per-query detail", key=f"eval_detail_{run['id']}"):
                detail = api_client.get_eval_run(run["id"])
                for pq in detail.get("per_query", []):
                    status = "❌ error" if pq.get("error") else pq.get("trust_status", "?")
                    st.write(f"**{pq.get('id')}** — {status} — {pq.get('query', '')[:80]}")
