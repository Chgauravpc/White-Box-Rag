"""
harness.py — Offline evaluation harness for the RAG pipeline.

Runs a golden-dataset JSONL file through ingestion/pipeline.py::run_query_pipeline
directly (no HTTP round-trip), then aggregates industry-standard RAG metrics:
retrieval hit-rate/MRR (only for items with ground-truth expected_section_ids),
faithfulness/context/citation/answer-relevancy means, Trust Gate status
distribution, and abstention rate — PLUS (via eval/scoring.py) real
label-vs-prediction accuracy: precision/recall/F1 of the abstain decision
against `expected_abstain`, and retrieval hit@k/recall@k/nDCG@k/MRR/MAP
against graded relevance judgments, each with a confidence interval. Before
scoring.py existed, nothing in this file compared a prediction to a label —
abstention_rate and the other means below are unlabeled rates, not accuracy.

Run-level provenance and durability (finding #4, #8): `capture_run_provenance()`
snapshots everything that can make two runs produce different numbers
(RunConfig, git commit, corpus state, active calibration, model identities);
eval/routes.py persists an eval_runs row BEFORE execution (status='running')
and finalizes it in a finally-block, so a crash leaves a diagnosable
'failed'/'running' row instead of nothing. Each item is written to the
`eval_items` table the moment it completes (`_persist_item`), not just held
in memory until the final aggregate — a crash at item 299 of 300 keeps the
first 298. `resume_run` (via `run_eval(..., resume_from_run_id=...)`) skips
items already recorded for a given run.

`_aggregate()` is a pure function (no I/O, no models) so it can be unit
tested against synthetic per-query dicts — see tests/test_eval_harness.py.
The full `run_eval()` execution path calls the real pipeline (real LLM
calls) and is not exercised in the mocked test suite.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from shared.config import EVAL_CONCURRENCY
from shared.llm_pool_config import effective_eval_concurrency
from eval import scoring
from eval.schema import load_and_validate, relevant_chunk_keys_as_dict

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dataset(dataset_path: str) -> list[dict]:
    """Load a JSONL dataset, guarded per-line — one malformed line must not
    kill the whole run before a single query executes (finding #8). Bad
    lines are logged with their line number and skipped, not silently
    dropped and not fatal.
    """
    raw_items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"{dataset_path}:{line_no}: skipping malformed JSONL line ({e})")
    items, errors_by_id = load_and_validate(raw_items)
    if errors_by_id:
        logger.warning(f"{len(errors_by_id)} dataset item(s) failed schema validation: {errors_by_id}")
    return items


# ──────────────────────────────────────────────
#  Error taxonomy (finding #8) — string-matched against the exception
#  message, mirroring the retry-classification style already used in
#  shared/llm.py. Approximate by nature (there's no structured exception
#  hierarchy for these failure modes), but turns "num_errors: 12" into a
#  diagnosable breakdown instead of an opaque count.
# ──────────────────────────────────────────────

ERROR_CATEGORIES = (
    "RATE_LIMIT", "CAPACITY_EXHAUSTED", "LLM_API", "JSON_PARSE",
    "EMPTY_CORPUS", "TIMEOUT", "MODEL_LOAD", "CASSETTE_MISS", "UNKNOWN",
)


def classify_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "cassette" in msg:
        return "CASSETTE_MISS"  # LLM_MODE=replay with no recorded response (Phase 2, W2.5)
    if "rate limit" in msg or "429" in msg or "rate_limit" in msg:
        return "RATE_LIMIT"
    if "capacity" in msg or "insufficient_quota" in msg or "quota" in msg:
        return "CAPACITY_EXHAUSTED"
    if "json" in msg and ("decode" in msg or "parse" in msg or "expecting" in msg):
        return "JSON_PARSE"
    if "no relevant documents" in msg or "empty corpus" in msg or "chromadb is empty" in msg:
        return "EMPTY_CORPUS"
    if "timeout" in msg or "timed out" in msg:
        return "TIMEOUT"
    if "model" in msg and ("load" in msg or "not found" in msg or "download" in msg):
        return "MODEL_LOAD"
    if "connection" in msg or "network" in msg or isinstance(exc, ConnectionError):
        return "LLM_API"
    if "api" in msg or "500" in msg or "503" in msg or "502" in msg:
        return "LLM_API"
    return "UNKNOWN"


# ──────────────────────────────────────────────
#  Run provenance (finding #4)
# ──────────────────────────────────────────────

# _hash_file used to live here and read whole files into memory; it is now
# shared/hashing.py::hash_file (chunked), shared with the corpus manifest.
from shared.hashing import hash_file as _hash_file


def capture_run_provenance(dataset_path: str) -> dict:
    """Snapshot everything that can make two runs produce different numbers,
    for stamping onto the eval_runs row at start.

    `corpus_manifest` is the content-hashed W3.1 manifest
    (shared/corpus_manifest.py): every chunk's key AND the sha256 of its
    normalized text, so two runs can be compared on whether the corpus
    actually changed rather than just whether the file list did. The full
    per-chunk inventory is kept out of the stamped row — only the hashes and
    counts, since the row is provenance, not a second copy of the corpus.
    """
    from shared.runconfig import capture_run_config
    from shared.xai_matrices import model_identity
    from shared.corpus_manifest import build_manifest
    from verification.conformal import load_active_calibration

    run_config = capture_run_config()
    try:
        full = build_manifest()
        corpus_manifest = {k: v for k, v in full.items() if k != "documents"}
        corpus_manifest["documents"] = [
            {k: v for k, v in doc.items() if k != "chunks"} for doc in full.get("documents", [])
        ]
    except Exception as exc:
        # A provenance snapshot must never be the reason a run fails to start.
        corpus_manifest = {"error": f"{type(exc).__name__}: {exc}"}
    active_calibration = load_active_calibration()
    if active_calibration and active_calibration.get("threshold") == float("inf"):
        active_calibration = dict(active_calibration, threshold="inf")

    return {
        "run_config": run_config.to_dict(),
        "git_commit": run_config.git_commit,
        "corpus_manifest": corpus_manifest,
        "dataset_hash": _hash_file(dataset_path) if os.path.exists(dataset_path) else "",
        "active_calibration": active_calibration,
        "model_identities": model_identity(),
        # The threshold every item in this run will actually use — frozen
        # once here rather than re-read from disk per item (finding #20).
        "frozen_abstention_threshold": (
            active_calibration.get("threshold")
            if active_calibration and active_calibration.get("status") == "CALIBRATED"
            else None
        ),
    }


def _persist_item(run_id: Optional[int], item_id, status: str, error_category: Optional[str],
                   error_detail: Optional[str], result: dict) -> None:
    """Write one item's result the moment it completes. Tolerant of failure —
    a persistence error must not fail the eval item itself."""
    if run_id is None:
        return
    try:
        from shared.database import insert_eval_item
        insert_eval_item(run_id, item_id, status, error_category, error_detail, json.dumps(result, default=str))
    except Exception as e:
        logger.warning(f"Failed to persist eval item {item_id!r} for run {run_id}: {e}")


async def _run_one(
    item: dict,
    semaphore: asyncio.Semaphore,
    eval_mode: bool = True,
    threshold: Optional[float] = None,
    run_id: Optional[int] = None,
) -> dict:
    from ingestion.pipeline import run_query_pipeline  # local import avoids import-time model loading for pure-aggregation tests
    from verification.mitigation import nonconformity_score  # same import-time reasoning

    async with semaphore:
        try:
            report = await run_query_pipeline(item["query"], eval_mode=eval_mode, abstention_threshold=threshold)
        except Exception as e:
            category = classify_error(e)
            logger.error(f"Eval item {item.get('id')} failed [{category}]: {e}")
            result = {
                "id": item.get("id"),
                "query": item.get("query"),
                "expected_section_ids": item.get("expected_section_ids", []),
                "relevant_chunk_keys": item.get("relevant_chunk_keys", []),
                "expected_abstain": item.get("expected_abstain", False),
                "error": str(e),
                "error_category": category,
            }
            _persist_item(run_id, item.get("id"), "error", category, str(e), result)
            return result

        retrieved_chunk_ids = (
            report.xai_artifacts.retrieval.chunk_ids if report.xai_artifacts and report.xai_artifacts.retrieval else []
        )
        claims_total = len(report.claims)
        claims_stripped = sum(1 for c in report.claims if not c.retained)
        # Nonconformity score for conformal calibration: calls the SAME
        # function should_abstain uses (verification/mitigation.py), instead
        # of reconstructing an approximation from claims_total/claims_stripped
        # that silently diverges on two degenerate paths (empty retrieval,
        # all-claims-stripped) — see nonconformity_score's docstring. Callers
        # must exclude None, never invent a 0.0/1.0 fallback.
        primary_attrs = (
            report.xai_artifacts.attribution.primary_attributions
            if report.xai_artifacts and report.xai_artifacts.attribution else []
        )
        retained_flags = [c.retained for c in report.claims]
        raw_mean_penalty, _ = nonconformity_score(
            report.claims, report.verifications, primary_attrs, retained_flags, report.edition_conflicts,
        )
        mean_penalty = round(raw_mean_penalty, 6) if raw_mean_penalty is not None else None

        result = {
            "id": item.get("id"),
            "query": item.get("query"),
            "expected_section_ids": item.get("expected_section_ids", []),
            "relevant_chunk_keys": item.get("relevant_chunk_keys", []),
            "expected_abstain": item.get("expected_abstain", False),
            "mean_penalty": mean_penalty,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "faithfulness_raw": report.faithfulness_raw,
            "faithfulness_post": report.faithfulness_post,
            "context_relevance": report.scorecard.context_relevance if report.scorecard else 0.0,
            "context_diversity": report.scorecard.context_diversity if report.scorecard else 0.0,
            "citation_precision": report.scorecard.citation_precision if report.scorecard else 0.0,
            "answer_relevancy": report.scorecard.answer_relevancy if report.scorecard else 0.0,
            "context_utilization": report.scorecard.context_utilization if report.scorecard else 0.0,
            "paraphrase_stability": report.scorecard.paraphrase_stability if report.scorecard else 1.0,
            "trust_status": report.trust_gate.status if report.trust_gate else None,
            "abstained": report.abstained,
            "claims_total": claims_total,
            "claims_stripped": claims_stripped,
            "latency_ms": report.latency_ms,
            "llm_call_count": report.llm_call_count,
        }
        _persist_item(run_id, item.get("id"), "ok", None, None, result)
        return result


def _normalize_gathered(item: dict, raw_result) -> dict:
    """asyncio.gather(..., return_exceptions=True) can hand back an
    exception object instead of _run_one's return value — e.g. if the
    coroutine itself is cancelled (a disconnecting HTTP client) rather than
    _run_one's own try/except catching it. Normalize either shape into the
    same error-dict format so _aggregate never has to know the difference."""
    if isinstance(raw_result, BaseException):
        category = classify_error(raw_result)
        logger.error(f"Eval item {item.get('id')} failed outside _run_one's own guard [{category}]: {raw_result}")
        return {
            "id": item.get("id"),
            "query": item.get("query"),
            "expected_section_ids": item.get("expected_section_ids", []),
            "relevant_chunk_keys": item.get("relevant_chunk_keys", []),
            "expected_abstain": item.get("expected_abstain", False),
            "error": str(raw_result),
            "error_category": category,
        }
    return raw_result


def _aggregate(per_query_results: list[dict]) -> dict:
    """Pure aggregation over per-query result dicts (see `_run_one` for shape)."""
    n = len(per_query_results)
    errored = [r for r in per_query_results if r.get("error")]
    ok = [r for r in per_query_results if not r.get("error")]

    def _mean(key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in ok if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 6) if vals else default

    # The legacy hit_rate/MRR block that used to live here was removed: it
    # compared `expected_section_ids` (bare section ids, e.g. "1.1") against
    # `retrieved_chunk_ids`, which are canonical chunk keys
    # (publication|edition|section|ordinal, via retriever.py's
    # resolve_chunk_key). `cid in expected` could therefore never be true, so
    # hit_rate was structurally pinned at 0.0 for every item that had any
    # ground truth at all — a real-looking number that measured nothing, and
    # was reported next to the correct accuracy.retrieval block below.
    # Retrieval is now scored only by eval/scoring.py::retrieval_metrics
    # against graded `relevant_chunk_keys`.

    # Trust Gate status distribution
    status_counts: dict[str, int] = {}
    for r in ok:
        status = r.get("trust_status") or "Unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    abstention_rate = round(sum(1 for r in ok if r.get("abstained")) / len(ok), 6) if ok else 0.0

    strip_ratios = [
        (r["claims_stripped"] / r["claims_total"]) for r in ok if r.get("claims_total", 0) > 0
    ]
    mean_strip_ratio = round(sum(strip_ratios) / len(strip_ratios), 6) if strip_ratios else 0.0

    # Latency: mean + p95 per stage, and mean LLM calls/query
    stage_names: set[str] = set()
    for r in ok:
        stage_names.update((r.get("latency_ms") or {}).keys())
    latency_summary = {}
    for stage in stage_names:
        vals = sorted((r["latency_ms"][stage] for r in ok if stage in (r.get("latency_ms") or {})))
        if not vals:
            continue
        p95_idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
        latency_summary[stage] = {
            "mean_ms": round(sum(vals) / len(vals), 2),
            "p95_ms": round(vals[p95_idx], 2),
        }

    mean_llm_calls = _mean("llm_call_count")

    # Error-category breakdown (finding #8) — "12 errors" is now diagnosable.
    error_breakdown: dict[str, int] = {}
    for r in errored:
        cat = r.get("error_category") or "UNKNOWN"
        error_breakdown[cat] = error_breakdown.get(cat, 0) + 1

    # ── Label-vs-prediction accuracy (eval/scoring.py) ──
    # Abstention is always scorable — every item carries expected_abstain.
    # Retrieval accuracy is only scorable for items with real graded
    # relevance (relevant_chunk_keys); today that's empty for most items, so
    # this correctly reports "no ground truth" rather than a misleading
    # zero (see scoring.retrieval_metrics's n=0 branch).
    accuracy = {
        "abstention": scoring.abstention_metrics(
            [bool(r.get("expected_abstain")) for r in ok],
            [bool(r.get("abstained")) for r in ok],
        ),
        "retrieval": scoring.retrieval_metrics([
            {
                "retrieved": r.get("retrieved_chunk_ids", []),
                "relevant": relevant_chunk_keys_as_dict(r),
            }
            for r in ok
        ]),
    }

    return {
        "num_queries": n,
        "num_errors": len(errored),
        "error_breakdown": error_breakdown,
        # retrieval_hit_rate / retrieval_mrr / num_ground_truth_items are gone
        # (see the note above _aggregate's accuracy block). Retrieval quality
        # now lives only under accuracy.retrieval, which scores canonical
        # chunk keys against graded relevant_chunk_keys.
        "num_ground_truth_items": sum(
            1 for r in ok if r.get("relevant_chunk_keys")
        ),
        "accuracy": accuracy,
        "mean_faithfulness_raw": _mean("faithfulness_raw"),
        "mean_faithfulness_post": _mean("faithfulness_post"),
        "mean_context_relevance": _mean("context_relevance"),
        "mean_context_diversity": _mean("context_diversity"),
        "mean_citation_precision": _mean("citation_precision"),
        "mean_answer_relevancy": _mean("answer_relevancy"),
        "mean_context_utilization": _mean("context_utilization"),
        "mean_paraphrase_stability": _mean("paraphrase_stability", default=1.0),
        "trust_status_distribution": status_counts,
        "abstention_rate": abstention_rate,
        "mean_claims_stripped_ratio": mean_strip_ratio,
        "mean_llm_calls_per_query": mean_llm_calls,
        "latency_summary": latency_summary,
    }


def _frozen_threshold() -> Optional[float]:
    """Read the active calibrated threshold ONCE per run (not per item) —
    see run_query_pipeline's abstention_threshold docstring (finding #20)."""
    from verification.conformal import load_active_threshold
    return load_active_threshold()


async def run_eval(
    dataset_path: str,
    run_label: str = "",
    run_id: Optional[int] = None,
    eval_mode: bool = True,
    resume_from_run_id: Optional[int] = None,
) -> dict:
    """Runs the golden dataset through the real pipeline with bounded concurrency
    and returns {summary metrics, per_query results, started_at, finished_at}.

    `run_id`: an eval_runs row id, pre-created by the caller (eval/routes.py)
    with status='running' — each item is persisted to `eval_items` under this
    id as it completes. `resume_from_run_id`: skip items already recorded for
    that run and continue writing into the SAME run id (merging prior +
    new results into the final aggregate) rather than starting over.
    """
    items = _load_dataset(dataset_path)
    started_at = _utcnow_iso()

    prior_results = []
    if resume_from_run_id is not None:
        from shared.database import get_completed_item_ids, list_eval_items
        completed_ids = get_completed_item_ids(resume_from_run_id)
        if completed_ids:
            items = [item for item in items if str(item.get("id")) not in completed_ids]
            prior_results = [json.loads(row["result_json"]) for row in list_eval_items(resume_from_run_id)]
        run_id = resume_from_run_id
        logger.info(f"Resuming run {resume_from_run_id}: {len(completed_ids)} item(s) already done, {len(items)} remaining")

    threshold = _frozen_threshold()
    semaphore = asyncio.Semaphore(effective_eval_concurrency(EVAL_CONCURRENCY))
    gathered = await asyncio.gather(
        *[_run_one(item, semaphore, eval_mode=eval_mode, threshold=threshold, run_id=run_id) for item in items],
        return_exceptions=True,
    )
    new_results = [_normalize_gathered(item, r) for item, r in zip(items, gathered)]
    per_query_results = prior_results + new_results

    finished_at = _utcnow_iso()
    metrics = _aggregate(per_query_results)

    return {
        "run_label": run_label,
        "dataset_path": dataset_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "num_queries": len(per_query_results),
        "metrics": metrics,
        "per_query": per_query_results,
        "resumed_from_run_id": resume_from_run_id,
    }


async def run_calibration(
    dataset_path: str,
    alpha: float = 0.1,
    run_label: str = "",
    force: bool = False,
    eval_mode: bool = True,
) -> dict:
    """Run a labelled dataset through the live pipeline and conformally calibrate
    the abstention threshold from the *answerable* items' mean-penalty scores.

    Requires a live LLM API key (GROQ_API_KEY / OPENROUTER_API_KEY) + ingested corpus (executes the full pipeline).

    A calibration only becomes the ACTIVE threshold (persisted for
    should_abstain to read) when its status is CALIBRATED, or when
    `force=True` is explicitly passed. An INSUFFICIENT_N/NO_DATA result is
    still recorded in the tamper-evident history either way — refusing to
    apply it is what closes the fail-open bug (a too-small calibration set
    silently disabling abstention entirely); recording the attempt regardless
    keeps the audit trail honest about what was tried.
    """
    from verification.conformal import calibrate_threshold, save_calibration, record_calibration_history, CalibrationStatus

    items = _load_dataset(dataset_path)
    threshold = _frozen_threshold()
    semaphore = asyncio.Semaphore(effective_eval_concurrency(EVAL_CONCURRENCY))
    gathered = await asyncio.gather(
        *[_run_one(item, semaphore, eval_mode=eval_mode, threshold=threshold) for item in items],
        return_exceptions=True,
    )
    results = [_normalize_gathered(item, r) for item, r in zip(items, gathered)]

    answerable = [r for r in results if not r.get("error") and not r.get("expected_abstain", False)]
    scores = [r["mean_penalty"] for r in answerable if r.get("mean_penalty") is not None]

    calibration = calibrate_threshold(scores, alpha)
    calibration.update({
        "dataset_path": dataset_path,
        "run_label": run_label,
        "created_at": _utcnow_iso(),
    })

    applied = calibration["status"] == CalibrationStatus.CALIBRATED or force
    if applied:
        save_calibration(calibration)
    else:
        record_calibration_history(calibration)

    return {
        "calibration": calibration,
        "applied": applied,
        "num_items": len(items),
        "num_answerable": len(answerable),
        "num_errors": sum(1 for r in results if r.get("error")),
        "scores": sorted(scores),
        "per_query": results,
    }
