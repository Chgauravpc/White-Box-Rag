"""
harness.py — Offline evaluation harness for the RAG pipeline.

Runs a golden-dataset JSONL file through ingestion/pipeline.py::run_query_pipeline
directly (no HTTP round-trip), then aggregates industry-standard RAG metrics:
retrieval hit-rate/MRR (only for items with ground-truth expected_section_ids),
faithfulness/context/citation/answer-relevancy means, Trust Gate status
distribution, and abstention rate. This is what turns the per-query Trust
Scorecard into system-wide regression tracking over time.

`_aggregate()` is a pure function (no I/O, no models) so it can be unit
tested against synthetic per-query dicts — see tests/test_eval_harness.py.
The full `run_eval()` execution path calls the real pipeline (real Gemini
calls) and is not exercised in the mocked test suite.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

EVAL_CONCURRENCY = 3


def _load_dataset(dataset_path: str) -> list[dict]:
    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def _run_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    from ingestion.pipeline import run_query_pipeline  # local import avoids import-time model loading for pure-aggregation tests

    async with semaphore:
        try:
            report = await run_query_pipeline(item["query"])
        except Exception as e:
            logger.error(f"Eval item {item.get('id')} failed: {e}")
            return {
                "id": item.get("id"),
                "query": item.get("query"),
                "expected_section_ids": item.get("expected_section_ids", []),
                "error": str(e),
            }

        retrieved_chunk_ids = (
            report.xai_artifacts.retrieval.chunk_ids if report.xai_artifacts and report.xai_artifacts.retrieval else []
        )
        claims_total = len(report.claims)
        claims_stripped = sum(1 for c in report.claims if not c.retained)
        # Nonconformity score for conformal calibration: retained-set mean penalty
        # (= what should_abstain compares against the threshold), reconstructed here
        # from the report so should_abstain's signature stays unchanged.
        n_retained = claims_total - claims_stripped
        mean_penalty = round((1.0 - report.retained_trust_score) / max(1, n_retained), 6)

        return {
            "id": item.get("id"),
            "query": item.get("query"),
            "expected_section_ids": item.get("expected_section_ids", []),
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
            "gemini_call_count": report.gemini_call_count,
        }


def _aggregate(per_query_results: list[dict]) -> dict:
    """Pure aggregation over per-query result dicts (see `_run_one` for shape)."""
    n = len(per_query_results)
    errored = [r for r in per_query_results if r.get("error")]
    ok = [r for r in per_query_results if not r.get("error")]

    def _mean(key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in ok if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 6) if vals else default

    # Retrieval hit-rate / MRR — only over items with ground-truth expected_section_ids.
    ground_truth_items = [r for r in ok if r.get("expected_section_ids")]
    hits = 0
    reciprocal_ranks = []
    for r in ground_truth_items:
        expected = set(r["expected_section_ids"])
        retrieved = r.get("retrieved_chunk_ids", [])
        rank = None
        for i, cid in enumerate(retrieved):
            if cid in expected:
                rank = i + 1
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate = round(hits / len(ground_truth_items), 6) if ground_truth_items else None
    mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6) if reciprocal_ranks else None

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

    # Latency: mean + p95 per stage, and mean Gemini calls/query
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

    mean_gemini_calls = _mean("gemini_call_count")

    return {
        "num_queries": n,
        "num_errors": len(errored),
        "retrieval_hit_rate": hit_rate,
        "retrieval_mrr": mrr,
        "num_ground_truth_items": len(ground_truth_items),
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
        "mean_gemini_calls_per_query": mean_gemini_calls,
        "latency_summary": latency_summary,
    }


async def run_eval(dataset_path: str, run_label: str = "") -> dict:
    """Runs the golden dataset through the real pipeline with bounded concurrency
    and returns {summary metrics, per_query results, started_at, finished_at}.
    """
    items = _load_dataset(dataset_path)
    started_at = datetime.utcnow().isoformat()

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    per_query_results = await asyncio.gather(*[_run_one(item, semaphore) for item in items])
    per_query_results = list(per_query_results)

    finished_at = datetime.utcnow().isoformat()
    metrics = _aggregate(per_query_results)

    return {
        "run_label": run_label,
        "dataset_path": dataset_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "num_queries": len(items),
        "metrics": metrics,
        "per_query": per_query_results,
    }


async def run_calibration(dataset_path: str, alpha: float = 0.1, run_label: str = "") -> dict:
    """Run a labelled dataset through the live pipeline and conformally calibrate
    the abstention threshold from the *answerable* items' mean-penalty scores.

    Requires a live GEMINI_API_KEY + ingested corpus (executes the full pipeline).
    Persists the active calibration so should_abstain picks it up immediately.
    """
    from verification.conformal import calibrate_threshold, save_calibration

    items = _load_dataset(dataset_path)
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    results = list(await asyncio.gather(*[_run_one(item, semaphore) for item in items]))

    answerable = [r for r in results if not r.get("error") and not r.get("expected_abstain", False)]
    scores = [r["mean_penalty"] for r in answerable if r.get("mean_penalty") is not None]

    calibration = calibrate_threshold(scores, alpha)
    calibration.update({
        "dataset_path": dataset_path,
        "run_label": run_label,
        "created_at": datetime.utcnow().isoformat(),
    })
    save_calibration(calibration)

    return {
        "calibration": calibration,
        "num_items": len(items),
        "num_answerable": len(answerable),
        "num_errors": sum(1 for r in results if r.get("error")),
        "scores": sorted(scores),
        "per_query": results,
    }
