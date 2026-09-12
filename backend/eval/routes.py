"""
API Routes for the offline evaluation harness.

Endpoints:
  POST /api/eval/run          — Run the golden dataset through the real pipeline
  GET  /api/eval/runs         — List past run summaries (for trend charting)
  GET  /api/eval/runs/{id}    — Full run detail with per-query breakdown
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval.harness import run_eval, run_calibration, _load_dataset, capture_run_provenance
from shared.database import insert_eval_run_started, finalize_eval_run, list_eval_runs, get_eval_run
from verification.conformal import load_active_calibration, min_n_for_alpha

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["Evaluation"])

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
DEFAULT_CALIBRATION_DATASET = os.path.join(os.path.dirname(__file__), "calibration_dataset.jsonl")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunEvalRequest(BaseModel):
    dataset_path: str | None = None
    run_label: str = ""
    eval_mode: bool = True
    resume_from_run_id: int | None = None


class CalibrateRequest(BaseModel):
    dataset_path: str | None = None
    alpha: float = 0.1
    run_label: str = ""
    force: bool = False


@router.post("/run")
async def trigger_eval_run(request: RunEvalRequest):
    """Runs the golden dataset through the real pipeline (synchronous — datasets
    are expected to be tens of queries; runs with bounded concurrency internally).

    The eval_runs row is created BEFORE execution (status='running') and
    finalized in a finally-block — a crash or timeout mid-run leaves a
    diagnosable 'failed' row (with `error` set) instead of no record at all
    (finding #8). Pass `resume_from_run_id` to continue a prior run without
    re-running items it already completed (finding #8's resume capability).
    """
    dataset_path = request.dataset_path or DEFAULT_DATASET_PATH
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")

    if request.resume_from_run_id is not None:
        prior = get_eval_run(request.resume_from_run_id)
        if not prior:
            raise HTTPException(status_code=404, detail=f"Cannot resume: run {request.resume_from_run_id} not found.")
        run_id = request.resume_from_run_id
    else:
        provenance = capture_run_provenance(dataset_path)
        run_id = insert_eval_run_started(
            run_label=request.run_label,
            dataset_path=dataset_path,
            started_at=_utcnow_iso(),
            run_config_json=json.dumps(provenance["run_config"], default=str),
            git_commit=provenance["git_commit"],
            corpus_manifest_json=json.dumps(provenance["corpus_manifest"], default=str),
            dataset_hash=provenance["dataset_hash"],
            active_calibration_json=json.dumps(provenance["active_calibration"], default=str) if provenance["active_calibration"] else None,
            model_identities_json=json.dumps(provenance["model_identities"], default=str),
            eval_mode=request.eval_mode,
        )

    try:
        result = await run_eval(
            dataset_path, run_label=request.run_label, run_id=run_id,
            eval_mode=request.eval_mode, resume_from_run_id=request.resume_from_run_id,
        )
        finalize_eval_run(
            run_id, status="complete", finished_at=_utcnow_iso(),
            num_queries=result["num_queries"],
            metrics_json=json.dumps(result["metrics"]),
            per_query_json=json.dumps(result["per_query"], default=str),
        )
    except Exception as e:
        logger.error(f"Eval run failed: {e}")
        finalize_eval_run(run_id, status="failed", finished_at=_utcnow_iso(), error=str(e))
        raise HTTPException(status_code=500, detail=f"Eval run failed: {str(e)}")

    result["id"] = run_id
    return result


@router.post("/calibrate")
async def trigger_calibration(request: CalibrateRequest):
    """Conformally calibrate the abstention threshold from a labelled dataset.

    Runs the live pipeline (needs a live LLM API key + an ingested corpus), so it is
    an operator action. Persists the active calibration for should_abstain.
    """
    dataset_path = request.dataset_path or DEFAULT_CALIBRATION_DATASET
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Calibration dataset not found: {dataset_path}")
    if not (0.0 < request.alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in (0, 1).")

    # Pre-flight: don't burn LLM calls on a dataset that's obviously too small
    # for the requested alpha — check the labelled answerable count before
    # running anything live.
    items = _load_dataset(dataset_path)
    answerable_count = sum(1 for item in items if not item.get("expected_abstain", False))
    required = min_n_for_alpha(request.alpha)
    if answerable_count < required and not request.force:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset has {answerable_count} answerable (expected_abstain=false) item(s), but "
                f"α={request.alpha} needs at least {required} to produce a real threshold — the "
                f"calibration would come back INSUFFICIENT_N. Lower alpha, add more answerable "
                f"calibration items, or pass force=true to run it anyway (it will be recorded but "
                f"not applied as the active threshold)."
            ),
        )

    try:
        result = await run_calibration(dataset_path, alpha=request.alpha, run_label=request.run_label, force=request.force)
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")
    return result


@router.get("/calibration")
async def get_calibration():
    """The stored conformal calibration (if any), and whether it's actually
    trusted for abstention decisions right now.

    `effective` is False whenever `status != CALIBRATED` — a stored
    INSUFFICIENT_N/NO_DATA calibration (or a legacy file with no status at
    all) never overrides the fixed ceiling, regardless of what its numeric
    `threshold` field says. See verification/conformal.py::load_active_threshold.
    """
    calib = load_active_calibration()
    if calib is None:
        return {"active": False, "effective": False, "message": "No calibration set — abstention uses the fixed ceiling."}
    # +inf isn't JSON-serialisable; expose it as a string.
    if calib.get("threshold") == float("inf"):
        calib["threshold"] = "inf"
    effective = calib.get("status") == "CALIBRATED"
    return {
        "active": True,
        "effective": effective,
        "calibration": calib,
        "message": None if effective else (
            f"Stored calibration status is {calib.get('status', 'UNKNOWN')} — abstention is using the "
            f"fixed ceiling, not this calibration's threshold."
        ),
    }


@router.get("/runs")
async def get_eval_runs():
    """List past run summaries — id, label, timing, status, and aggregate metrics — for trend charting."""
    runs = list_eval_runs()
    for r in runs:
        try:
            r["metrics"] = json.loads(r.pop("metrics_json"))
        except Exception:
            r["metrics"] = {}
    return runs


@router.get("/runs/{run_id}")
async def get_eval_run_detail(run_id: int):
    """Full run detail, including provenance and per-query breakdown."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found.")
    try:
        run["metrics"] = json.loads(run.pop("metrics_json"))
    except Exception:
        run["metrics"] = {}
    try:
        run["per_query"] = json.loads(run.pop("per_query_json"))
    except Exception:
        run["per_query"] = []
    for field in ("run_config_json", "corpus_manifest_json", "active_calibration_json", "model_identities_json"):
        raw = run.pop(field, None)
        try:
            run[field[:-5]] = json.loads(raw) if raw else None
        except Exception:
            run[field[:-5]] = None
    return run


@router.get("/runs/{run_id}/items")
async def get_eval_run_items(run_id: int):
    """Per-item durable records for a run (eval_items) — the crash-safe
    source, independent of whether the run's final aggregate step ever
    completed. Useful for inspecting a 'failed'/'running' run's partial progress."""
    from shared.database import list_eval_items
    if not get_eval_run(run_id):
        raise HTTPException(status_code=404, detail="Eval run not found.")
    items = list_eval_items(run_id)
    for item in items:
        try:
            item["result"] = json.loads(item.pop("result_json"))
        except Exception:
            item["result"] = None
    return items
