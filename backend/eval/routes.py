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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval.harness import run_eval, run_calibration
from shared.database import insert_eval_run, list_eval_runs, get_eval_run
from verification.conformal import load_active_calibration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["Evaluation"])

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
DEFAULT_CALIBRATION_DATASET = os.path.join(os.path.dirname(__file__), "calibration_dataset.jsonl")


class RunEvalRequest(BaseModel):
    dataset_path: str | None = None
    run_label: str = ""


class CalibrateRequest(BaseModel):
    dataset_path: str | None = None
    alpha: float = 0.1
    run_label: str = ""


@router.post("/run")
async def trigger_eval_run(request: RunEvalRequest):
    """Runs the golden dataset through the real pipeline (synchronous — datasets
    are expected to be tens of queries; runs with bounded concurrency internally).
    """
    dataset_path = request.dataset_path or DEFAULT_DATASET_PATH
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")

    try:
        result = await run_eval(dataset_path, run_label=request.run_label)
    except Exception as e:
        logger.error(f"Eval run failed: {e}")
        raise HTTPException(status_code=500, detail=f"Eval run failed: {str(e)}")

    run_id = insert_eval_run(
        run_label=result["run_label"],
        dataset_path=result["dataset_path"],
        started_at=result["started_at"],
        finished_at=result["finished_at"],
        num_queries=result["num_queries"],
        metrics_json=json.dumps(result["metrics"]),
        per_query_json=json.dumps(result["per_query"], default=str),
    )
    result["id"] = run_id
    return result


@router.post("/calibrate")
async def trigger_calibration(request: CalibrateRequest):
    """Conformally calibrate the abstention threshold from a labelled dataset.

    Runs the live pipeline (needs GEMINI_API_KEY + an ingested corpus), so it is
    an operator action. Persists the active calibration for should_abstain.
    """
    dataset_path = request.dataset_path or DEFAULT_CALIBRATION_DATASET
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Calibration dataset not found: {dataset_path}")
    if not (0.0 < request.alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in (0, 1).")

    try:
        result = await run_calibration(dataset_path, alpha=request.alpha, run_label=request.run_label)
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")
    return result


@router.get("/calibration")
async def get_calibration():
    """The active conformal calibration, or a fallback notice if none is set."""
    calib = load_active_calibration()
    if calib is None:
        return {"active": False, "message": "No calibration set — abstention uses the fixed ceiling."}
    # +inf isn't JSON-serialisable; expose it as a string.
    if calib.get("threshold") == float("inf"):
        calib["threshold"] = "inf"
    return {"active": True, "calibration": calib}


@router.get("/runs")
async def get_eval_runs():
    """List past run summaries — id, label, timing, and aggregate metrics — for trend charting."""
    runs = list_eval_runs()
    for r in runs:
        try:
            r["metrics"] = json.loads(r.pop("metrics_json"))
        except Exception:
            r["metrics"] = {}
    return runs


@router.get("/runs/{run_id}")
async def get_eval_run_detail(run_id: int):
    """Full run detail, including per-query breakdown."""
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
    return run
