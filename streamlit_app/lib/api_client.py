"""
api_client.py — thin requests wrapper around the FastAPI backend.

One function per backend endpoint, mirroring what the old React
`frontend/src/lib/api.ts` did. Centralizes error handling in `_request()` so
individual pages don't each need their own try/except boilerplate.

Streamlit runs as a separate process from the backend (no dev-proxy like the
old Vite setup) — BACKEND_API_URL must point at the full backend base URL.
"""

import os
from typing import Any, Optional

import requests

BASE_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000/api")

SHORT_TIMEOUT = 15
LONG_TIMEOUT = 180  # /query and /brd/validate run a multi-model pipeline (several Gemini calls)


class ApiError(Exception):
    """Raised for any failed API call — network error, timeout, or non-2xx response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _request(method: str, path: str, timeout: int = SHORT_TIMEOUT, **kwargs) -> Any:
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise ApiError(f"Could not reach the backend at {BASE_URL}. Is `uvicorn gateway:app` running? ({e})")
    except requests.exceptions.Timeout:
        raise ApiError(f"Request to {path} timed out after {timeout}s.")
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Request to {path} failed: {e}")

    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(f"{path} -> {resp.status_code}: {detail}", status_code=resp.status_code)

    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.content


# ── Health ────────────────────────────────────────────────────────────────

def health_check() -> dict:
    """/health lives outside the /api prefix, so it needs its own URL."""
    health_url = BASE_URL.rsplit("/api", 1)[0] + "/health"
    try:
        resp = requests.get(health_url, timeout=SHORT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Backend health check failed at {health_url}: {e}")


# ── Ingestion ─────────────────────────────────────────────────────────────

def ingest_document(filename: str, file_bytes: bytes, publication: str, edition_date: str = "") -> dict:
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {"publication": publication, "edition_date": edition_date}
    return _request("POST", "/ingest", timeout=LONG_TIMEOUT, files=files, data=data)


def list_documents() -> list:
    return _request("GET", "/documents")


def list_sections(publication: str, edition: str) -> list:
    return _request("GET", f"/sections/{publication}/{edition}")


# ── Query (real RAG pipeline) ────────────────────────────────────────────

def query_rag(query: str, filters: Optional[dict] = None) -> dict:
    payload = {"query": query, "filters": filters}
    return _request("POST", "/query", timeout=LONG_TIMEOUT, json=payload)


# ── Verification sandbox ─────────────────────────────────────────────────

def verify_claims(answer: str, claims: list) -> dict:
    payload = {"answer": answer, "claims": claims}
    return _request("POST", "/verify/", timeout=LONG_TIMEOUT, json=payload)


def check_conflicts(publication: str, topic: str, older_date: str, older_text: str,
                     newer_date: str, newer_text: str, section_id: str) -> dict:
    payload = {
        "publication": publication, "topic": topic,
        "older_date": older_date, "older_text": older_text,
        "newer_date": newer_date, "newer_text": newer_text,
        "section_id": section_id,
    }
    return _request("POST", "/verify/check-conflicts", timeout=LONG_TIMEOUT, json=payload)


# ── BRD Validator ─────────────────────────────────────────────────────────

def get_sample_brd() -> dict:
    return _request("GET", "/brd/sample")


def upload_brd(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes)}
    return _request("POST", "/brd/upload", timeout=LONG_TIMEOUT, files=files)


def validate_brd(requirements: list[str], source_filename: Optional[str] = None) -> dict:
    payload = {"requirements": requirements, "source_filename": source_filename}
    return _request("POST", "/brd/validate", timeout=LONG_TIMEOUT, json=payload)


def list_brd_runs() -> list:
    return _request("GET", "/brd/runs")


def get_brd_run(run_id: int) -> dict:
    return _request("GET", f"/brd/runs/{run_id}")


# ── Audit Trail ───────────────────────────────────────────────────────────

def list_audit_logs() -> list:
    return _request("GET", "/audit/logs")


def get_audit_report(audit_id: int) -> dict:
    return _request("GET", f"/audit/{audit_id}")


def download_audit(audit_id: int) -> bytes:
    return _request("GET", f"/audit/{audit_id}/download")


def verify_audit_integrity() -> dict:
    """Tamper-evident check: walk the audit hash-chain and report if it's intact."""
    return _request("GET", "/audit/verify-integrity")


# ── Governance: Human-in-the-loop review ──────────────────────────────────

def list_review_queue() -> dict:
    return _request("GET", "/review/queue")


def resolve_review(audit_id: int, reviewer: str, action: str, note: str = "") -> dict:
    payload = {"reviewer": reviewer, "action": action, "note": note}
    return _request("POST", f"/review/{audit_id}/resolve", json=payload)


def get_review_history(audit_id: int) -> dict:
    return _request("GET", f"/review/{audit_id}/history")


# ── Evaluation Harness ────────────────────────────────────────────────────

def run_eval(dataset_path: Optional[str] = None, run_label: str = "") -> dict:
    payload = {"dataset_path": dataset_path, "run_label": run_label}
    return _request("POST", "/eval/run", timeout=LONG_TIMEOUT, json=payload)


def list_eval_runs() -> list:
    return _request("GET", "/eval/runs")


def get_eval_run(run_id: int) -> dict:
    return _request("GET", f"/eval/runs/{run_id}")


def calibrate_conformal(alpha: float = 0.1, dataset_path: Optional[str] = None, run_label: str = "") -> dict:
    payload = {"alpha": alpha, "dataset_path": dataset_path, "run_label": run_label}
    return _request("POST", "/eval/calibrate", timeout=LONG_TIMEOUT, json=payload)


def get_active_calibration() -> dict:
    return _request("GET", "/eval/calibration")
