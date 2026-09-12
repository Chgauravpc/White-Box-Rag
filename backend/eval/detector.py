"""
detector.py — the detector evaluation plane.

The end-to-end harness (`eval/harness.py`) measures retrieval, generation,
verification and governance all at once, so a change in any one of them moves
the headline number and you cannot say which. The detector plane isolates the
one question "given a claim and its evidence, does this system correctly
decide the claim is unsupported?" — no retrieval, no generation, no LLM call,
no database. That is what makes it cheap enough to run over thousands of
items, and it is the shape external hallucination benchmarks already ship in:
FEVER, HaluEval and RAGTruth all supply (claim, evidence, label) directly.

Two deliberate choices:

* **The premise is never normalized by default** (`profile="none"`). The
  normalizer profiles exist to strip PDF artifacts from *this* system's
  corpus; running them over a benchmark's own evidence text would mean
  measuring a number nobody else can reproduce. `shared/text_normalize.py`
  documents `none` as existing for exactly this.

* **The prediction is the shipped decision rule.** `y_pred_flagged` comes from
  `verification/mitigation.py::_should_strip`, the same function the live
  pipeline uses to decide whether a claim survives into the mitigated answer.
  Re-implementing "flagged" here would measure a detector this repo does not
  actually ship.

Positive class: a claim is "hallucinated" when the evidence does not support
it, i.e. label != SUPPORTED. That deliberately groups REFUTED (the evidence
contradicts the claim) with NEI (the evidence is silent), because the system's
job is to not assert either. They are reported separately too — the label is
passed through as `error_types`, so `recall_by_error_type` breaks the number
down into "catches X% of contradictions, Y% of unsupported" rather than
hiding the difference in one aggregate.
"""

import json
import logging

from eval import scoring
from shared.models import NLIVerdict, VerificationResult
from shared.xai_matrices import verify_claims_batch
from verification.mitigation import _should_strip

logger = logging.getLogger(__name__)

# The label vocabulary, matching eval/schema.py::CLAIM_LABELS.
SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NEI"
DETECTOR_LABELS = frozenset({SUPPORTED, REFUTED, NEI})

# Benchmark label spellings mapped onto ours. FEVER uses SUPPORTS/REFUTES/
# NOT ENOUGH INFO; HaluEval and RAGTruth are binary.
LABEL_ALIASES = {
    "SUPPORTS": SUPPORTED, "SUPPORTED": SUPPORTED, "ENTAILMENT": SUPPORTED,
    "TRUE": SUPPORTED, "FACTUAL": SUPPORTED, "OK": SUPPORTED,
    "REFUTES": REFUTED, "REFUTED": REFUTED, "CONTRADICTION": REFUTED,
    "FALSE": REFUTED, "HALLUCINATED": REFUTED,
    "NOT ENOUGH INFO": NEI, "NOT_ENOUGH_INFO": NEI, "NEI": NEI,
    "NEUTRAL": NEI, "UNVERIFIABLE": NEI,
}

REQUIRED_FIELDS = ("id", "claim", "premise", "label")


def normalize_label(label) -> str:
    """Map a benchmark's label spelling onto ours; "" if unrecognized."""
    if label is None:
        return ""
    return LABEL_ALIASES.get(str(label).strip().upper(), "")


def validate_detector_item(item: dict) -> list[str]:
    """Return validation errors; empty means valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"missing required field '{field}'")
    if "label" in item and not normalize_label(item["label"]):
        errors.append(
            f"unrecognized label {item['label']!r} — expected one of {sorted(DETECTOR_LABELS)} "
            f"or a known benchmark alias ({sorted(LABEL_ALIASES)[:4]}...)"
        )
    if "claim" in item and not str(item.get("claim", "")).strip():
        errors.append("claim is empty")
    return errors


def load_detector_dataset(path: str) -> tuple[list[dict], dict]:
    """Load a JSONL detector dataset.

    Mirrors `harness._load_dataset`'s per-line guard: one malformed line is
    reported with its line number and skipped, never allowed to kill a run
    before a single item is scored.
    """
    items, errors_by_id = [], {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL at %s:%d — %s", path, lineno, exc)
                errors_by_id[f"<line {lineno}>"] = [f"malformed JSON: {exc}"]
                continue
            errs = validate_detector_item(item)
            if errs:
                errors_by_id[item.get("id", f"<line {lineno}>")] = errs
                continue
            items.append(item)
    return items, errors_by_id


def run_detector_eval(items: list[dict], profile: str = "none") -> dict:
    """Score the claim-verification stage against per-claim labels.

    Pure: no LLM, no retrieval, no database. `profile` is the premise
    normalizer; leave it at "none" for external benchmarks.
    """
    if not items:
        return {
            "n": 0,
            "premise_normalizer": profile,
            "metrics": scoring.detector_metrics([], []),
            "verdict_confusion": {},
            "per_item": [],
        }

    raw = verify_claims_batch([(it["claim"], it.get("premise", "")) for it in items], profile=profile)

    y_true, y_pred, scores_, error_types, per_item = [], [], [], [], []
    verdict_confusion: dict[str, dict[str, int]] = {}

    for item, result in zip(items, raw):
        label = normalize_label(item["label"])
        verification = VerificationResult(
            claim_text=result["claim_text"],
            verdict=NLIVerdict(result["verdict"]),
            entailment_score=result["entailment_score"],
            evidence_status=result.get("evidence_status", "ok"),
        )
        flagged = _should_strip(verification)

        y_true.append(label != SUPPORTED)
        y_pred.append(flagged)
        # Oriented so HIGHER means more likely hallucinated, per
        # scoring.detector_metrics's contract.
        scores_.append(1.0 - result["entailment_score"])
        # A dataset carrying a finer-grained error taxonomy (RAGTruth's
        # Evident/Subtle x Conflict/Baseless Info) gets its recall broken down
        # by that instead of by the coarse label. "Catches 94% of evident
        # conflicts, 41% of subtle baseless info" is what makes the number
        # actionable; falling back to the label keeps simpler datasets working.
        error_types.append(item.get("error_type") or label)

        verdict_confusion.setdefault(label, {}).setdefault(result["verdict"], 0)
        verdict_confusion[label][result["verdict"]] += 1

        per_item.append({
            "id": item.get("id"),
            "label": label,
            "verdict": result["verdict"],
            "entailment_score": result["entailment_score"],
            "evidence_status": result.get("evidence_status", "ok"),
            "error_type": item.get("error_type") or "",
            "flagged": flagged,
            "correct": (label != SUPPORTED) == flagged,
        })

    return {
        "n": len(items),
        "premise_normalizer": profile,
        "metrics": scoring.detector_metrics(y_true, y_pred, scores_, error_types),
        # Raw NLI verdict vs. gold label — separates "the decision rule is
        # miscalibrated" from "the NLI model is wrong", which a single
        # precision/recall pair cannot distinguish.
        "verdict_confusion": verdict_confusion,
        "per_item": per_item,
    }


def run_detector_eval_from_file(path: str, profile: str = "none") -> dict:
    items, errors_by_id = load_detector_dataset(path)
    result = run_detector_eval(items, profile=profile)
    result["dataset_path"] = path
    result["skipped_items"] = errors_by_id
    return result
