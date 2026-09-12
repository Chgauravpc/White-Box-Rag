"""
schema.py — eval dataset schema (v2) + validator + v1 compatibility loader.

v1 (the current golden_dataset.jsonl / calibration_dataset.jsonl shape) only
carries `expected_section_ids` (often empty) and `expected_abstain`, which is
why retrieval metrics are dead code today and only the abstention decision
can be scored at all. v2 adds graded relevance (for nDCG, not just hit/miss)
and per-claim reference labels (for real hallucination-detector accuracy) —
the ground truth backend/eval/scoring.py needs. Populating v2 fields at scale
is W3.2 in the benchmark-readiness roadmap; this module only defines and
validates the shape, and upgrades v1 items so existing datasets keep working
during the transition.
"""

from typing import Any

SCHEMA_VERSION = 2

QUERY_TYPES = {"factoid", "multi_hop", "summary", "numeric", "unanswerable", "near_miss", "paraphrase_of", ""}
ANSWERABILITY = {"ANSWERABLE", "UNANSWERABLE", "PARTIAL"}
CLAIM_LABELS = {"SUPPORTED", "REFUTED", "NEI"}
SPLITS = {"dev", "test", "calibration", ""}

REQUIRED_FIELDS = ("id", "query", "expected_abstain")


def _is_v2_shape(item: dict) -> bool:
    return item.get("schema_version") == SCHEMA_VERSION


def upgrade_v1_item(item: dict) -> dict:
    """Best-effort upgrade of a v1 item to the v2 shape.

    v1's `expected_section_ids` are bare section IDs, not canonical chunk
    keys (shared/chunk_key.py) — they can collide across documents and can't
    be resolved to a graded relevance judgment without re-running retrieval.
    They're carried forward as ungraded (grade=1) `relevant_chunk_keys` so
    existing datasets keep working, but this is a compatibility shim, not a
    substitute for re-labeling with real chunk keys (W3.2).
    """
    if _is_v2_shape(item):
        return item
    upgraded = dict(item)
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded.setdefault("corpus_id", "")
    upgraded.setdefault("query_type", "")
    section_ids = item.get("expected_section_ids", [])
    upgraded.setdefault("relevant_chunk_keys", [{"key": sid, "grade": 1} for sid in section_ids])
    upgraded.setdefault("reference_answer", "")
    upgraded.setdefault("reference_claims", [])
    upgraded.setdefault("answerability", "UNANSWERABLE" if item.get("expected_abstain") else "ANSWERABLE")
    upgraded.setdefault("difficulty", "")
    upgraded.setdefault("split", "")
    upgraded.setdefault("source", "hand")
    upgraded.setdefault("license", "")
    upgraded.setdefault("provenance", {})
    upgraded.setdefault("notes", item.get("notes", ""))
    return upgraded


def validate_item(item: dict) -> list[str]:
    """Return a list of validation error strings; empty means valid.
    Only enforces the fields scoring.py actually depends on today — this is
    intentionally not a strict schema lock, since W3.2 will still be
    iterating on the exact shape.
    """
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"missing required field '{field}'")

    if "query_type" in item and item["query_type"] not in QUERY_TYPES:
        errors.append(f"invalid query_type '{item['query_type']}' — must be one of {sorted(QUERY_TYPES)}")

    if "answerability" in item and item["answerability"] and item["answerability"] not in ANSWERABILITY:
        errors.append(f"invalid answerability '{item['answerability']}' — must be one of {sorted(ANSWERABILITY)}")

    if "split" in item and item["split"] not in SPLITS:
        errors.append(f"invalid split '{item['split']}' — must be one of {sorted(SPLITS)}")

    for claim in item.get("reference_claims", []):
        if not isinstance(claim, dict) or "text" not in claim:
            errors.append("reference_claims entries must be objects with a 'text' field")
            continue
        label = claim.get("label")
        if label is not None and label not in CLAIM_LABELS:
            errors.append(f"invalid reference_claims label '{label}' — must be one of {sorted(CLAIM_LABELS)}")

    for rel in item.get("relevant_chunk_keys", []):
        if not isinstance(rel, dict) or "key" not in rel:
            errors.append("relevant_chunk_keys entries must be objects with a 'key' field")
            continue
        grade = rel.get("grade")
        if grade is not None and not (isinstance(grade, int) and 0 <= grade <= 3):
            errors.append(f"invalid relevant_chunk_keys grade '{grade}' — must be an int in [0, 3]")
        text_hash = rel.get("text_sha256")
        if text_hash is not None and not (isinstance(text_hash, str) and len(text_hash) == 64):
            errors.append(f"invalid relevant_chunk_keys text_sha256 '{text_hash}' — must be a 64-char sha256 hex digest")

    return errors


def load_and_validate(items: list[dict], upgrade: bool = True) -> tuple[list[dict], dict[Any, list[str]]]:
    """Upgrade v1 items (if `upgrade`) and validate every item.

    Returns (items, errors_by_id) — `errors_by_id` only contains entries for
    items that failed validation, so `not errors_by_id` means everything's
    clean. Never raises: a bad dataset should be reported, not crash the
    loader (mirrors harness.py's per-line JSON guard).
    """
    processed = [upgrade_v1_item(item) if upgrade else item for item in items]
    errors_by_id = {}
    for item in processed:
        errs = validate_item(item)
        if errs:
            errors_by_id[item.get("id", "<missing id>")] = errs
    return processed, errors_by_id


def relevant_chunk_keys_as_dict(item: dict, manifest: dict | None = None) -> dict[str, int]:
    """Convert an item's `relevant_chunk_keys` list into the {chunk_key: grade}
    shape `eval/scoring.py::retrieval_metrics` expects.

    A chunk_key encodes a position, so it moves whenever the corpus is
    re-chunked. When a `manifest` is supplied, labels carrying `text_sha256`
    are resolved by content first and re-pointed at the key the text lives at
    now — so a re-chunk degrades the label set visibly (via
    corpus_manifest.resolve_labels) instead of silently scoring against the
    wrong passages. Without a manifest, behaviour is unchanged.
    """
    graded = {}
    by_hash = {}
    if manifest:
        from shared.corpus_manifest import _chunk_index
        _, by_hash = _chunk_index(manifest)
    for rel in item.get("relevant_chunk_keys", []):
        key, grade = rel.get("key"), rel.get("grade", 1)
        text_hash = rel.get("text_sha256")
        if by_hash and text_hash and text_hash in by_hash:
            for live_key in by_hash[text_hash]:
                graded[live_key] = grade
        elif key:
            graded[key] = grade
    return graded
