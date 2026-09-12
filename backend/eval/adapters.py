"""
adapters.py — convert external hallucination benchmarks into detector items.

`eval/detector.py` scores rows of `(claim, premise, label)`. Public benchmarks
already carry exactly that information, in their own shapes. These converters
translate; they deliberately do **not** download anything. Which benchmarks to
use is a licensing and prioritization decision for the project owner, and a
converter that silently fetches several gigabytes on first call is not a thing
anyone should have to discover at runtime.

Each converter takes already-parsed rows and returns `(items, report)`, where
`report` says what was dropped and why. Nothing is silently discarded: a
benchmark row that cannot become a well-formed detector item is a fact about
coverage, and a number computed over an unknown subset of a benchmark is not
comparable to anyone else's number on that benchmark.

---

**The FEVER NEI trap** — the reason this module exists rather than a ten-line
loop.

FEVER's NOT ENOUGH INFO claims have *no gold evidence by construction*: the
annotation says "the corpus does not support or refute this", so the evidence
field is empty. Feed those to the detector with an empty premise and the
empty-premise guard fires, the claim is flagged, and it scores as a correct
catch — 100% recall on NEI, measuring nothing at all. The detector was never
run; the item was decided by the absence of input.

So NEI rows without resolvable evidence are **skipped by default** and counted
in the report. Scoring FEVER's NEI class honestly requires supplying candidate
evidence (what a retriever actually returned for that claim), which is a
retrieval-plane concern, not something this converter can invent. Pass
`keep_unresolvable_nei=True` only if you have read this paragraph and want the
items anyway — they will be marked `premise_source="none"`.
"""

import logging

logger = logging.getLogger(__name__)

FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "NOT_ENOUGH_INFO"}


def _item(id_, claim, premise, label, **extra):
    return dict({"id": str(id_), "claim": claim, "premise": premise, "label": label}, **extra)


# ── HaluEval ────────────────────────────────────────────────

def from_halueval(rows, task: str = "qa", id_prefix: str = "halueval") -> tuple[list[dict], dict]:
    """Convert HaluEval rows into detector items.

    HaluEval is self-contained — each row carries the knowledge/context plus
    both a correct answer and a hallucinated one — so each row yields **two**
    items, one SUPPORTED and one REFUTED against the same premise. That
    pairing is the point: it controls for premise difficulty, so a difference
    in score between the two is attributable to the claim, not to the
    evidence. It also makes the resulting set balanced by construction.

    `task` selects the field names: "qa" (knowledge), "dialogue"
    (dialogue_history + knowledge), or "summarization" (document).
    """
    premise_fields = {
        "qa": ("knowledge",),
        "dialogue": ("knowledge", "dialogue_history"),
        "summarization": ("document",),
    }.get(task)
    if premise_fields is None:
        raise ValueError(f"unknown HaluEval task {task!r} — expected qa, dialogue or summarization")

    right_field, halluc_field = {
        "qa": ("right_answer", "hallucinated_answer"),
        "dialogue": ("right_response", "hallucinated_response"),
        "summarization": ("right_summary", "hallucinated_summary"),
    }[task]

    items, skipped = [], {"no_premise": 0, "no_claim": 0}
    for i, row in enumerate(rows):
        premise = " ".join(str(row.get(f, "")).strip() for f in premise_fields).strip()
        if not premise:
            skipped["no_premise"] += 1
            continue
        base = row.get("id", i)
        for field, label, suffix in (
            (right_field, "SUPPORTED", "pos"),
            (halluc_field, "REFUTED", "neg"),
        ):
            claim = str(row.get(field, "")).strip()
            if not claim:
                skipped["no_claim"] += 1
                continue
            items.append(_item(
                f"{id_prefix}-{task}-{base}-{suffix}", claim, premise, label,
                source=f"halueval/{task}", premise_source="gold",
            ))
    return items, {"n_rows": len(rows), "n_items": len(items), "skipped": skipped}


# ── FEVER ───────────────────────────────────────────────────

def _fever_evidence_keys(row) -> list[tuple]:
    """Flatten FEVER's nested evidence into unique (page, sentence_id) pairs.

    The shape is [[[annotation_id, evidence_id, page, sentence_id], ...], ...]
    — the outer list is annotators, the inner one is a conjunction of
    sentences that together justify the claim. Nulls appear for NEI rows.
    """
    keys, seen = [], set()
    for group in row.get("evidence") or []:
        for entry in group or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            page, sent_id = entry[2], entry[3]
            if page is None or sent_id is None:
                continue
            key = (page, sent_id)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def from_fever(
    rows,
    resolve_evidence=None,
    keep_unresolvable_nei: bool = False,
    id_prefix: str = "fever",
) -> tuple[list[dict], dict]:
    """Convert FEVER claim rows into detector items.

    FEVER ships claims and *pointers* to evidence (wikipedia page + sentence
    index); the sentence text lives in a separate wiki dump. So this takes
    `resolve_evidence(page, sentence_id) -> str`, which keeps the (large,
    licence-encumbered) corpus out of this repo and makes the converter
    testable. Evidence sentences for one claim are joined in annotation order.

    NEI handling: see this module's docstring. NEI claims have no gold
    evidence by construction, so scoring them against an empty premise
    measures the empty-premise guard, not the detector.
    """
    items = []
    report = {
        "n_rows": len(rows), "skipped_bad_label": 0, "skipped_unresolvable_evidence": 0,
        "skipped_nei_without_evidence": 0, "kept_nei_without_evidence": 0,
    }

    for i, row in enumerate(rows):
        label_raw = str(row.get("label", "")).strip().upper()
        if label_raw not in FEVER_LABELS:
            report["skipped_bad_label"] += 1
            continue
        claim = str(row.get("claim", "")).strip()
        if not claim:
            report["skipped_bad_label"] += 1
            continue

        is_nei = label_raw in {"NOT ENOUGH INFO", "NOT_ENOUGH_INFO"}
        keys = _fever_evidence_keys(row)

        sentences = []
        if keys and resolve_evidence is not None:
            for page, sent_id in keys:
                try:
                    text = (resolve_evidence(page, sent_id) or "").strip()
                except Exception as exc:
                    logger.warning("Evidence lookup failed for %s#%s: %s", page, sent_id, exc)
                    text = ""
                if text:
                    sentences.append(text)
        premise = " ".join(sentences).strip()

        if not premise:
            if is_nei:
                if not keep_unresolvable_nei:
                    report["skipped_nei_without_evidence"] += 1
                    continue
                report["kept_nei_without_evidence"] += 1
            else:
                # A SUPPORTS/REFUTES claim whose evidence could not be
                # resolved is unusable: the label asserts a relationship to
                # text we do not have.
                report["skipped_unresolvable_evidence"] += 1
                continue

        items.append(_item(
            f"{id_prefix}-{row.get('id', i)}", claim, premise, label_raw,
            source="fever", premise_source="gold" if premise else "none",
            evidence_keys=[list(k) for k in keys],
        ))

    report["n_items"] = len(items)
    return items, report


# ── Generic ─────────────────────────────────────────────────

def from_generic(
    rows, claim_field: str, premise_field: str, label_field: str,
    id_field: str | None = None, id_prefix: str = "generic",
) -> tuple[list[dict], dict]:
    """Field-mapped conversion, for a benchmark with no dedicated converter.

    Labels still pass through `detector.normalize_label`, so an unrecognized
    spelling is reported rather than coerced into a class.
    """
    from eval.detector import normalize_label

    items, skipped = [], {"missing_field": 0, "unknown_label": 0}
    for i, row in enumerate(rows):
        claim = str(row.get(claim_field, "")).strip()
        premise = str(row.get(premise_field, "")).strip()
        if not claim:
            skipped["missing_field"] += 1
            continue
        if not normalize_label(row.get(label_field)):
            skipped["unknown_label"] += 1
            continue
        ident = row.get(id_field, i) if id_field else i
        items.append(_item(
            f"{id_prefix}-{ident}", claim, premise, str(row.get(label_field)),
            source=id_prefix, premise_source="gold" if premise else "none",
        ))
    return items, {"n_rows": len(rows), "n_items": len(items), "skipped": skipped}


# ── Output ──────────────────────────────────────────────────

def write_detector_dataset(items: list[dict], path: str) -> str:
    """Write detector items as JSONL, validating each one first.

    Validation here rather than at read time means a malformed converter
    output is caught where it was produced.
    """
    import json

    from eval.detector import validate_detector_item

    bad = {it.get("id"): errs for it in items if (errs := validate_detector_item(it))}
    if bad:
        raise ValueError(f"refusing to write {len(bad)} invalid detector item(s): {list(bad.items())[:3]}")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path
