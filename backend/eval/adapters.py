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
import re

logger = logging.getLogger(__name__)

FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "NOT_ENOUGH_INFO"}

# Mirrors eval/detector.py's vocabulary. Declared here rather than imported at
# module scope because every other cross-module reference in this file is a
# deliberate function-local import.
SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NEI"


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


# ── Validity ────────────────────────────────────────────────

def dataset_validity_report(items: list[dict]) -> dict:
    """Check a converted dataset for shortcuts that would invalidate a score.

    Written after a real incident. HaluEval QA's `right_answer` is an *answer*
    ("Sidney Lumet") while its `hallucinated_answer` is a *sentence* ("First
    for Women was started first."). Converted naively, the supported class had
    a median claim length of 2 words and the refuted class 10 — separable by
    length alone, with no reference to the evidence at all. Worse for an
    NLI-based detector specifically: a bare noun phrase is not a proposition,
    so nothing can entail it, and correct answers were returned NEUTRAL and
    flagged as hallucinations. The resulting F1 described the benchmark's
    shape, not the detector.

    A number computed on a dataset with a shortcut this strong is not a
    measurement, so this is reported loudly rather than left to be discovered
    in the confusion matrix.
    """
    import statistics

    from eval.detector import normalize_label

    by_label: dict[str, list[int]] = {}
    for it in items:
        label = normalize_label(it.get("label"))
        if label:
            by_label.setdefault(label, []).append(len(str(it.get("claim", "")).split()))

    lengths = {
        label: {
            "n": len(v),
            "median_words": statistics.median(v) if v else 0,
            "mean_words": round(statistics.mean(v), 1) if v else 0,
            "pct_under_4_words": round(100 * sum(1 for x in v if x < 4) / len(v), 1) if v else 0,
        }
        for label, v in sorted(by_label.items())
    }

    warnings = []
    medians = {label: st["median_words"] for label, st in lengths.items() if st["n"]}
    if len(medians) >= 2:
        lo, hi = min(medians.values()), max(medians.values())
        # 3x is well past anything topic variation explains; at that point the
        # label is predictable from claim length without reading the premise.
        if lo > 0 and hi / lo >= 3:
            warnings.append(
                f"claim length differs {hi/lo:.1f}x across labels (medians {medians}) — "
                f"the label is largely predictable from length alone, so any score on this "
                f"dataset may measure the shortcut rather than the detector"
            )
        elif lo == 0 and hi > 0:
            warnings.append(f"one label class has zero-length claims (medians {medians})")

    for label, st in lengths.items():
        if st["n"] and st["pct_under_4_words"] >= 50:
            warnings.append(
                f"{st['pct_under_4_words']:.0f}% of '{label}' claims are under 4 words — these are "
                f"likely entity fragments, not propositions. An NLI detector cannot entail a "
                f"fragment, so they will be scored NEUTRAL regardless of the evidence"
            )

    if len(lengths) < 2:
        warnings.append("fewer than two label classes present — precision/recall degenerate "
                        "and AUROC is undefined")

    return {"n_items": len(items), "claim_length_by_label": lengths, "warnings": warnings}


# ── RAGTruth ────────────────────────────────────────────────
#
# RAGTruth is the closest public benchmark to what this system actually does:
# it annotates hallucinated SPANS inside LLM responses generated from retrieved
# context. Two things follow from that shape.
#
# First, spans must become sentence labels, because this pipeline verifies
# claims (sentences), not spans. The propagation rule is stated explicitly
# rather than buried: **a sentence is hallucinated if its character range
# overlaps any annotated span.** That is a real methodological choice — it
# labels a whole sentence positive even when only three words of it are
# unsupported — and it is reported in every converted item
# (`propagation_rule`) so a number produced from this data can be compared
# against someone else's only when their rule matches.
#
# Second, RAGTruth's four span types (Evident/Subtle x Conflict/Baseless Info)
# are carried through as `error_type`, which `scoring.detector_metrics` breaks
# down into per-type recall. "Catches 94% of evident conflicts, 41% of subtle
# baseless info" is a far more useful statement than one F1, and the subtle
# classes are exactly where single-premise NLI is expected to be weakest.

RAGTRUTH_LABEL_MAP = {
    # Contradicted by the context.
    "Evident Conflict": REFUTED,
    "Subtle Conflict": REFUTED,
    # Not present in the context at all — unsupported rather than contradicted,
    # which is NEI, not REFUTED. Both land in the detector's positive class
    # (label != SUPPORTED) but the distinction is preserved for the breakdown.
    "Evident Baseless Info": NEI,
    "Subtle Baseless Info": NEI,
}

# Sentence boundary: terminal punctuation followed by whitespace, or a newline.
# A regex rather than spaCy on purpose — the conversion must be deterministic
# and reproducible by anyone, without a model download, and must yield exact
# character offsets to intersect against span ranges.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


# A "sentence" consisting only of a list marker — "1.", "2)", "3." — is a
# numbering artifact, not a claim. Generated answers are full of them, and
# scoring them as claims is meaningless: nothing can entail "2.", so the model
# returns NEUTRAL and the item counts as a false positive. In the first
# RAGTruth run 126 of 1,502 scored items were bare list markers and 18% of all
# false positives were fragments of this kind.
_LIST_MARKER = re.compile(r"^\(?[0-9]+[.)\]]?$|^[-*•]$")


_LEADING_MARKER = re.compile(r"^(\(?[0-9]+[.)\]]|[-*•])\s+")


def _strip_leading_markers(spans):
    """Remove a leading "1. " or "- " from a sentence.

    `start` is advanced past the marker rather than the text being edited in
    place, so `text[start:end] == sentence` still holds — those offsets are
    what span overlap is computed against.
    """
    out = []
    for start, end, text in spans:
        m = _LEADING_MARKER.match(text)
        if m:
            start += m.end()
            text = text[m.end():]
        if text:
            out.append((start, end, text))
    return out


def _drop_list_markers(spans):
    """Remove bare list markers from the sentence list.

    Dropped rather than merged into the following sentence: merging would have
    to extend that sentence's start offset over the marker, breaking the
    invariant that `text[start:end] == sentence`, and those offsets are what
    span overlap is computed against. Dropping is safe for overlap because an
    annotation covering "1. Remove the blue field" still overlaps the range of
    "Remove the blue field"; only a span covering the digit alone would be
    lost, which does not occur.
    """
    return [(start, end, text) for start, end, text in spans if not _LIST_MARKER.match(text)]


def _raw_split(text: str) -> list[tuple[int, int, str]]:
    """Sentence spans before list markers are removed."""
    text = text or ""
    spans, pos = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        chunk = text[pos:end].strip()
        if chunk:
            start = pos + (len(text[pos:end]) - len(text[pos:end].lstrip()))
            spans.append((start, start + len(chunk), chunk))
        pos = match.end()
    tail = text[pos:].strip()
    if tail:
        start = pos + (len(text[pos:]) - len(text[pos:].lstrip()))
        spans.append((start, start + len(tail), tail))
    return spans


def split_sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) sentence triples, list-numbering artifacts removed.

    Offsets index back into `text`, i.e. text[start:end] == sentence.
    """
    return _strip_leading_markers(_drop_list_markers(_raw_split(text)))


def is_proposition(text: str, min_words: int = 4) -> bool:
    """Whether a sentence is something an NLI model can meaningfully judge.

    Entailment is a relation between propositions. A bullet fragment or a bare
    heading is not one, so no premise can entail it and it is scored NEUTRAL
    regardless of the evidence — a guaranteed false positive that says nothing
    about the detector.
    """
    return len(re.findall(r"[A-Za-z']+", text or "")) >= min_words


def _ragtruth_premise(source: dict) -> str:
    """The evidence text for one source record.

    QA carries the retrieved `passages` — the case that matches this system.
    Summary carries the article. Data2txt carries a structured business record
    rather than prose; it is serialized here, but an NLI model reasoning over a
    serialized dict is doing a different task, so it is excluded by default.
    """
    info = source.get("source_info")
    if isinstance(info, str):
        return info.strip()
    if isinstance(info, dict):
        if "passages" in info:
            return str(info["passages"]).strip()
        return "\n".join(f"{k}: {v}" for k, v in info.items()).strip()
    return ""


def from_ragtruth(
    responses, sources, task_types=("QA", "Summary"), split=None, id_prefix="ragtruth",
    min_claim_words: int = 4,
) -> tuple[list[dict], dict]:
    """Convert RAGTruth responses into sentence-level detector items.

    `sources` may be a list of source records or a {source_id: record} map.
    `task_types` defaults to QA and Summary; Data2txt is excluded because its
    premise is a structured record rather than text (pass it explicitly to
    include it). `split` filters to "train" or "test".
    """
    if not isinstance(sources, dict):
        sources = {s.get("source_id"): s for s in sources}
    wanted_tasks = set(task_types) if task_types else None

    items = []
    report = {
        "n_responses": len(responses), "used_responses": 0,
        "skipped_wrong_split": 0, "skipped_wrong_task": 0,
        "skipped_no_source": 0, "skipped_no_premise": 0, "skipped_no_sentences": 0,
        "sentences_total": 0, "sentences_hallucinated": 0,
        "skipped_list_markers": 0, "skipped_fragments": 0, "skipped_fragments_annotated": 0,
        "min_claim_words": min_claim_words,
        "by_error_type": {},
        "propagation_rule": "sentence_overlaps_any_span",
    }

    for response in responses:
        if split and response.get("split") != split:
            report["skipped_wrong_split"] += 1
            continue
        source = sources.get(response.get("source_id"))
        if source is None:
            report["skipped_no_source"] += 1
            continue
        if wanted_tasks and source.get("task_type") not in wanted_tasks:
            report["skipped_wrong_task"] += 1
            continue
        premise = _ragtruth_premise(source)
        if not premise:
            report["skipped_no_premise"] += 1
            continue

        text = response.get("response") or ""
        raw_sentences = _raw_split(text)
        sentences = _strip_leading_markers(_drop_list_markers(raw_sentences))
        report["skipped_list_markers"] += len(raw_sentences) - len(sentences)
        if not sentences:
            report["skipped_no_sentences"] += 1
            continue

        spans = [s for s in (response.get("labels") or [])
                 if isinstance(s, dict) and s.get("start") is not None and s.get("end") is not None]
        report["used_responses"] += 1

        for idx, (start, end, sentence) in enumerate(sentences):
            if not is_proposition(sentence, min_claim_words):
                # Counted, never silently dropped: excluding items changes what
                # the resulting number covers.
                report["skipped_fragments"] += 1
                if [sp for sp in spans if sp["start"] < end and start < sp["end"]]:
                    report["skipped_fragments_annotated"] += 1
                continue
            # Half-open interval intersection; a span touching only the
            # boundary is not an overlap.
            hits = [s for s in spans if s["start"] < end and start < s["end"]]
            report["sentences_total"] += 1
            if hits:
                # Most severe wins when a sentence carries several span types:
                # a Conflict is a stronger claim about the sentence than
                # Baseless Info, and reporting the weaker one would understate
                # what the detector had to catch.
                hits.sort(key=lambda s: 0 if "Conflict" in str(s.get("label_type")) else 1)
                error_type = str(hits[0].get("label_type") or "Unknown")
                label = RAGTRUTH_LABEL_MAP.get(error_type, NEI)
                report["sentences_hallucinated"] += 1
                report["by_error_type"][error_type] = report["by_error_type"].get(error_type, 0) + 1
            else:
                error_type, label = "", SUPPORTED

            items.append(_item(
                f"{id_prefix}-{response.get('id', 'x')}-{idx}", sentence, premise, label,
                source="ragtruth", premise_source="gold",
                task_type=source.get("task_type"), model=response.get("model"),
                split=response.get("split"), error_type=error_type,
                propagation_rule="sentence_overlaps_any_span",
            ))

    report["n_items"] = len(items)
    return items, report
