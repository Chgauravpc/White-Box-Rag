"""
retrieval.py — the retrieval evaluation plane.

`run_query_pipeline` calls `rag_query` unconditionally, so until now there was
no way to measure retrieval without also paying for generation: scoring
retrieval over 1,000 queries meant 1,000+ LLM calls, which is what makes a
BEIR-scale sweep unaffordable and a per-commit retrieval regression check
impossible. This plane runs `hybrid_retrieve` and scores it. Nothing else —
**zero LLM calls**, no generation, no verification, no audit write.

It deliberately reuses the v2 dataset shape (`eval/schema.py`) rather than
inventing a third format: `relevant_chunk_keys` is already defined there, so
the same dataset drives the end-to-end and retrieval planes and the two
cannot disagree about what "relevant" means.

**Label durability is the hard part, and this is where the corpus manifest
earns its keep.** A `relevant_chunk_key` encodes a position
(publication|edition|section|ordinal), so re-chunking the corpus moves it and
a label silently starts pointing at different text — the metric would keep
producing a number, just not the one you think. Pass a frozen manifest
(`shared/corpus_manifest.py`) and two things happen: labels carrying a
`text_sha256` are resolved by **content** first, so a moved chunk is followed
rather than counted as a miss; and the result carries a `label_health` block
saying what fraction of labels actually resolved. A retrieval score computed
over labels that no longer resolve is worse than no score, so it is reported
next to the number rather than left to be discovered later.
"""

import json
import logging

from eval import scoring
from eval.schema import load_and_validate, relevant_chunk_keys_as_dict
from shared.chunk_key import resolve_chunk_key

logger = logging.getLogger(__name__)


def _chunk_key_of(chunk) -> str:
    """The canonical identity of a retrieved chunk — the same resolution
    `ingestion/retriever.py` uses, so retrieved ids and labeled ids are in the
    same namespace. Anything else (a bare section_id, say) silently scores
    zero for every query."""
    return resolve_chunk_key(
        chunk.publication_name, chunk.edition_date, chunk.section_id,
        chunk.chunk_text, getattr(chunk, "chunk_key", "") or "",
    )


def load_retrieval_dataset(path: str) -> tuple[list[dict], dict]:
    """Load a v2 (or auto-upgraded v1) eval dataset as retrieval items."""
    raw = []
    parse_errors = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL at %s:%d — %s", path, lineno, exc)
                parse_errors[f"<line {lineno}>"] = [f"malformed JSON: {exc}"]
    items, errors_by_id = load_and_validate(raw)
    errors_by_id.update(parse_errors)
    return items, errors_by_id


def run_retrieval_eval(
    items: list[dict],
    top_k: int | None = None,
    manifest: dict | None = None,
    retrieve_fn=None,
) -> dict:
    """Score retrieval against graded `relevant_chunk_keys`. No LLM calls.

    `retrieve_fn(query, top_k) -> list[ChunkMetadata]` is injectable so this is
    testable without a populated ChromaDB; it defaults to `hybrid_retrieve`.
    """
    if retrieve_fn is None:
        from ingestion.retriever import hybrid_retrieve
        from shared.config import FINAL_TOP_K

        top_k = top_k or FINAL_TOP_K
        retrieve_fn = hybrid_retrieve

    per_query = []
    per_item = []
    unlabeled = 0
    label_totals = {"total": 0, "resolved": 0, "moved": 0, "missing": 0, "unverifiable": 0}

    for item in items:
        relevant = relevant_chunk_keys_as_dict(item, manifest)
        if not relevant:
            # No ground truth is not a zero — it means this item cannot speak
            # to retrieval quality at all, and must be excluded rather than
            # dragging the mean down.
            unlabeled += 1
            per_item.append({
                "id": item.get("id"), "scorable": False,
                "reason": "no relevant_chunk_keys",
            })
            continue

        chunks = retrieve_fn(item["query"], top_k)
        retrieved = [_chunk_key_of(c) for c in chunks]
        per_query.append({"retrieved": retrieved, "relevant": relevant})

        entry = {
            "id": item.get("id"), "scorable": True,
            "num_relevant": len(relevant), "num_retrieved": len(retrieved),
            "hit": bool(set(retrieved) & set(relevant)),
        }

        if manifest:
            health = _label_health(manifest, item)
            for k in ("total", "resolved", "moved", "missing", "unverifiable"):
                label_totals[k] += health[k]
            entry["label_health"] = health
        per_item.append(entry)

    result = {
        "n_scored": len(per_query),
        "n_unlabeled": unlabeled,
        "top_k": top_k,
        "metrics": scoring.retrieval_metrics(per_query),
        "per_item": per_item,
    }
    if manifest:
        result["label_health"] = dict(
            label_totals,
            corpus_hash=manifest.get("corpus_hash"),
            resolved_fraction=(
                round(label_totals["resolved"] / label_totals["total"], 6)
                if label_totals["total"] else None
            ),
        )
    return result


def _label_health(manifest: dict, item: dict) -> dict:
    """Per-item label resolution counts against the frozen corpus."""
    from shared.corpus_manifest import resolve_labels

    r = resolve_labels(manifest, item.get("relevant_chunk_keys", []))
    return {
        "total": r["total"],
        "resolved": len(r["resolved"]),
        "moved": len(r["moved"]),
        "missing": len(r["missing"]),
        "unverifiable": len(r["unverifiable"]),
    }


def run_retrieval_eval_from_file(
    path: str, top_k: int | None = None, corpus_id: str | None = None
) -> dict:
    items, errors_by_id = load_retrieval_dataset(path)
    manifest = None
    if corpus_id:
        from shared.corpus_manifest import load_manifest

        manifest = load_manifest(corpus_id)
        if manifest is None:
            # Silently scoring without the manifest would drop exactly the
            # label-durability guarantee the caller asked for.
            raise FileNotFoundError(
                f"No frozen corpus manifest '{corpus_id}' — freeze one first, "
                f"or omit corpus_id to score without label-durability checking."
            )
    result = run_retrieval_eval(items, top_k=top_k, manifest=manifest)
    result["dataset_path"] = path
    result["corpus_id"] = corpus_id
    result["skipped_items"] = errors_by_id
    return result
