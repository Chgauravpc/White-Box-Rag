"""
stability.py — Paraphrase / self-consistency stability check.

Only triggers a second LLM generation when the primary answer's trust
status isn't already SAFE (bounds cost/latency — most queries never pay for
this). The *comparison* between the two samples is pure NLI claim-agreement
via the already-loaded CrossEncoder — the LLM's role is strictly "generate a
second sample," never "judge whether they agree." Consistent with the
project's "LLM extracts, Math judges" rule: no LLM-as-judge.

Design note: an earlier version compared whole-answer embedding cosine
similarity, but two answers drawn from the same sources tend to share heavy
domain vocabulary regardless of factual agreement, so that signal saturates
near 1.0 and barely discriminates. Claim-level NLI entailment is sensitive to
actual factual drift between the two samples instead.
"""

import logging
from typing import List

from shared import config
from shared.models import Claim, TrustStatus
from shared.xai_matrices import anli_predict, entailment_score_of, prepare_premise

logger = logging.getLogger(__name__)

STABILITY_TEMPERATURE = config.STABILITY_TEMPERATURE


async def compute_paraphrase_stability(
    query: str,
    chunks: list[dict],
    primary_claims: List[Claim],
    trust_status: TrustStatus,
) -> tuple[float, str]:
    """Returns (stability_score, explanation)."""
    if trust_status == TrustStatus.SAFE:
        return 1.0, "Skipped — primary answer already Safe"

    retained_claims = [c for c in primary_claims if c.retained]
    if not retained_claims:
        return 0.0, "No retained claims to check stability against"

    # Local import: avoids a circular import (rag.py doesn't import this module).
    from ingestion.rag import rag_query

    try:
        second_response, _ = await rag_query(query, chunks, temperature=STABILITY_TEMPERATURE)
    except Exception as e:
        logger.warning(f"Paraphrase stability generation failed: {e}")
        return 1.0, "Skipped — second sample generation failed"

    second_answer = second_response.answer
    if not second_answer:
        return 1.0, "Skipped — second sample was empty"

    # Premise preparation goes through the same prepare_premise() the verdict
    # and the XAI matrix use, rather than a third independent copy of the
    # normalize/focus logic — this used to skip normalization entirely and
    # apply the length threshold to the raw text.
    pairs = []
    for c in retained_claims:
        focused, _deleted, status = prepare_premise(c.text, second_answer)
        if status == "ok":
            pairs.append((focused, c.text))

    if not pairs:
        return 1.0, "Skipped — second sample had no usable premise"

    # Off the event loop, and index order comes from the active checkpoint
    # rather than a hardcoded assumption (shared/xai_matrices.py::NLI_LABELS):
    # a different NLI_MODEL can emit a different order, which would silently
    # read "neutral" as "entailment".
    scores = await anli_predict(pairs, apply_softmax=True)
    entailment_scores = [entailment_score_of(s) for s in scores]
    stability = round(sum(entailment_scores) / len(entailment_scores), 6)

    return stability, f"Compared {len(retained_claims)} retained claim(s) against a resampled answer via NLI entailment"
