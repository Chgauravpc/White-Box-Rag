from typing import List, Tuple
import numpy as np

from shared.models import Claim, VerificationResult, NLIVerdict
from shared.xai_matrices import averify_claims_batch, abuild_entailment_matrix


async def verify_all_claims(
    claims: List[Claim],
) -> Tuple[List[VerificationResult], np.ndarray, List[str], List[List[str]]]:
    """Verifies claims via batched DeBERTa CrossEncoder (Matrix 2).

    Returns:
        (verifications, E_matrix, focused_passages, premise_deletions)
        focused_passages[i] is the sentence subset actually fed to NLI for claim i —
        stored in Claim.focused_passage for full audit traceability.
        premise_deletions[i] is the lines PREMISE_NORMALIZER removed from claim i's
        source passage before NLI scoring — stored in Claim.premise_deletions so
        premise mutation is auditable rather than invisible.
    """
    if not claims:
        return [], np.array([]), [], []

    pairs = [(claim.text, claim.source_passage) for claim in claims]

    # Batched NLI, off the event loop (shared/xai_matrices.py::averify_claims_batch)
    # — the synchronous CrossEncoder pass used to block every concurrently
    # running query's LLM calls. Premise preparation happens inside.
    raw_results = await averify_claims_batch(pairs)

    verifications     = []
    focused_passages  = []
    premise_deletions = []
    for claim, result in zip(claims, raw_results):
        verifications.append(
            VerificationResult(
                claim_text=claim.text,
                verdict=result["verdict"],
                entailment_score=result["entailment_score"],
                explanation=(
                    f"CrossEncoder Probabilities -> "
                    f"Entail: {result['entailment_score']:.2f}, "
                    f"Contradict: {result['contradiction_score']:.2f}, "
                    f"Neutral: {result['neutral_score']:.2f}"
                    if result.get("evidence_status", "ok") == "ok"
                    else f"No NLI premise ({result.get('evidence_status')}) — claim not verifiable against any source."
                ),
                evidence_status=result.get("evidence_status", "ok"),
            )
        )
        focused_passages.append(result.get("focused_passage", claim.source_passage))
        premise_deletions.append(result.get("premise_deletions", []))

    # Build full E matrix for XAI visualization
    claim_texts = [c.text for c in claims]
    passages    = list(dict.fromkeys([c.source_passage for c in claims if c.source_passage]))
    if passages:
        E_matrix, _ = await abuild_entailment_matrix(claim_texts, passages)
    else:
        E_matrix = np.array([])

    return verifications, E_matrix, focused_passages, premise_deletions
