from typing import List, Tuple
import numpy as np

from shared.models import Claim, VerificationResult, NLIVerdict
from shared.xai_matrices import verify_claims_batch, build_entailment_matrix


async def verify_all_claims(
    claims: List[Claim],
) -> Tuple[List[VerificationResult], np.ndarray, List[str]]:
    """Verifies claims via batched DeBERTa CrossEncoder (Matrix 2).

    Returns:
        (verifications, E_matrix, focused_passages)
        focused_passages[i] is the sentence subset actually fed to NLI for claim i —
        stored in Claim.focused_passage for full audit traceability.
    """
    if not claims:
        return [], np.array([]), []

    pairs = [(claim.text, claim.source_passage) for claim in claims]

    # Batched NLI — also applies extract_relevant_sentences() internally
    raw_results = verify_claims_batch(pairs)

    verifications    = []
    focused_passages = []
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
                )
            )
        )
        focused_passages.append(result.get("focused_passage", claim.source_passage))

    # Build full E matrix for XAI visualization
    claim_texts = [c.text for c in claims]
    passages    = list(dict.fromkeys([c.source_passage for c in claims if c.source_passage]))
    if passages:
        E_matrix, _ = build_entailment_matrix(claim_texts, passages)
    else:
        E_matrix = np.array([])

    return verifications, E_matrix, focused_passages
