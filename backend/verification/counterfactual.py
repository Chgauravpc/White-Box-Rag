"""
counterfactual.py — contrastive "what would change the verdict" explanations.

For a non-SAFE answer, show the human which claim, if removed, most improves the
trust outcome — e.g. "removing claim 3 raises the score 0.55 → 0.85 and flips
Non-Compliant → Needs Review." Pure math, no Gemini.

Two correctness points drive the design:

1. Trust-gate STATUS is decided by *which penalty types remain* (the boolean
   flags in trust_gate.compute_trust_gate), NOT by a threshold on overall_score.
   So a projected status can't be read off a projected scalar — we recompute the
   gate on the leave-one-out claim set to get the exact status and score.

2. compute_shapley_contributions returns arrays SORTED by descending phi, so its
   indices are NOT aligned with the verifications list. We therefore derive each
   claim's phi/reasons from a single-claim Shapley call (reusing the canonical
   penalty tables) rather than indexing the sorted arrays.
"""

from typing import List

from shared.models import VerificationResult, EditionConflict, TrustGate
from verification.trust_gate import compute_trust_gate
from shared.xai_matrices import compute_shapley_contributions


def compute_counterfactuals(
    verifications: List[VerificationResult],
    primary_attributions: list,
    conflicts: List[EditionConflict],
    trust_gate: TrustGate,
) -> list[dict]:
    """Per penalising claim: its contribution phi and the exact gate outcome if removed.

    `primary_attributions[i]` corresponds to `verifications[i]`. Conflicts are
    passed through unchanged to each recomputation — removing a claim never
    resolves an edition conflict, so the counterfactual honestly reflects that.
    Returns a list sorted by phi desc; the top entry is flagged primary_driver.
    """
    primary_attributions = primary_attributions or []
    current_status = trust_gate.status
    n = len(verifications)
    out: list[dict] = []

    for i in range(n):
        # Per-claim phi + reasons via the canonical Shapley logic (single-claim call).
        attr_i = [primary_attributions[i]] if i < len(primary_attributions) else []
        single = compute_shapley_contributions([verifications[i].model_dump()], attr_i)
        phi = single["shapley_values"][0] if single["shapley_values"] else 0.0
        if phi <= 0:
            continue  # this claim contributes no penalty — removing it changes nothing
        reasons = single["penalty_reasons"][0] if single["penalty_reasons"] else []

        # Exact projected outcome: recompute the gate with claim i removed.
        loo_verifs = [verifications[j] for j in range(n) if j != i]
        loo_attrs = [primary_attributions[j] for j in range(len(primary_attributions)) if j != i]
        projected = compute_trust_gate(loo_verifs, conflicts, loo_attrs)

        out.append({
            "claim_text": verifications[i].claim_text,
            "phi": round(phi, 4),
            "penalty_reasons": reasons,
            "score_if_removed": round(projected.overall_score, 4),
            "status_if_removed": projected.status.value,
            "flips_status": projected.status != current_status,
            "primary_driver": False,
        })

    out.sort(key=lambda r: r["phi"], reverse=True)
    if out:
        out[0]["primary_driver"] = True
    return out
