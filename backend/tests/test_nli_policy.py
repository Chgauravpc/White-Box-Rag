"""
Tests for shared/nli_policy.py and the empty-premise guard (Phase 4, finding #10).

Two things are being pinned here:

1. `trust_gate` and `compute_shapley_contributions` must charge the same
   penalties for the same claim. Their docstrings have promised this for a
   while and they have drifted three times; these tests compare the two
   computations directly rather than checking each against a hand-computed
   constant, so a future divergence fails loudly regardless of what the
   penalty values happen to be.

2. An empty NLI premise must never reach the model. A CrossEncoder handed
   ("", claim) returns a perfectly well-formed softmax, which is the most
   dangerous possible output for a hallucination detector: a fabricated
   confidence that is indistinguishable downstream from a measured one.
"""

import numpy as np

from shared import config
from shared.nli_policy import (
    is_contradiction, is_entailment, is_neutral, nli_penalty_flags, normalize_verdict,
)
from shared.models import NLIVerdict, VerificationResult
from shared.xai_matrices import compute_shapley_contributions, verify_claims_batch
from verification.trust_gate import compute_trust_gate


class TestVerdictPredicates:
    def test_both_spellings_of_contradiction(self):
        assert is_contradiction("CONTRADICTION") and is_contradiction("CONTRADICTED")
        assert is_contradiction(NLIVerdict.CONTRADICTION) and is_contradiction(NLIVerdict.CONTRADICTED)

    def test_both_spellings_of_neutral(self):
        assert is_neutral("NEUTRAL") and is_neutral("NOT_ENOUGH_INFO")
        assert is_neutral(NLIVerdict.NEUTRAL) and is_neutral(NLIVerdict.NOT_ENOUGH_INFO)

    def test_both_spellings_of_entailment(self):
        assert is_entailment("ENTAILMENT") and is_entailment("SUPPORTED")

    def test_enum_normalizes_to_bare_value_not_repr(self):
        """NLIVerdict is a str Enum; str() on a member can yield
        'NLIVerdict.NEUTRAL' on some versions, which would match nothing."""
        assert normalize_verdict(NLIVerdict.NEUTRAL) == "NEUTRAL"

    def test_none_and_unknown_are_inert(self):
        assert normalize_verdict(None) == ""
        flags = nli_penalty_flags("SOMETHING_ELSE", 0.95)
        assert not flags["contradiction"] and not flags["neutral"]


class TestPenaltyFlags:
    def test_contradiction_skips_confidence_band(self):
        """A contradiction always carries a low entailment score; charging the
        confidence penalty too would penalize one fact twice."""
        flags = nli_penalty_flags("CONTRADICTION", 0.01)
        assert flags["contradiction"] and not flags["low_confidence"]
        assert flags["penalty"] == config.PENALTY_CONTRADICTION

    def test_neutral_does_stack_with_low_confidence(self):
        flags = nli_penalty_flags("NEUTRAL", 0.1)
        assert flags["neutral"] and flags["low_confidence"]
        assert flags["penalty"] == config.PENALTY_NEUTRAL + config.PENALTY_LOW_CONFIDENCE

    def test_high_confidence_entailment_is_free(self):
        assert nli_penalty_flags("ENTAILMENT", 0.99)["penalty"] == 0.0


def _shapley_score(verdict, score):
    out = compute_shapley_contributions(
        [{"verdict": verdict, "entailment_score": score, "claim_text": "c"}],
        primary_attributions=[{}],
    )
    return out["overall_score"]


def _gate_score(verdict, score):
    return compute_trust_gate(
        [VerificationResult(claim_text="c", verdict=verdict, entailment_score=score)],
        conflicts=[],
        primary_attributions=[{}],
    ).overall_score


class TestGateAndShapleyAgree:
    """The two computations must never disagree, for ANY verdict spelling."""

    def test_alias_and_canonical_spellings_agree(self):
        for verdict in ["CONTRADICTION", "CONTRADICTED", "NEUTRAL", "NOT_ENOUGH_INFO",
                        "ENTAILMENT", "SUPPORTED"]:
            for score in [0.0, 0.3, 0.6, 0.95]:
                assert _gate_score(verdict, score) == _shapley_score(verdict, score), (
                    f"trust_gate and Shapley disagree on {verdict} @ {score}"
                )

    def test_canonical_contradicted_is_not_treated_as_benign(self):
        """Regression: Shapley used to match only the literal 'CONTRADICTION',
        so a CONTRADICTED verdict skipped the contradiction penalty there while
        the trust gate applied it."""
        assert _shapley_score("CONTRADICTED", 0.1) == _shapley_score("CONTRADICTION", 0.1)

    def test_canonical_not_enough_info_is_not_treated_as_benign(self):
        assert _shapley_score("NOT_ENOUGH_INFO", 0.1) == _shapley_score("NEUTRAL", 0.1)


class TestEmptyPremiseGuard:
    def test_empty_premise_never_reaches_the_model(self, monkeypatch):
        """The core fix: no premise means no prediction, not a fabricated one."""
        calls = []
        import shared.xai_matrices as xm

        def _spy(pairs, **kw):
            calls.append(list(pairs))
            raise AssertionError("NLI model must not be called for an empty premise")

        monkeypatch.setattr(xm._nli, "predict", _spy)
        results = verify_claims_batch([("some claim", "")])
        assert calls == []
        assert results[0]["verdict"] == "NOT_ENOUGH_INFO"
        assert results[0]["evidence_status"] == "no_premise"
        assert results[0]["entailment_score"] == 0.0

    def test_scores_are_structural_zero_not_a_distribution(self):
        r = verify_claims_batch([("claim", "")])[0]
        assert (r["entailment_score"], r["contradiction_score"], r["neutral_score"]) == (0.0, 0.0, 0.0)

    def test_normalizer_deleting_everything_is_distinguished(self, monkeypatch):
        """A passage that existed but was entirely removed by the normalizer is
        a normalizer bug, not a property of the claim — it must not be
        reported as 'no_premise'."""
        import shared.xai_matrices as xm
        monkeypatch.setattr(xm, "normalize_premise", lambda text, profile: ("", ["everything"]))
        r = verify_claims_batch([("claim", "a real passage was here")])[0]
        assert r["evidence_status"] == "normalizer_deleted_all"
        assert r["verdict"] == "NOT_ENOUGH_INFO"

    def test_real_premise_still_scored_normally(self):
        r = verify_claims_batch([("claim", "A real supporting passage about reserves.")])[0]
        assert r["evidence_status"] == "ok"
        assert r["verdict"] in {"CONTRADICTION", "ENTAILMENT", "NEUTRAL"}

    def test_mixed_batch_keeps_results_aligned(self, monkeypatch):
        """Filtering empty premises out of the model batch must not shift the
        results of the claims that DID get scored."""
        import shared.xai_matrices as xm
        monkeypatch.setattr(
            xm._nli, "predict",
            lambda pairs, **kw: np.array([[0.0, 1.0, 0.0] for _ in pairs], dtype="float32"),
        )
        results = verify_claims_batch([
            ("claim A", ""),
            ("claim B", "A real passage."),
            ("claim C", ""),
        ])
        assert [r["claim_text"] for r in results] == ["claim A", "claim B", "claim C"]
        assert [r["evidence_status"] for r in results] == ["no_premise", "ok", "no_premise"]
        assert results[1]["entailment_score"] == 1.0

    def test_all_empty_batch_does_not_call_predict_with_empty_list(self, monkeypatch):
        import shared.xai_matrices as xm

        def _spy(pairs, **kw):
            raise AssertionError("predict() called with an empty batch")

        monkeypatch.setattr(xm._nli, "predict", _spy)
        results = verify_claims_batch([("a", ""), ("b", "")])
        assert len(results) == 2


class TestStripReasonHonesty:
    def test_no_evidence_claim_is_not_labelled_low_confidence(self):
        """It was reported as 'low_confidence' — attributing a judgement to a
        model that never ran on the claim."""
        from shared.models import Claim
        from verification.mitigation import filter_claims

        claims = [Claim(text="Unattributable sentence.")]
        verifications = [VerificationResult(
            claim_text="Unattributable sentence.", verdict=NLIVerdict.NOT_ENOUGH_INFO,
            entailment_score=0.0, evidence_status="no_premise",
        )]
        _, retained, reasons = filter_claims(claims, verifications, [{}])
        assert retained == [False]
        assert reasons == ["no_evidence:no_premise"]

    def test_genuinely_low_scored_claim_still_reads_low_confidence(self):
        from shared.models import Claim
        from verification.mitigation import filter_claims

        claims = [Claim(text="Weakly supported.")]
        verifications = [VerificationResult(
            claim_text="Weakly supported.", verdict=NLIVerdict.NEUTRAL,
            entailment_score=0.2, evidence_status="ok",
        )]
        _, retained, reasons = filter_claims(claims, verifications, [{}])
        assert retained == [False] and reasons == ["low_confidence"]
