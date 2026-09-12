"""
Tests for shared/config.py — typed env readers, and that the centralized
penalty/threshold constants actually reach the modules that used to
independently redeclare them (a real drift risk before this refactor: e.g.
WEAK_ATTRIBUTION_SCORE existed as three separate literals).
"""

import importlib
import os

import pytest


@pytest.fixture
def reload_config():
    """Reload shared.config after mutating os.environ, then restore it."""
    import shared.config as config_module
    saved_env = dict(os.environ)

    def _reload():
        importlib.reload(config_module)
        return config_module

    yield _reload
    os.environ.clear()
    os.environ.update(saved_env)
    importlib.reload(config_module)


class TestTypedEnvReaders:
    def test_env_float_overrides_default(self, reload_config):
        os.environ["ABSTENTION_MEAN_PENALTY_CEIL"] = "0.42"
        config = reload_config()
        assert config.ABSTENTION_MEAN_PENALTY_CEIL == 0.42

    def test_env_int_overrides_default(self, reload_config):
        os.environ["FINAL_TOP_K"] = "15"
        config = reload_config()
        assert config.FINAL_TOP_K == 15

    def test_missing_env_var_uses_default(self, reload_config):
        os.environ.pop("RRF_K", None)
        config = reload_config()
        assert config.RRF_K == 60

    def test_global_seed_defaults_to_none_when_unset(self, reload_config):
        os.environ.pop("GLOBAL_SEED", None)
        config = reload_config()
        assert config.GLOBAL_SEED is None

    def test_global_seed_zero_is_distinct_from_unset(self, reload_config):
        """0 is a valid seed — must not be conflated with 'not set'."""
        os.environ["GLOBAL_SEED"] = "0"
        config = reload_config()
        assert config.GLOBAL_SEED == 0


class TestPenaltyConstantsAreCentralized:
    """Regression test for the drift bug: WEAK_ATTRIBUTION_SCORE (and others)
    used to be redeclared as independent literals in three modules."""

    def test_weak_attribution_score_is_identical_across_modules(self):
        from shared import config
        from shared import xai_matrices
        from verification import trust_gate
        from verification import mitigation

        assert xai_matrices.WEAK_ATTRIBUTION_SCORE == config.WEAK_ATTRIBUTION_SCORE
        assert trust_gate.WEAK_ATTRIBUTION_SCORE == config.WEAK_ATTRIBUTION_SCORE
        assert mitigation.WEAK_ATTRIBUTION_SCORE == config.WEAK_ATTRIBUTION_SCORE

    def test_trust_gate_and_shapley_use_the_same_penalty_values(self):
        """trust_gate.compute_trust_gate and xai_matrices.compute_shapley_contributions
        are documented to apply identical penalties so their scores stay
        consistent — verify they actually read the same config constants."""
        from verification.trust_gate import compute_trust_gate
        from shared.xai_matrices import compute_shapley_contributions
        from shared.models import VerificationResult, NLIVerdict

        verifications = [
            VerificationResult(claim_text="x", verdict=NLIVerdict.CONTRADICTION, entailment_score=0.1, explanation=""),
        ]
        gate = compute_trust_gate(verifications, [], [])
        shapley = compute_shapley_contributions([v.model_dump() for v in verifications], [])
        assert gate.overall_score == pytest.approx(shapley["overall_score"])
