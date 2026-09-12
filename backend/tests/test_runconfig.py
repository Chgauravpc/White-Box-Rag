"""
Tests for shared/runconfig.py — the frozen run-configuration snapshot used
for provenance (which config produced a given run's numbers).
"""

from shared.runconfig import RunConfig, capture_run_config


def _sample_config(**overrides) -> RunConfig:
    base = dict(
        llm_provider="groq", llm_model="llama-3.3-70b-versatile",
        generation_temperature=0.2, compliance_temperature=0.1, stability_temperature=0.7,
        global_seed=None, premise_normalizer="generic", domain_profile="generic",
        dense_top_k=20, sparse_top_k=20, final_top_k=10, rrf_k=60,
        min_attribution_score=0.45, ambiguity_gap_threshold=0.02, weak_attribution_score=0.65,
        strip_entailment_floor=0.5, abstention_mean_penalty_ceil=0.25,
        conflict_prefilter_cosine=0.90, conflict_contradiction_threshold=0.70,
        xai_embedding_model="BAAI/bge-large-en-v1.5", xai_embedding_model_revision=None,
        nli_model="cross-encoder/nli-deberta-v3-base", nli_model_revision=None,
        chroma_embedding_model="all-MiniLM-L6-v2", spacy_model="en_core_web_sm",
        chunk_max_tokens=512, chunk_overlap_tokens=50, parser_version="1.0",
        git_commit="abc123",
    )
    base.update(overrides)
    return RunConfig(**base)


class TestRunConfig:
    def test_content_hash_is_deterministic(self):
        a = _sample_config()
        b = _sample_config()
        assert a.content_hash() == b.content_hash()

    def test_content_hash_ignores_git_commit(self):
        """git_commit describes WHERE a run happened, not what config produced
        it — two runs on different commits with identical config must hash
        the same, or every run would look unique even with no real change."""
        a = _sample_config(git_commit="commit-one")
        b = _sample_config(git_commit="commit-two")
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_a_real_config_difference(self):
        a = _sample_config()
        b = _sample_config(abstention_mean_penalty_ceil=0.5)
        assert a.content_hash() != b.content_hash()

    def test_to_dict_includes_content_hash(self):
        d = _sample_config().to_dict()
        assert d["content_hash"] == _sample_config().content_hash()
        assert d["llm_provider"] == "groq"

    def test_frozen_is_immutable(self):
        cfg = _sample_config()
        try:
            cfg.llm_provider = "openrouter"
            assert False, "RunConfig should be frozen"
        except Exception:
            pass


class TestCaptureRunConfig:
    def test_captures_the_active_config_module_values(self):
        cfg = capture_run_config()
        assert cfg.llm_provider in ("groq", "openrouter")
        assert isinstance(cfg.dense_top_k, int)
        assert isinstance(cfg.content_hash(), str) and len(cfg.content_hash()) == 64

    def test_git_commit_is_a_string_even_if_git_is_unavailable(self):
        cfg = capture_run_config()
        assert isinstance(cfg.git_commit, str)
