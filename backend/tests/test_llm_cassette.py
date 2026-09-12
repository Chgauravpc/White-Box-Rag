"""
Tests for shared/llm_cassette.py (Phase 2, W2.5) — record/replay determinism.
"""

import shared.llm_cassette as llm_cassette
from shared.llm_cassette import Cassette, cassette_key, cassette_scope, active_cassette_name


class TestCassetteKey:
    def test_same_inputs_produce_same_key(self):
        k1 = cassette_key("groq", "m1", 0.2, 42, "sys", "prompt")
        k2 = cassette_key("groq", "m1", 0.2, 42, "sys", "prompt")
        assert k1 == k2

    def test_prompt_change_changes_key(self):
        k1 = cassette_key("groq", "m1", 0.2, 42, "sys", "prompt A")
        k2 = cassette_key("groq", "m1", 0.2, 42, "sys", "prompt B")
        assert k1 != k2

    def test_seed_none_differs_from_seed_zero(self):
        k1 = cassette_key("groq", "m1", 0.2, None, "sys", "prompt")
        k2 = cassette_key("groq", "m1", 0.2, 0, "sys", "prompt")
        assert k1 != k2

    def test_missing_system_instruction_normalizes_to_empty_string(self):
        k1 = cassette_key("groq", "m1", 0.2, None, None, "prompt")
        k2 = cassette_key("groq", "m1", 0.2, None, "", "prompt")
        assert k1 == k2


class TestCassetteRoundTrip:
    def test_append_then_get_returns_recorded_response(self, tmp_path):
        cassette = Cassette(str(tmp_path / "c.jsonl"))
        key = cassette_key("groq", "m1", 0.2, None, None, "hello")
        cassette.append(key, "groq", "m1", "hi there")
        assert cassette.get(key) == "hi there"

    def test_missing_key_returns_none(self, tmp_path):
        cassette = Cassette(str(tmp_path / "c.jsonl"))
        assert cassette.get("nonexistent") is None

    def test_reload_from_disk_sees_prior_appends(self, tmp_path):
        path = tmp_path / "c.jsonl"
        key = cassette_key("groq", "m1", 0.2, None, None, "hello")
        Cassette(str(path)).append(key, "groq", "m1", "hi there")
        reloaded = Cassette(str(path))
        assert reloaded.get(key) == "hi there"

    def test_re_recording_same_key_overwrites_on_next_load(self, tmp_path):
        path = tmp_path / "c.jsonl"
        key = cassette_key("groq", "m1", 0.2, None, None, "hello")
        Cassette(str(path)).append(key, "groq", "m1", "first response")
        Cassette(str(path)).append(key, "groq", "m1", "second response")
        reloaded = Cassette(str(path))
        assert reloaded.get(key) == "second response"

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(
            '{"key": "k1", "provider": "groq", "model": "m1", "response": "ok"}\n'
            'not valid json\n',
            encoding="utf-8",
        )
        cassette = Cassette(str(path))
        assert cassette.get("k1") == "ok"


class TestCassetteScope:
    def test_default_name_when_no_scope_active(self):
        assert active_cassette_name() == "default"

    def test_scope_sets_and_restores_name(self):
        assert active_cassette_name() == "default"
        with cassette_scope("my_dataset"):
            assert active_cassette_name() == "my_dataset"
        assert active_cassette_name() == "default"

    def test_get_cassette_uses_active_scope_name(self, tmp_path, monkeypatch):
        llm_cassette.reset_cassettes()
        key = cassette_key("groq", "m1", 0.2, None, None, "hi")
        with cassette_scope("scoped_name"):
            cassette = llm_cassette.get_cassette(str(tmp_path))
            cassette.append(key, "groq", "m1", "resp")
        assert (tmp_path / "scoped_name.jsonl").exists()
