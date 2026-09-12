"""
Tests for eval/schema.py — dataset v2 shape, validation, and the v1
compatibility upgrade.
"""

from eval.schema import (
    upgrade_v1_item,
    validate_item,
    load_and_validate,
    relevant_chunk_keys_as_dict,
    SCHEMA_VERSION,
)


V1_ITEM = {
    "id": "q001",
    "query": "What is discussed in the ingested documents?",
    "expected_section_ids": ["1.1", "2.3"],
    "expected_abstain": False,
    "notes": "a v1-style item",
}


class TestUpgradeV1Item:
    def test_stamps_schema_version(self):
        upgraded = upgrade_v1_item(V1_ITEM)
        assert upgraded["schema_version"] == SCHEMA_VERSION

    def test_expected_section_ids_become_ungraded_relevant_chunk_keys(self):
        upgraded = upgrade_v1_item(V1_ITEM)
        assert upgraded["relevant_chunk_keys"] == [
            {"key": "1.1", "grade": 1},
            {"key": "2.3", "grade": 1},
        ]

    def test_answerability_derived_from_expected_abstain(self):
        answerable = upgrade_v1_item({**V1_ITEM, "expected_abstain": False})
        unanswerable = upgrade_v1_item({**V1_ITEM, "expected_abstain": True})
        assert answerable["answerability"] == "ANSWERABLE"
        assert unanswerable["answerability"] == "UNANSWERABLE"

    def test_already_v2_item_passes_through_unchanged(self):
        v2_item = {"id": "q1", "query": "x", "expected_abstain": False, "schema_version": SCHEMA_VERSION}
        assert upgrade_v1_item(v2_item) == v2_item


class TestValidateItem:
    def test_valid_upgraded_item_has_no_errors(self):
        upgraded = upgrade_v1_item(V1_ITEM)
        assert validate_item(upgraded) == []

    def test_missing_required_field_is_reported(self):
        errors = validate_item({"query": "x"})
        assert any("id" in e for e in errors)
        assert any("expected_abstain" in e for e in errors)

    def test_invalid_query_type_is_reported(self):
        errors = validate_item({"id": "1", "query": "x", "expected_abstain": False, "query_type": "not-a-type"})
        assert any("query_type" in e for e in errors)

    def test_invalid_claim_label_is_reported(self):
        item = {
            "id": "1", "query": "x", "expected_abstain": False,
            "reference_claims": [{"text": "a claim", "label": "MAYBE"}],
        }
        errors = validate_item(item)
        assert any("label" in e for e in errors)

    def test_invalid_relevance_grade_is_reported(self):
        item = {
            "id": "1", "query": "x", "expected_abstain": False,
            "relevant_chunk_keys": [{"key": "A|B|1.1|0", "grade": 7}],
        }
        errors = validate_item(item)
        assert any("grade" in e for e in errors)


class TestLoadAndValidate:
    def test_upgrades_and_reports_no_errors_for_v1_dataset(self):
        items, errors_by_id = load_and_validate([V1_ITEM])
        assert errors_by_id == {}
        assert items[0]["schema_version"] == SCHEMA_VERSION

    def test_never_raises_on_a_bad_item(self):
        items, errors_by_id = load_and_validate([{"query": "missing id and expected_abstain"}])
        assert len(errors_by_id) == 1

    def test_upgrade_false_skips_the_v1_shim(self):
        items, _ = load_and_validate([V1_ITEM], upgrade=False)
        assert "relevant_chunk_keys" not in items[0]


class TestRelevantChunkKeysAsDict:
    def test_converts_to_key_grade_mapping(self):
        item = {"relevant_chunk_keys": [{"key": "A|B|1.1|0", "grade": 3}, {"key": "A|B|2.1|1", "grade": 1}]}
        assert relevant_chunk_keys_as_dict(item) == {"A|B|1.1|0": 3, "A|B|2.1|1": 1}

    def test_missing_field_returns_empty_dict(self):
        assert relevant_chunk_keys_as_dict({}) == {}

    def test_default_grade_is_one(self):
        item = {"relevant_chunk_keys": [{"key": "A|B|1.1|0"}]}
        assert relevant_chunk_keys_as_dict(item) == {"A|B|1.1|0": 1}
