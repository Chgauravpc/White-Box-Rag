"""
Tests for shared/text_normalize.py — profile-based NLI premise normalization.

Pure functions over strings; no ML models needed.
"""

from shared.text_normalize import normalize_premise


NUMERIC_FACT_PASSAGE = "Reserves increased from USD 668B to USD 700B.\n668,000 700,000\nNASA\nThe agency reported the figures."


class TestNoneProfile:
    def test_leaves_text_completely_untouched(self):
        clean, deleted = normalize_premise(NUMERIC_FACT_PASSAGE, "none")
        assert clean == NUMERIC_FACT_PASSAGE
        assert deleted == []


class TestGenericProfile:
    def test_strips_only_page_markers(self):
        text = "12 | P a g e\nThe policy applies to all users.\nNASA\n668,000 700,000"
        clean, deleted = normalize_premise(text, "generic")
        assert "12 | P a g e" not in clean
        assert "12 | P a g e" in deleted
        # A numeric-fact line and a short all-caps entity line must survive —
        # this is the exact evidence financial_reports would have deleted.
        assert "NASA" in clean
        assert "668,000 700,000" in clean

    def test_numeric_fact_claim_verdict_is_not_starved_of_evidence(self):
        clean, _ = normalize_premise(NUMERIC_FACT_PASSAGE, "generic")
        assert "668B" in clean and "700B" in clean

    def test_unknown_profile_falls_back_to_generic(self):
        clean, _ = normalize_premise("12 | P a g e\nHello world.", "not-a-real-profile")
        assert "12 | P a g e" not in clean
        assert "Hello world." in clean


class TestFinancialReportsProfile:
    def test_strips_numeric_rows_and_short_all_caps_lines(self):
        clean, deleted = normalize_premise(NUMERIC_FACT_PASSAGE, "financial_reports")
        # Legacy behaviour: numeric rows and short all-caps lines are removed —
        # this is why the profile is opt-in, not the default, on other domains.
        assert "668,000 700,000" not in clean
        assert "NASA" not in clean
        assert "668,000 700,000" in deleted
        assert "NASA" in deleted

    def test_prose_sentence_survives(self):
        clean, _ = normalize_premise(NUMERIC_FACT_PASSAGE, "financial_reports")
        assert "Reserves increased from USD 668B to USD 700B." in clean


class TestNormalizationInvariance:
    """The same evidence should not be silently mutilated under the default
    profile just because the corpus happens to resemble a financial report."""

    def test_generic_default_preserves_evidence_financial_reports_would_delete(self):
        _, deleted_generic = normalize_premise(NUMERIC_FACT_PASSAGE, "generic")
        _, deleted_financial = normalize_premise(NUMERIC_FACT_PASSAGE, "financial_reports")
        assert deleted_generic == []
        assert len(deleted_financial) > 0
