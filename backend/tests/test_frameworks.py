"""
Tests for the regulatory framework catalog (Feature 5).

Static data + coverage math — no I/O, no models, no Gemini.
"""

from compliance.frameworks import get_frameworks, _STATUS_WEIGHT

_REQUIRED_CONTROL_KEYS = {"control_id", "title", "requirement", "satisfied_by", "evidence", "status"}


def test_catalog_loads_and_is_nonempty():
    fws = get_frameworks()
    assert len(fws) >= 2
    names = {fw["framework"] for fw in fws}
    assert "EU AI Act" in names
    assert "NIST AI RMF 1.0" in names


def test_every_control_has_required_keys_and_valid_status():
    for fw in get_frameworks():
        assert fw["controls"], f"{fw['framework']} has no controls"
        for c in fw["controls"]:
            missing = _REQUIRED_CONTROL_KEYS - set(c)
            assert not missing, f"{fw['framework']}/{c.get('control_id')} missing {missing}"
            assert c["status"] in _STATUS_WEIGHT
            assert isinstance(c["satisfied_by"], list) and c["satisfied_by"]
            assert isinstance(c["evidence"], list) and c["evidence"]


def test_coverage_math_matches_status_weights():
    for fw in get_frameworks():
        controls = fw["controls"]
        expected = sum(_STATUS_WEIGHT[c["status"]] for c in controls) / len(controls)
        assert abs(fw["coverage"] - round(expected, 4)) < 1e-9
        assert 0.0 <= fw["coverage"] <= 1.0


def test_status_counts_sum_to_num_controls():
    for fw in get_frameworks():
        assert sum(fw["status_counts"].values()) == fw["num_controls"] == len(fw["controls"])
