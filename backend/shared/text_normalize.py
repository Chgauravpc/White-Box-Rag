"""
text_normalize.py — profile-based premise normalization for NLI verification.

`strip_table_noise` used to be one fixed set of regexes (tuned to RBI-style
financial-report PDFs) applied to every NLI premise regardless of corpus
domain — silently deleting short all-caps lines (e.g. named entities like
"NASA") and numeric lines (exactly the evidence a numeric-fact claim needs)
on any other kind of document. Normalization is now an explicit, swappable
profile so the same claim + evidence produces the same verdict regardless of
domain, and every deletion is reported rather than invisible.
"""

import re

_RE_NUMERIC_ROW    = re.compile(r"^[\d\s\(\),\.\-]+$")
_RE_PAGE_MARKER    = re.compile(r"^\d+\s*\|\s*P\s*a\s*g\s*e$", re.IGNORECASE)
_RE_TABLE_HEADER   = re.compile(r"^(Month\s*End|FCA|Gold|SDR|RTP|Forex\s*Reserves|USD\s*Million|Rs\.?\s*Crore|Table\s*\d+|Chart\s*\d+).*$", re.IGNORECASE)
_RE_ALL_CAPS_SHORT = re.compile(r"^[A-Z\s\-\.]{1,30}$")
_RE_DATE_NUM_ROW   = re.compile(r"^[A-Za-z]+-\d{2,4}\s+[\d\s\(\),\.\-]+$")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n")]


def _profile_none(text: str) -> tuple[str, list[str]]:
    """No mutation at all. External benchmark adapters pin this so nothing
    touches the benchmark's own evidence text."""
    return text, []


def _profile_generic(text: str) -> tuple[str, list[str]]:
    """Domain-neutral cleanup: blank lines and PDF page-number artifacts only.
    Safe for any corpus — a page marker is an extraction artifact, not content."""
    deleted, kept = [], []
    for line in _lines(text):
        if not line:
            continue
        if _RE_PAGE_MARKER.match(line):
            deleted.append(line)
            continue
        kept.append(line)
    return " ".join(kept), deleted


def _profile_financial_reports(text: str) -> tuple[str, list[str]]:
    """Legacy behaviour: strips numeric table rows, table headers, and short
    all-caps lines tuned to RBI-style financial-report PDFs. Opt-in only —
    on any other kind of document this deletes exactly the evidence a
    numeric-fact or named-entity claim needs."""
    deleted, kept = [], []
    for line in _lines(text):
        if not line:
            continue
        if _RE_NUMERIC_ROW.match(line):
            deleted.append(line); continue
        if _RE_PAGE_MARKER.match(line):
            deleted.append(line); continue
        if _RE_TABLE_HEADER.match(line):
            deleted.append(line); continue
        if _RE_ALL_CAPS_SHORT.match(line) and len(line) < 25:
            deleted.append(line); continue
        if _RE_DATE_NUM_ROW.match(line):
            deleted.append(line); continue
        kept.append(line)
    return " ".join(kept), deleted


_PROFILES = {
    "none": _profile_none,
    "generic": _profile_generic,
    "financial_reports": _profile_financial_reports,
}


def normalize_premise(text: str, profile: str = "generic") -> tuple[str, list[str]]:
    """Clean an NLI premise per the named profile.

    Returns (clean_text, deleted_lines) — deletions are always reported (even
    an empty list under "none") so premise mutation is auditable rather than
    invisible. Unknown profile names fall back to "generic".
    """
    fn = _PROFILES.get(profile, _profile_generic)
    return fn(text)
