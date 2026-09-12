"""
conftest.py -- Root pytest configuration.
Mocks ML models BEFORE any backend module imports them at module level.
"""
import hashlib
import os
import sys
import tempfile
from unittest.mock import MagicMock
import numpy as np
import pytest

# ── Path & env setup ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

_tmp = tempfile.mkdtemp()
os.environ["CHROMA_PATH"]    = os.path.join(_tmp, "chromadb")
os.environ["SQLITE_PATH"]    = os.path.join(_tmp, "metadata.db")
os.environ["GROQ_API_KEY"] = "test-key-not-used"


# ── Fake ML models ────────────────────────────────────────────────────────────

# These fakes are DETERMINISTIC: the same input always produces the same
# output. They used to return np.random.rand(...), which made every assertion
# about a verdict, a similarity ranking or a score distribution depend on the
# global RNG — a test could pass or fail run to run, and a genuine regression
# was indistinguishable from a bad draw. Anything asserting on model output
# needs a stable function, not merely a plausible one.
#
# The seed is a SHA-256 of the input text, not Python's hash(): str hashing is
# salted per process, so hash()-seeded fakes would be stable within one run and
# different across runs — the worst of both worlds.

def _seed_from(*parts) -> int:
    digest = hashlib.sha256("|~|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _make_encoder(dim=1024):
    enc = MagicMock()
    def _encode(texts, **kw):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        # Per-text seeding, so identical strings embed identically and
        # cosine similarity between two texts is a stable, reproducible number.
        out = np.stack([
            np.random.RandomState(_seed_from("encode", t)).rand(dim).astype("float32")
            for t in texts
        ]) if texts else np.zeros((0, dim), dtype="float32")
        return out
    enc.encode.side_effect = _encode
    return enc


def _make_nli():
    nli = MagicMock()
    def _predict(pairs, apply_softmax=True, **kw):
        pairs = list(pairs)
        if not pairs:
            return np.zeros((0, 3), dtype="float32")
        rows = []
        for pair in pairs:
            premise, hypothesis = (pair if isinstance(pair, (tuple, list)) and len(pair) == 2
                                   else (pair, ""))
            raw = np.abs(np.random.RandomState(
                _seed_from("nli", premise, hypothesis)).rand(3)).astype("float32")
            rows.append(raw / raw.sum())
        return np.stack(rows)
    nli.predict.side_effect = _predict
    return nli


def _make_spacy_nlp():
    class _Sent:
        def __init__(self, t): self.text = t
    class _Doc:
        def __init__(self, text):
            parts = [p.strip() for p in text.replace("\n", " ").split(". ") if p.strip()]
            self.sents = [_Sent(p) for p in parts] if parts else [_Sent(text)]
    nlp = MagicMock()
    nlp.side_effect = lambda text: _Doc(text)
    return nlp


# Patch sentence_transformers BEFORE any backend module is imported
_st_mod = MagicMock()
_st_mod.SentenceTransformer.side_effect = lambda *a, **kw: _make_encoder()
_st_mod.CrossEncoder.side_effect        = lambda *a, **kw: _make_nli()

sys.modules.setdefault("sentence_transformers", _st_mod)
sys.modules.setdefault("spacy", MagicMock(load=MagicMock(return_value=_make_spacy_nlp())))


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_chunks():
    return [
        {"chunk_text": "Reserves increased from USD 668B to USD 700B.", "section_id": "I.2.1", "publication_name": "FER", "edition_date": "Oct 2025"},
        {"chunk_text": "The current account deficit narrowed in Q2.",   "section_id": "I.2.2", "publication_name": "FER", "edition_date": "Oct 2025"},
        {"chunk_text": "Gold reserves rose by USD 12 billion.",          "section_id": "I.5",   "publication_name": "FER", "edition_date": "Oct 2025"},
    ]


@pytest.fixture()
def sample_verifications():
    from shared.models import VerificationResult
    return [
        VerificationResult(claim_text="Reserves rose.",  verdict="ENTAILMENT", entailment_score=0.97, explanation=""),
        VerificationResult(claim_text="Deficit narrowed.", verdict="ENTAILMENT", entailment_score=0.88, explanation=""),
    ]


@pytest.fixture()
def sample_claims():
    from shared.models import Claim
    return [
        Claim(text="Reserves rose.", source_section_id="I.2.1", source_passage="Reserves increased from USD 668B to USD 700B.", source_publication="FER", source_edition="Oct 2025"),
        Claim(text="Deficit narrowed.", source_section_id="I.2.2", source_passage="The current account deficit narrowed in Q2.", source_publication="FER", source_edition="Oct 2025"),
    ]
