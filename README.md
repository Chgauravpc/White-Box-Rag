# White-Box RAG — Explainable AI Governance & Trust Framework

A **domain-agnostic, compliance-grade governance layer** that sits on top of any RAG pipeline and makes it *auditable*. Instead of trusting an LLM's answer blindly, White-Box RAG decomposes every response into individual claims, verifies each one against its source, computes a mathematically-derived trust score, and emits a deterministic **SAFE / NEEDS-HUMAN-REVIEW / NON-COMPLIANT** gate — with a full audit trail persisted to SQLite.

The core idea: **the retrieval, ranking, conflict-detection, and trust-scoring layers use pure linear algebra and deterministic rules — no LLM in the verification loop.** That makes every governance decision reconstructible, explainable, and repeatable.

> Built as a hackathon project; the reference domain is RBI banking-compliance (mapping a Business Requirements Document against regulatory circulars), but the framework works on any document corpus.

---

## Why "White-Box"?

Most RAG systems are black boxes: a query goes in, an answer comes out, and you have no principled way to know *which source drove which claim* or *how much to trust it*. White-Box RAG exposes the math behind every stage:

| Stage | Method | What it explains |
|-------|--------|------------------|
| **Retriever** | Cosine similarity `S = q·Kᵀ / (‖q‖‖k‖)` + BM25, fused with Reciprocal Rank Fusion | *Why* each document was chosen |
| **Generator** | Forced inline citations `[PUB·EDITION·SECTION·CHUNK]` per claim (attention-matrix proxy) | *What* source content each claim used |
| **Verifier** | NLI entailment `P(ENTAILMENT \| source, claim)` | Whether each claim is grounded or hallucinated |
| **Trust Gate** | Shapley-style additive penalty scoring + threshold classifier | *How much* to trust the whole answer |
| **Compliance** | Semantic alignment + gap/violation analysis | How well requirements map to the corpus |
| **Audit** | Full JSON trace → SQLite | End-to-end reconstructible decision log |

Full mathematical spec: [`xai_math_spec.md`](xai_math_spec.md).

---

## Architecture

Four backend "blueprints" mounted under a single FastAPI gateway, plus a Streamlit control panel.

```
Query
  │
  ▼
[BP1 · Ingestion & RAG]      hybrid retrieval (dense + BM25 → RRF) → answer with per-claim citations
  │
  ▼
[BP2 · Verification & Trust] NLI claim verification → faithfulness score → Shapley trust gate
  │
  ▼
[BP3 · Compliance & Audit]   BRD-to-corpus mapping, gap/violation analysis → SQLite audit report
  │
  ▼
[Eval Harness]               offline RAGAS-style faithfulness / retrieval metrics
```

### Trust Gate (deterministic)

```
Overall_Score = 1.0
             − 0.5  × (unresolved edition conflict)
             − 0.3  × (per CONTRADICTION claim)
             − 0.1  × (per NEUTRAL claim)
             − 0.2  × (per claim with entailment < 0.5)
             − 0.05 × (per claim with 0.5 ≤ entailment ≤ 0.8)
             clamped to [0, 1]

Status = NON_COMPLIANT       if any contradiction / low-entailment / unresolved conflict
         NEEDS_HUMAN_REVIEW  if any neutral / medium-confidence claim
         SAFE                if all claims entailed with high confidence and no conflicts
```

Every gate decision can be fully reconstructed from the individual claim verdicts — no hidden state.

---

## Tech Stack

- **Backend:** Python, FastAPI
- **Retrieval:** ChromaDB (dense), `rank-bm25` (sparse), Reciprocal Rank Fusion
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, sentence-transformers)
- **Generation & NLI:** Google Gemini
- **Storage / Audit:** SQLite
- **Frontend:** Streamlit dashboard
- **Evaluation:** RAGAS-style offline harness

---

## Project Structure

```
White Box RAG/
├── backend/
│   ├── gateway.py                 # FastAPI entrypoint — mounts all routers under /api
│   ├── ingestion/                 # BP1: PDF parsing, chunking, hybrid retrieval, RAG generation
│   │   ├── pdf_parser.py
│   │   ├── pipeline.py
│   │   ├── retriever.py           # dense + BM25 + RRF fusion
│   │   └── rag.py                 # answer generation + claim parsing
│   ├── verification/              # BP2: trust & hallucination detection
│   │   ├── nli_engine.py          # entailment scoring
│   │   ├── scorecard.py           # RAGAS-style faithfulness
│   │   ├── edition_conflict.py    # cross-edition conflict detection
│   │   ├── trust_gate.py          # Shapley-style trust gate
│   │   ├── stability.py
│   │   └── mitigation.py
│   ├── compliance/                # BP3: BRD mapping, gap analysis, audit
│   │   ├── brd_parser.py
│   │   ├── mapper.py              # requirement → corpus alignment scoring
│   │   └── audit.py               # JSON audit report → SQLite
│   ├── eval/                      # offline evaluation harness
│   └── shared/                    # config, DB, Gemini client, XAI matrices, models
├── streamlit_app/                 # Streamlit UI (ingest, query, verify, compliance, audit, eval)
├── xai_math_spec.md               # full mathematical specification
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API key

### 2. Setup

```bash
git clone <repo-url>
cd "White Box RAG"

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # then add your GEMINI_API_KEY
```

### 3. Run the backend

```bash
cd backend
uvicorn gateway:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 4. Run the Streamlit UI

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

---

## Key API Endpoints

All routes are mounted under `/api`:

| Blueprint | Purpose |
|-----------|---------|
| **Ingestion** | Upload/parse documents, chunk, embed, and query with hybrid retrieval |
| **Verification** | Verify claims, score faithfulness, run the trust gate |
| **Compliance** | Map requirements to the corpus, produce gap/violation analysis and audit reports |
| **Eval** | Run the offline evaluation harness |

See the interactive OpenAPI docs at `/docs` for full request/response schemas.

---

## Testing

```bash
cd backend
pytest
```

---

## What Makes This Different

| | White-Box RAG | Standard RAG |
|---|---|---|
| Per-claim source attribution | ✅ Forced citations | ⚠️ Opaque |
| Hallucination detection | ✅ NLI entailment per claim | ❌ None |
| Trust score | ✅ Deterministic, reconstructible | ❌ None |
| Verification layer | ✅ Pure math, no LLM | — |
| Audit trail | ✅ Full JSON → SQLite | ❌ None |
| Compliance gating | ✅ SAFE / REVIEW / NON-COMPLIANT | ❌ None |

---

## License

MIT
