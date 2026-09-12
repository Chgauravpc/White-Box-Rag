# White-Box RAG — Explainable AI Governance & Trust Framework

A **domain-agnostic, compliance-grade governance layer** that sits on top of any RAG pipeline and makes it *auditable*. Instead of trusting an LLM's answer blindly, White-Box RAG decomposes every response into individual claims, verifies each one against its source, computes a mathematically-derived trust score, and emits a deterministic **SAFE / NEEDS-HUMAN-REVIEW / NON-COMPLIANT** gate — with a full audit trail persisted to SQLite.

The core idea: **the retrieval, ranking, conflict-detection, and trust-scoring layers use pure linear algebra and deterministic rules — no LLM in the verification loop.** That makes every governance decision reconstructible, explainable, and repeatable.

> Built as a hackathon project; the framework is domain-agnostic by design — ingestion accepts any free-text collection label, and both the NLI-premise normalization and the compliance-prompt persona are swappable domain profiles (`generic` by default; a `financial_reports` profile preserves the original RBI banking-compliance behavior for continuity). See `backend/shared/text_normalize.py` and `backend/compliance/domain_profiles.py`.

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
[BP3 · Compliance & Audit]   BRD-to-corpus mapping, gap/violation analysis → hash-chained SQLite audit report
  │
  ▼
[Governance]                 counterfactuals, conformal abstention, HITL review queue, framework mapping
  │
  ▼
[Eval Harness]               offline RAGAS-style faithfulness / retrieval metrics + conformal calibration
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

## Governance Suite

On top of the trust layer, five features make the system defensible to an auditor or regulator — all pure-math, no LLM in the decision loop. Full details: [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md).

| Feature | What it adds | Pillar |
|---|---|---|
| **Tamper-evident audit log** | Every audit record is SHA-256 hash-chained to its predecessor; any edit/deletion/reorder is detectable via `GET /api/audit/verify-integrity`. | Governance |
| **Human-in-the-loop review queue** | The `NEEDS_HUMAN_REVIEW` verdict routes to a real workflow with reviewer identity and an append-only, chained resolution history. | Responsible |
| **Counterfactual explanations** | For a non-SAFE answer, shows *what would change the verdict* — which claim, if removed, most improves the trust outcome (exact leave-one-out gate recomputation). | Explainability |
| **Conformal abstention** | Replaces the hand-tuned abstention ceiling with a split-conformal threshold calibrated to a statistical risk target α. | Responsible |
| **Regulatory mapping** | Static mapping of capabilities to **EU AI Act** and **NIST AI RMF** controls, with honest coverage. | Governance |

---

## Tech Stack

- **Backend:** Python, FastAPI
- **Retrieval:** ChromaDB (dense), `rank-bm25` (sparse), Reciprocal Rank Fusion
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, sentence-transformers) for retrieval; `BAAI/bge-large-en-v1.5` (1024-dim) for attribution/similarity scoring — deliberately independent models, both configurable/pinnable, see [`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md)
- **NLI Verification:** `cross-encoder/nli-deberta-v3-base` (local, sentence-transformers)
- **Generation:** Groq or OpenRouter (config-driven, OpenAI-compatible API), behind a multi-key/multi-provider pool with TPM-aware admission and failover (`backend/shared/llm_pool.py`) plus record/replay cassettes for reproducible/offline runs (`backend/shared/llm_cassette.py`) — see [`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md)'s Phase 2
- **Storage / Audit:** SQLite
- **Frontend:** Streamlit dashboard
- **Evaluation:** RAGAS-style offline harness + label-vs-prediction accuracy scoring (`backend/eval/scoring.py`) — precision/recall/F1/nDCG/AUROC, each with a confidence interval
- **CI:** GitHub Actions (`.github/workflows/tests.yml`) — the full test suite on every push/PR

---

## Project Structure

```
White Box RAG/
├── backend/
│   ├── gateway.py                 # FastAPI entrypoint — mounts all routers under /api
│   ├── ingestion/                 # BP1: PDF parsing, chunking, hybrid retrieval, RAG generation
│   │   ├── pdf_parser.py          # section detection + chunking; emits canonical chunk_key
│   │   ├── pipeline.py            # canonical /query orchestration
│   │   ├── retriever.py           # dense + BM25 + RRF fusion
│   │   └── rag.py                 # answer generation + claim parsing
│   ├── verification/              # BP2: trust & hallucination detection
│   │   ├── nli_engine.py          # entailment scoring
│   │   ├── scorecard.py           # RAGAS-style faithfulness
│   │   ├── edition_conflict.py    # cross-edition conflict detection
│   │   ├── trust_gate.py          # Shapley-style trust gate
│   │   ├── stability.py
│   │   ├── mitigation.py          # claim filtering + abstention + nonconformity_score
│   │   ├── counterfactual.py      # "what would change the verdict" (Governance)
│   │   └── conformal.py           # split-conformal abstention calibration (Governance)
│   ├── compliance/                # BP3: BRD mapping, gap analysis, audit
│   │   ├── brd_parser.py
│   │   ├── mapper.py              # requirement → corpus alignment scoring
│   │   ├── audit.py               # JSON audit report → SQLite
│   │   ├── frameworks.py          # EU AI Act / NIST AI RMF control catalog (Governance)
│   │   └── domain_profiles.py     # swappable persona/vocabulary (generic default; financial_reports)
│   ├── governance/                # HITL review queue router (Governance)
│   ├── eval/                      # offline evaluation harness + conformal calibration
│   │   ├── harness.py             # run_eval / run_calibration (end-to-end plane)
│   │   ├── detector.py            # detector plane: claim+premise+label → P/R/F1/AUROC, no retrieval/LLM
│   │   ├── retrieval.py           # retrieval plane: graded chunk labels → nDCG/recall@k, zero LLM calls
│   │   ├── adapters.py            # HaluEval / FEVER / generic → detector items
│   │   ├── fever_wiki.py          # on-disk SQLite index of FEVER's wiki dump
│   │   ├── scoring.py             # label-vs-prediction accuracy: P/R/F1/nDCG/AUROC + CIs
│   │   └── schema.py              # dataset v2 shape + validator + v1 compatibility loader
│   ├── scripts/                   # manual, live-credential operator tools (never pytest-collected)
│   └── shared/                    # config, DB, LLM client (Groq/OpenRouter), XAI matrices, models
│       ├── audit_chain.py         # tamper-evident hash-chain primitives (Governance)
│       ├── chunk_key.py           # canonical globally-unique chunk identity
│       ├── nli_policy.py          # single source of truth for NLI verdict penalties
│       ├── corpus_manifest.py    # frozen content-hashed corpus snapshot + drift diff (W3.1)
│       ├── hashing.py            # file/text content-hash primitives
│       ├── text_normalize.py      # profile-based NLI premise normalization (generic default)
│       ├── runconfig.py           # frozen run-configuration snapshot (provenance)
│       ├── config.py              # every threshold/temperature/model identity, env-overridable
│       ├── llm.py                 # call_llm()/call_llm_meta() — public LLM entry point (Phase 2)
│       ├── llm_pool.py            # multi-key/multi-provider pool: admission, failover, error classification
│       ├── llm_pool_config.py     # pool topology from env (key lists, provider order, rate limits)
│       ├── llm_cassette.py        # record/replay cassettes (LLM_MODE=record|replay)
│       └── llm_routes.py          # GET /api/llm/pool — live per-endpoint health/headroom
├── streamlit_app/                 # Streamlit UI (ingest, query, verify, compliance,
│                                  #   audit, review queue, regulatory mapping, eval)
├── .github/workflows/tests.yml    # CI: full test suite on every push/PR
├── docs/
│   ├── GOVERNANCE.md              # governance suite documentation
│   └── BENCHMARK_READINESS.md     # benchmark-readiness roadmap + progress log
├── xai_math_spec.md               # full mathematical specification
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A Groq API key (free tier at [console.groq.com](https://console.groq.com)) and/or an OpenRouter API key

### 2. Setup

```bash
git clone <repo-url>
cd "White Box RAG"

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # then set LLM_PROVIDER (groq|openrouter) and the matching API key
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
| **Compliance** | Map requirements to the corpus, produce gap/violation analysis and audit reports; `GET /api/audit/verify-integrity` (chain check), `GET /api/compliance/frameworks` (regulatory mapping) |
| **Governance** | HITL review of flagged audits: `GET /api/review/queue`, `POST /api/review/{id}/resolve`, `GET /api/review/{id}/history` |
| **Eval** | Run the offline harness; `POST /api/eval/detector` (detector plane: per-claim hallucination-detection accuracy, no retrieval/LLM); `POST /api/eval/retrieval` (retrieval plane, zero LLM calls); `POST /api/eval/corpus/freeze` + `GET /api/eval/corpus/{id}` (freeze a content-hashed corpus, check drift); `POST /api/eval/calibrate` (`force?`) + `GET /api/eval/calibration` (conformal abstention, reports `status`/`effective`) |
| **LLM pool** | `GET /api/llm/pool` — live per-endpoint health/headroom snapshot (never the raw key, only its fingerprint) |

See the interactive OpenAPI docs at `/docs` for full request/response schemas.

---

## Testing

```bash
cd backend
pytest
```

394 tests, entirely against mocked ML models (`backend/conftest.py`) and a fake `AsyncOpenAI` client (`backend/tests/test_llm_pool.py`) — no API key or model download needed. Runs automatically on every push/PR via `.github/workflows/tests.yml`.

---

## Benchmark Readiness

This project is mid-way through a self-directed benchmark-readiness effort: an independent review audited the eval harness against the standard of "can we credibly publish a hallucination-detection/retrieval number," found 22 issues (from "the harness never compares a prediction to a label" to a conformal-calibration fail-open bug), and a phased roadmap was built to close them — reproducibility, an external-benchmark-ready detector plane, a multi-provider key pool, and CI gating.

**Progress and the full roadmap:** [`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md).

---

## What Makes This Different

| | White-Box RAG | Standard RAG |
|---|---|---|
| Per-claim source attribution | ✅ Forced citations | ⚠️ Opaque |
| Hallucination detection | ✅ NLI entailment per claim | ❌ None |
| Trust score | ✅ Deterministic, reconstructible | ❌ None |
| Verification layer | ✅ Pure math, no LLM | — |
| Audit trail | ✅ Full JSON → SQLite, **SHA-256 hash-chained** | ❌ None |
| Compliance gating | ✅ SAFE / REVIEW / NON-COMPLIANT | ❌ None |
| Human oversight | ✅ HITL review queue w/ chained resolutions | ❌ None |
| Counterfactual explanations | ✅ "What would change the verdict" | ❌ None |
| Abstention threshold | ✅ Conformal-calibrated (risk target α) | ❌ None |
| Regulatory alignment | ✅ EU AI Act / NIST AI RMF mapping | ❌ None |

---

## License

MIT
