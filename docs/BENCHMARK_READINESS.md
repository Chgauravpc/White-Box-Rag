# Benchmark Readiness — Roadmap & Progress

This is the living record of the benchmark-readiness effort: making this
repo's evaluation numbers (retrieval quality, hallucination-detection
accuracy, governance effectiveness) reproducible, externally validated, and
CI-gated, instead of self-referential rates computed over a handful of
hand-written queries.

**Origin.** An Opus-level code review audited `backend/eval/` and the
verification pipeline specifically against "could we credibly publish a
number from this," and found 22 issues — from "the harness never compares a
prediction to a label" (nothing computed precision/recall/F1 against any
ground truth) to a conformal-calibration bug that silently *disabled*
abstention rather than calibrating it. A follow-up planning pass turned those
findings, plus explicit requirements from the project owner, into a phased
roadmap (9 phases, ~30 workstreams). This document tracks progress against
that roadmap and is the durable in-repo reference — update it whenever a
workstream lands, don't rely on chat history or git log alone to reconstruct
why a change happened.

---

## Roadmap phases (summary)

| Phase | Goal |
|---|---|
| **0** ✅ | Measurement prerequisites — chunk identity, config externalization, kill the conformal fail-open, CI skeleton |
| **1** ✅ | Scoring & provenance core — a real label-vs-prediction scorer; run durability |
| **2** ✅ | Multi-provider (Groq **+** OpenRouter) API key pool |
| **3** 🟡 | Dataset scale-up — hundreds of labeled items, graded retrieval ground truth (W3.1 corpus identity ✅; labeling not started) |
| **4** 🟡 | Evaluation planes (all three done), NLI verdict integrity, premise unification, async model calls, plane persistence - only multi-premise NLI remains |
| **5** 🟡 | External benchmark adapters (HaluEval/FEVER/generic converters done; RAGTruth + BEIR open; needs real data) |
| **6** | Ablation & baseline comparison (plain-RAG vs. full governance) |
| **7** | CLI entrypoints + two-tier CI (fast offline / nightly live) |
| **8** | Statistical reporting, deterministic tests, documentation |

Full workstream-level detail (file-by-file plan, effort sizing, open
decisions) lives in the roadmap artifact produced by the planning pass; this
document tracks what's actually landed, in this repo, against that plan.

---

## Completed

### Phase 0 — Measurement prerequisites ✅ (all four workstreams)

**W0.1 — Canonical chunk identity.** Two documents sharing a bare `section_id`
(both have a "1.1") used to collide: `_deduplicate_chunks` silently dropped
one document's chunk, and `RetrievalMatrix.chunk_ids` / `AttributionMatrix.chunk_ids`
were ambiguous across documents. New `shared/chunk_key.py` (`build_chunk_key`,
`resolve_chunk_key` with a content-addressed fallback for pre-migration data).
`ingestion/pdf_parser.py` now emits the canonical key as both the ChromaDB id
and a `chunk_key` metadata field; `ingestion/retriever.py`'s three duplicated
`_key()` f-strings converge on `resolve_chunk_key`; `ingestion/pipeline.py::_deduplicate_chunks`
rekeys on chunk identity instead of bare `section_id`. Also fixed the same
collision class in `EntailmentMatrix.passage_ids` (was keyed on bare
`source_section_id`). Tests: `test_chunk_key.py`, `test_pipeline.py`, plus two
new cases in `test_pdf_parser.py` exercising the real `ingest_pdf` code path.

**W0.2 — Configuration externalization + determinism.** ~12 thresholds that
were hardcoded module-level literals (in some cases duplicated identically
across 2-3 files — `WEAK_ATTRIBUTION_SCORE = 0.65` existed independently in
`xai_matrices.py`, `trust_gate.py`, and `mitigation.py`) are now centralized,
typed, env-overridable values in `shared/config.py`, with today's value as
the default (behavior-preserving). New `shared/runconfig.py::RunConfig` — a
frozen, content-hashed snapshot of everything that can make two runs produce
different numbers (provider/model/temperature/thresholds/model revisions/git
commit) — the seam future run-provenance persistence (Phase 1/2) will stamp
onto every eval run. `SentenceTransformer`/`CrossEncoder` now load with an
optional `revision=` pin (`XAI_EMBEDDING_MODEL_REVISION`, `NLI_MODEL_REVISION`
— unset by default; a wrong guessed revision is worse than no pin).
`GLOBAL_SEED` wired into `shared/llm.py::call_llm` as a best-effort `seed`
param. `torch`/`transformers`/`tokenizers` pinned explicitly in
`requirements.txt` (previously only transitive via `sentence-transformers`).
**Bonus correctness fix found by the new tests:** `trust_gate.py` was missing
a guard `xai_matrices.py::compute_shapley_contributions` already had, so a
claim that was both `CONTRADICTION` and had `entailment_score < 0.5` was
double-penalized in the Trust Gate but single-penalized in Shapley — same
status outcome, but the two numeric scores silently diverged from the
"mirrors Shapley exactly" invariant both modules' docstrings promise. Fixed;
regression test in `test_config.py`.

**W0.3 — Conformal calibration fail-open, fixed.** `verification/conformal.py`
now returns an explicit `status` (`NO_DATA` / `INSUFFICIENT_N` / `CALIBRATED`).
`load_active_threshold()` returns `None` (triggering the fixed-ceiling
fallback) whenever status isn't `CALIBRATED` — closing the bug where a
too-small calibration set produced a mathematical threshold of `+∞`, and
because `mean_penalty > +∞` is never true, that *silently disabled
abstention entirely*, making the "calibrated" system strictly more
permissive than before calibration. Added `min_n_for_alpha(α) = ⌈1/α⌉ - 1`
and `recommended_n_for_alpha`; `POST /api/eval/calibrate` pre-flight-checks
the dataset's answerable-item count against it *before* spending LLM calls,
and only applies a result as active when `status == CALIBRATED` (or an
explicit `force=true`) — verified against the exact scenario from the review
(5 answerable items, α=0.1 → `INSUFFICIENT_N` → `load_active_threshold()`
now correctly returns `None`). Extracted `mitigation.py::nonconformity_score`
as the single function both `should_abstain` and the eval harness call, so
the harness no longer reconstructs an approximation from
`claims_total`/`claims_stripped` that fabricated `0.0`/`1.0` on two
degenerate paths (empty retrieval; every claim stripped) — both now
correctly return `None`, excluded from calibration rather than injected as
data. Calibration state is now its own hash-chained SQLite table
(`calibrations`, mirroring `review_actions`) — previously the one governance
artifact outside the tamper-evident chain. Streamlit's Evaluation view shows
calibration status, the required n for the selected α, and a hard warning
(not a quiet fallback) when a stored calibration isn't actually effective.

**W0.4 — CI skeleton.** `.github/workflows/tests.yml`: the full test suite on
every push/PR (mocked models, no API key needed — see `backend/conftest.py`)
plus a non-blocking `ruff` pass (`E9,F` only — real-bug rules, not full style;
~18 pre-existing unused-import findings are tracked as backlog, not fixed
inline here). `backend/test_bp3.py` — never a real pytest test (no
assertions, ran a live LLM call chain unconditionally at import time) — moved
to `backend/scripts/manual_smoke_bp3.py` with a proper `if __name__` guard,
and `pytest.ini`'s `testpaths = tests` is now documented as intentional
rather than looking like an accidental exclusion.

### Phase 4 (partial) — Domain-neutral NLI premise normalization

**W4.3 — done ahead of the rest of Phase 4** (it was a hard prerequisite for
any external detector benchmark, so it was tackled before Phase 4's planes
work). `strip_table_noise` used to run on *every* NLI premise unconditionally
and delete any short all-caps line (would delete FEVER-style entities like
"NASA") and any numeric line (deletes exactly the evidence a numeric-fact
claim needs) — tuned to RBI-report PDFs, applied regardless of corpus domain.
New `shared/text_normalize.py`: three explicit profiles — `none` (untouched,
for external benchmark adapters), `generic` (strips only PDF page-marker
artifacts — new default), `financial_reports` (the old behavior, opt-in via
`PREMISE_NORMALIZER`). Every deletion is now recorded (`Claim.premise_deletions`)
instead of silently discarded. Same fix for the compliance layer: new
`compliance/domain_profiles.py` (`generic` default, `financial_reports`
opt-in) replaces the hardcoded "RBI regulatory analyst" persona and
KYC/FSR-style fixed enums in `brd_parser.py`/`audit.py`.

### Phase 1 — Label-vs-prediction scoring

**W1.1 — done.** Before this, nothing in the repo compared a prediction to a
label — `expected_abstain` was loaded and discarded; `abstention_rate` and
every other harness metric was an unlabeled rate, not an accuracy score. New
`eval/scoring.py`: `binary_classification`, `abstention_metrics` (+ the
wrongful-abstention rate a conformal α is supposed to bound),
`detector_metrics` (AUROC/AUPRC + per-error-type recall), `retrieval_metrics`
(hit@k/recall@k/precision@k/MRR/**nDCG@k**/MAP, bootstrapped over queries),
`trust_gate_metrics` (3-class confusion + Cohen's κ), `calibration_metrics`
(ECE), `paired_delta` (paired bootstrap + McNemar, for Phase 6 ablations).
Every metric returns `{value, n, stddev, ci_low, ci_high, method}`. New
`eval/schema.py`: dataset v2 shape + validator + a v1-compatibility loader so
the existing tiny datasets keep working (auto-upgraded). `eval/harness.py::_aggregate`
now emits a real `accuracy` block — abstention is scored unconditionally;
retrieval is scored whenever `relevant_chunk_keys` exist (today: none do, so
it correctly reports "no ground truth" rather than a fabricated number, until
W3.2 populates real labels).

### Phase 1 (continued) — Run provenance & durability

**W1.2 — done.** Before this, an eval run recorded nothing about what
produced it (finding #4) and a crash at item 299 of 300 lost all 299
completed results (finding #8). `eval/routes.py` now creates the `eval_runs`
row **before** execution (`status='running'`) and finalizes it in a
`finally`-equivalent block (`complete`/`failed`, with `error` set on
failure) — a crash leaves a diagnosable row, not silence. Each item is
written to a new `eval_items` table (`shared/database.py`) the instant it
completes, independent of whether the run's final aggregate step ever runs;
`get_completed_item_ids()` + `run_eval(..., resume_from_run_id=...)` let a
run resume by skipping items already recorded rather than starting over.
`eval/harness.py::capture_run_provenance()` snapshots everything that can
make two runs produce different numbers — `RunConfig` (W0.2), git commit, a
lightweight corpus manifest (`list_documents()` — not yet W3.1's fully
content-hashed version), the active calibration, and resolved model
identities (`xai_matrices.model_identity()`) — stamped onto the run row at
start. The abstention threshold is now read from disk **once** per run and
frozen (`run_query_pipeline(..., abstention_threshold=...)`), not re-read
per item, closing finding #20 for real (W0.3 only built the injection seam).
A rough `classify_error()` taxonomy (`RATE_LIMIT`/`JSON_PARSE`/`EMPTY_CORPUS`/
`TIMEOUT`/`LLM_API`/`CAPACITY_EXHAUSTED`/`MODEL_LOAD`/`UNKNOWN`) turns
`num_errors: 12` into a diagnosable breakdown; `asyncio.gather(...,
return_exceptions=True)` stops one item's exception (or a cancelled
coroutine, e.g. a disconnecting HTTP client) from losing every other
in-flight item. New `run_query_pipeline(..., eval_mode=True)` flag skips
requirement-mapping and audit-report generation — governance bookkeeping no
benchmark metric reads — cutting LLM calls per item roughly in half (verified
4→2 in a live smoke test) and, as a direct consequence, keeping benchmark
queries out of the production tamper-evident audit trail entirely (`log_id`
stays `None`, so embedding storage and chain finalization are skipped too).
The live `/query` HTTP route never sets `eval_mode`, so interactive-query
behavior is completely unchanged. `_load_dataset` is now also guarded
per-line — a single malformed JSONL line is logged with its line number and
skipped instead of raising `JSONDecodeError` and killing the run before a
single query executes. Also fixed while in this area: every
event timestamp across the codebase (`audit_logs`, `eval_runs`,
`calibrations`, `review_actions`, `AuditReport`'s default) now uses
timezone-aware UTC — previously `eval_runs` used naive `datetime.utcnow()`
while everything else used naive **local** `datetime.now()`, so two tables'
timestamps weren't even in the same time base (finding #22). And
`compliance/audit.py`'s LLM narrative call now degrades instead of raising
on a parse/API failure (finding #17) — a malformed audit narrative no longer
fails the whole query; every mathematically-derived field (claims,
verifications, trust_gate, etc.) is still populated, with the failure
recorded in the new `AuditReport.audit_report_error` field instead.

**Deliberately deferred, not forgotten:** the original workstream sketch for
`eval_runs` also listed `provider_summary_json`, `pipeline_profile`,
`llm_mode` (live/record/replay), `seed`, `trial_index`, and an
`eval_run_endpoints` table. None of those exist yet **as `eval_runs`
columns**, on purpose — `pipeline_profile`/`seed`/`trial_index` are Phase 6
(profile registry) and Phase 8 (repeated trials) concepts. `llm_mode` and
per-endpoint provenance now exist as *runtime* concepts (Phase 2, below —
`LLM_MODE` env var, `LLMCallRecord`/`AuditReport.llm_calls`), but
`capture_run_provenance()` doesn't yet stamp "which mode/pool config was
active" onto the `eval_runs` row itself — that integration is left for
whichever of Phase 6/8 first needs to compare runs made in different modes.
Adding empty columns for a comparison nothing needs yet would be schema for
its own sake; `_ensure_column`'s idempotent-migration pattern (already used
throughout `shared/database.py`) makes adding them cheap when that day comes.
`eval_mode` (a plain boolean) is the one profile axis that exists today, and
is captured now.

`parent_run_id` **is** a real column (unlike the ones above), but isn't
populated yet: `run_eval(..., resume_from_run_id=...)` continues a resumed
run under the SAME `run_id` (new `eval_items` rows appended, `eval_runs`
re-finalized over the merged prior+new results) rather than spawning a new
child row linked via `parent_run_id`. Simpler, and it fully solves finding
#8 (no completed item is ever lost) — the tradeoff is that a run resumed
long after its original start reports a `started_at`→`finished_at` span
covering the whole gap, not just the active work. `parent_run_id` is left
available for Phase 8's repeated-trials use case, where it's the more
natural fit anyway (one root run, several sibling trial runs).

### Phase 2 — Multi-provider (Groq + OpenRouter) API key pool ✅

Addresses finding #9 (no seam for multiple keys per provider — a
process-global singleton client, no per-key rate-limit state, key-blind
retry) and the per-call-provenance half of finding #4.

**W2.1 — Configuration & pool topology.** New `shared/llm_pool_config.py`.
Keys become **lists**: `GROQ_API_KEYS` / `OPENROUTER_API_KEYS` (comma-
separated) — unset falls back to the legacy singular `GROQ_API_KEY` /
`OPENROUTER_API_KEY`, so an existing single-key `.env` is a "pool of one"
with zero changes required. `LLM_PROVIDER_ORDER` defaults to **just** the
legacy `LLM_PROVIDER` value — i.e. even if both providers' keys happen to be
set, the pool talks to exactly one provider unless you explicitly list more
than one (this was worth a dedicated test: an earlier version of this
function auto-appended any provider with configured keys regardless of
opt-in, which would have silently started using both providers the moment a
second key showed up in `.env` — caught by
`test_llm_pool_config.py::test_default_order_is_single_legacy_provider_only`
before it shipped). `LLM_ALLOW_CROSS_PROVIDER_FAILOVER` and
`LLM_STRICT_SINGLE_MODEL` both default to `false`. Per-key RPM/TPM limits
(`GROQ_RPM_PER_KEY` etc.) ship with conservative placeholder defaults, not a
promise about your plan tier — see "Open decisions" below.

**W2.2 — Data structures & state.** New `shared/llm_pool.py`. `LLMEndpoint`
(frozen: provider, base_url, model, raw key — `key_fingerprint` is a
SHA-256(key)[:12] property, the raw key is never logged or returned from any
endpoint). `EndpointState` (inflight count; a `_SlidingWindow` of
`(timestamp, amount)` pairs per RPM/TPM, pruned on an injectable clock —
`time.monotonic` by default, swappable for tests — rather than real 60s
waits; cooldown/health/error counters). `LLMPool` holds one `AsyncOpenAI`
client **per key** (never shared — connection pools and auth headers can't
interleave) behind an `asyncio.Lock`/`Condition` pair. Admission and release
are split: `reserve()` books the estimated token cost into the window the
instant an endpoint is chosen (so two concurrent calls can't both see the
same headroom and over-admit), `release()` corrects the window to the
*actual* token usage once the response lands (or records the failure
category if it didn't).

**W2.3 — Selection, rate-limit intelligence & failover.** Admission ranks
available endpoints by **highest remaining TPM headroom** (not round-robin —
round-robin is what makes N keys trip their limits in lockstep, since
they'd all get hit at the same cadence). `classify_llm_error()` replaces the
pre-pool client's `"500" in str(e)` substring match with typed-exception
dispatch (`openai.RateLimitError` → `FAILOVER` + 5s cooldown,
`AuthenticationError`/`PermissionDeniedError` → `DISABLE_ENDPOINT`
permanently, `APITimeoutError`/`APIConnectionError`/5xx → `RETRY_SAME_KEY`
with exponential backoff, 4xx/`BadRequestError` → `FATAL`, raised
immediately without burning the remaining attempt budget). Response headers
(`x-ratelimit-remaining-requests`/`-tokens`) are opportunistically reconciled
into the local window when a provider sends them — best-effort, since
neither Groq nor OpenRouter document this as guaranteed, so headroom falls
back to the local sliding-window estimate whenever a header is absent.
`LLM_STRICT_SINGLE_MODEL=true` pins the provider chosen on a call's first
attempt for every retry within that call, even when
`LLM_ALLOW_CROSS_PROVIDER_FAILOVER=true` — closing the benchmark-validity
trap where a Groq→OpenRouter failover mid-run silently changes which model
produced an item. An item that can't get capacity within the pinned
provider raises `AllEndpointsExhausted` (message contains "capacity
exhausted" so `eval/harness.py::classify_error` buckets it correctly) rather
than ever falling over to a different model.

**W2.4 — Public interface of `shared/llm.py` + per-call provenance.**
`call_llm(prompt, system_instruction=None, temperature=0.2, max_retries=3,
seed=None) -> str` keeps its **exact** pre-pool signature and return type —
`rag.py`, `audit.py`, `mapper.py`, `brd_parser.py` are untouched. New
`call_llm_meta(...) -> LLMCallResult` returns endpoint id, provider, model,
key fingerprint, token counts, latency, and retry history for callers that
want it. New `llm_call_scope()` (a `contextvars.ContextVar`-based collector)
+ `get_current_llm_calls()`: wrap a block in the scope and every
`LLMCallRecord` made anywhere inside it — across any number of nested
function calls — is collected, without threading a parameter through every
intermediate function. This is correct under the eval harness's
`asyncio.gather` fan-out because each spawned coroutine becomes its own
`asyncio.Task`, and each `Task` gets its own **copy** of the context at
creation — one item's collected calls can never leak into a concurrent
sibling item's list. `ingestion/pipeline.py::run_query_pipeline` is now a
thin wrapper that opens one scope around the whole query and stamps the
collected list onto the returned `AuditReport.llm_calls` (new field,
`shared/models.py::LLMCallRecord`) before returning — `llm_call_count` (the
existing plain per-call-site counter) is untouched, so `llm_calls` is
additive provenance, not a replacement; it can be shorter than
`llm_call_count` for any future code path that calls `call_llm` outside a
scope. New `GET /api/llm/pool` (`shared/llm_routes.py`) returns a live
per-endpoint snapshot (health, inflight, RPM/TPM headroom, error counts,
`key_fingerprint`) — verified never to render the raw key even in the
snapshot's string form (`test_snapshot_never_exposes_raw_key`).

**W2.5 — Record/replay cassettes.** New `shared/llm_cassette.py`. A cassette
JSONL file maps `sha256(provider|model|temperature|seed|system|prompt) ->
response text`. `LLM_MODE=live` (default): unchanged behavior, no cassette
touched. `LLM_MODE=record`: call the pool normally, then append the
(key, response) pair. `LLM_MODE=replay`: look up the key and return it
directly — **never** touches the pool (no API key needed at all, verified by
a test that makes `get_pool()` raise `AssertionError` if it's ever called
in replay mode) — and a miss is a hard `KeyError` naming the provider/model,
never a silent fall-through to a live call, so a replay run is either fully
offline-reproducible or fails loudly. `cassette_scope(name)` (another
contextvar) lets a caller (Phase 3+'s eval harness integration) name the
active cassette per dataset/run; unset, it's `"default"`. This directly
serves three separate mandatory requirements from the original review: it's
the only way to make an end-to-end run genuinely bit-reproducible (provider
`seed` is documented as best-effort by both Groq and OpenRouter, never a
guarantee); it lets CI run the full pipeline with zero API keys and zero
flakiness; and it will let Phase 6 ablations differ *only* in the governance
layer rather than being confounded by fresh LLM resampling. **Not yet
wired**: nothing in this repo sets `LLM_MODE=record`/`replay` automatically
today — it's an operator-set env var, and no dataset ships a cassette yet.
That wiring (naming a cassette after the dataset, a CI job that runs in
replay mode) is follow-on work for whichever of Phase 5/7 needs it first.

**W2.6 — `EVAL_CONCURRENCY` as a function of pool size.** New
`shared/llm_pool_config.py::effective_eval_concurrency(default)`: when
`EVAL_CONCURRENCY_AUTO=true`, item-level concurrency in `eval/harness.py`
becomes the pool's total call-level capacity (Σ each endpoint's
`max_concurrency`) instead of the static `EVAL_CONCURRENCY` sized for one
key — falling back to `default` whenever auto-scaling is off, no pool is
configured (the test suite has no live keys), or the pool reports zero
capacity. **Two of the three non-key ceilings the original workstream
flagged are deliberately not addressed yet**, and are called out here
rather than silently left: the audit `_chain_lock` in `ingestion/pipeline.py`
still serializes finalization (harmless today since `eval_mode=True` skips
audit-report generation entirely, so benchmark runs never reach that lock —
but it would matter the moment a non-`eval_mode` run needs to scale past a
few keys); and NLI/embedding calls in `shared/xai_matrices.py` are still
synchronous, blocking the event loop rather than running in
`asyncio.to_thread` — fine at low concurrency, a real ceiling above roughly
4 concurrent items once the key pool stops being the bottleneck. Both are
scoped-out follow-on work, not forgotten.

**Honest scoping.** RPM/TPM-per-key defaults are placeholders (see
`shared/llm_pool_config.py`'s docstring) — the "how many keys, what plan
tier" open decision from the original roadmap is still the project owner's
call; the pool works correctly at any count (including the default single
key most `.env` files still have), it just can't estimate real throughput
until those numbers are real. No dataset or CI job uses `LLM_MODE=record`/
`replay` yet (see W2.5). The rate-limit-header reconciliation in W2.3 is
best-effort and untested against real Groq/OpenRouter response headers
(no live keys were exercised in this session) — it degrades to the local
sliding-window estimate whenever a header is absent, which is the common
case until verified otherwise.

**Tests:** `tests/test_llm_pool.py` (31 — endpoint state math, error
classification against real `openai` exception types, admission ordering,
failover/disable/fatal paths, strict-single-model pinning, key-fingerprint
never-leaks), `tests/test_llm_pool_config.py` (12 — key/order defaults and
the single-provider-by-default regression above), `tests/test_llm_cassette.py`
(12 — key determinism, record/replay round-trip, malformed-line guard),
`tests/test_llm.py` (10 — `call_llm`/`call_llm_meta` back-compat, scope
collection/isolation, replay hit/miss). All against a fake `AsyncOpenAI`
client and an injectable clock — no real network calls, no real 60-second
waits.

### Phase 3 (partial) — Corpus identity & the frozen manifest

**W3.1 — done. W3.2 (the labels themselves) not started.**

The goal of Phase 3 is hundreds of items with graded `relevant_chunk_keys`. A
planning review made the case that labeling *first* would have burned the
labeling budget: a `chunk_key` encodes a **position**
(`publication|edition|section|ordinal`), so a label is only meaningful against
one exact corpus state, and nothing in the repo could detect that state
changing. Three defects had to be fixed before a single label was worth
writing.

**Chunk ordinals were document-global, not per-section.**
`pdf_parser.py::ingest_pdf` did `for i, chunk in enumerate(all_chunks)` over
every chunk in the document and passed `i` as the ordinal — while
`chunk_key.py::build_chunk_key` documents that field as "disambiguates
multiple chunks *within the same section*". So one extra paragraph in section
1.1 renumbered every chunk in the rest of the document, silently re-pointing
every label at different text. Ordinals now restart per section; regression
test `test_earlier_section_growth_does_not_renumber_later_sections` grows an
early section and asserts a later section's key is untouched (it fails under
the old behavior, where `2.1|0` became `2.1|2`).

**Re-ingestion left orphans.** `collection.upsert(ids=...)` only overwrites the
ids it is handed, and nothing anywhere deleted chunks. Re-ingesting a document
that now yields fewer chunks left the previous ingest's surplus live in the
collection and still retrievable. `ingest_pdf` now deletes the
`(publication, edition)` slice before upserting.

**`documents` rows duplicated on re-ingest.** `insert_document` was an
unconditional INSERT, so re-ingesting produced two rows with the same identity
and different `chunk_count`s, and `list_documents()` returned both — meaning
every provenance snapshot double-counted the corpus. Replaced by
`upsert_document` (keyed on publication+edition); `list_latest_documents()`
deduplicates historical rows for existing databases. `insert_document` was
deleted rather than left dead, so the bug can't be reintroduced by reuse.

**The manifest itself.** New `shared/corpus_manifest.py` (in `shared/`, not
`eval/` — it is storage domain knowledge, and `ingestion -> eval -> ingestion`
would be an import cycle). `build_manifest()` reads chunk text from ChromaDB
(authoritative — it is what ingestion writes and retrieval reads) and
reconciles it against SQLite, recording any disagreement as a first-class
`inconsistencies` finding instead of silently resolving it. Every chunk is
recorded as `(chunk_key, sha256(normalized_text))`: the key says *where* a
chunk was, the hash says *whether it is still the same chunk*. New
`shared/hashing.py` holds both primitives — `hash_file` (chunked; promoted
from `harness.py::_hash_file`, which read whole files into memory) and
`hash_text`, which NFC-normalizes and collapses whitespace under an explicitly
versioned `TEXT_HASH_NORMALIZER` so a PyMuPDF bump that changes only
line-joining doesn't report false drift.

`diff_manifest(frozen, live)` classifies drift in the terms labels actually
die from: **moved** (same text, new key — labels are recoverable by
re-pointing), **content_changed** (same key, different text — the
silent-invalidation case, where a label still "resolves" and is wrong), and
**removed**/**added**. `resolve_labels()` reports what fraction of a dataset's
labels still resolve, and deliberately reports key-only labels (no
`text_sha256`) as `unverifiable` rather than assuming them correct.
`eval/schema.py` gained an optional `text_sha256` per `relevant_chunk_keys`
entry, and `relevant_chunk_keys_as_dict(item, manifest=...)` resolves by
content first. An empty corpus yields the sentinel `corpus_hash = "EMPTY"`,
never a real-looking digest of nothing.

**Provenance is now real.** `capture_run_provenance` stamps the content-hashed
manifest (hashes and counts, not a second copy of the corpus) instead of a
bare `list_documents()` snapshot — closing the gap W1.2 explicitly flagged.
`RunConfig` gained `chunk_max_tokens`, `chunk_overlap_tokens` and
`parser_version`: those were missing, so `content_hash()` was blind to the
single change most able to invalidate a retrieval label. `PARSER_VERSION` is a
hand-bumped constant in `pdf_parser.py` (a git commit is too coarse — every
commit would invalidate a frozen corpus).

**Bonus correctness fix.** `eval/harness.py::_aggregate` reported a
`retrieval_hit_rate`/`retrieval_mrr` that compared `expected_section_ids`
(bare ids like `"1.1"`) against `retrieved_chunk_ids`, which are canonical
chunk keys — `cid in expected` could never be true, so the metric was
structurally pinned at **0.0** for every item that had any ground truth, and
was rendered in the Streamlit trend chart next to the correct
`accuracy.retrieval` block. The reason it survived is instructive: its unit
test fed `retrieved=["2.1", "1.1"]`, bare section ids the pipeline never
produces, so the arithmetic was verified against data that cannot occur.
Removed; retrieval is now scored only by `scoring.retrieval_metrics` over
graded `relevant_chunk_keys`, and the Streamlit view reads `nDCG@5`/`recall@5`
from there (or says plainly that a dataset has no graded ground truth).

**Tests:** `tests/test_corpus_manifest.py` (24 — normalization, the drift
taxonomy, label resolution, empty-corpus sentinel, SQLite/Chroma
reconciliation) plus three new `test_pdf_parser.py` cases for per-section
ordinals and orphan deletion. 280 tests pass.

**Still open — this is corpus identity, not a corpus.** No benchmark corpus is
frozen and no item is labeled yet; `golden_dataset.jsonl` is still 3 items and
`calibration_dataset.jsonl` 8, with zero `relevant_chunk_keys` between them.
The remaining decisions are the project owner's: which corpus (public and
scriptable, so a published number is reproducible, vs. private), the labeling
budget, and whether to hand-label at all versus going to Phase 4+5 first,
where RAGTruth/HaluEval/FEVER arrive **with** labels. Note that Phase 5 is
blocked on Phase 4's plane split, not on Phase 3 — hand-labeling is not on the
critical path to an external benchmark number. Also deliberately deferred:
drift **policy** (should a `corpus_id` mismatch refuse to start a run, as W0.3
does for calibration?) and any CI gating on it — there is nothing to gate
until labels exist.

### Phase 4 (partial) — NLI verdict integrity (finding #10, first half)

**Done: the empty-premise guard and the shared penalty policy. Multi-premise
verification deliberately NOT shipped (see below).**

**An empty premise was being scored.** `xai_matrices.py::verify_claims_batch`
built its NLI pairs as `(focused, claim)` and sent the whole batch to the
CrossEncoder — including pairs whose premise was `""`. A CrossEncoder handed
`("", claim)` does not error; it returns a perfectly well-formed 3-way
softmax, and `argmax` turns that into a verdict. So a claim with **no evidence
at all** received a real-looking verdict and confidence, indistinguishable
downstream from a measured one. For a hallucination detector that is close to
the worst available failure mode: the fabricated number flows into the trust
gate, the Shapley contributions, the faithfulness score and the audit record.

The model is now never called on an empty premise. Such claims get
`NOT_ENOUGH_INFO` with structural zeros and a new
`VerificationResult.evidence_status` recording *why* there was no premise:
`no_premise` (the claim had no source passage — an unattributable sentence, or
empty retrieval) or `normalizer_deleted_all` (a passage existed and
`PREMISE_NORMALIZER` removed every line of it, which signals a normalizer
misconfiguration and now also emits a `logger.warning` rather than a silent
NEI). `evidence_status` exists precisely so nothing has to infer "was this
measured?" from a score of 0.0. An all-empty batch also no longer calls
`predict([])`.

**Honest strip reasons.** `mitigation.py` stripped these claims via
`entailment_score < STRIP_ENTAILMENT_FLOOR` and labelled them
`"low_confidence"` — attributing a judgement to a model that never ran on the
claim. They are still stripped (retaining an unverifiable sentence in a
mitigated answer would be a governance regression, and this module cannot
distinguish legitimate discourse glue from an ungrounded assertion without
attribution context it isn't given), but the decision is now an explicit
branch and the reason reads `no_evidence:<status>`.

**One penalty policy instead of two copies.** `trust_gate.py` and
`xai_matrices.py::compute_shapley_contributions` both convert
`(verdict, entailment_score)` into penalties, and both docstrings promise they
mirror each other. They had drifted **three** times. The third was live and
undetected: Shapley matched the bare literals `"CONTRADICTION"` / `"NEUTRAL"`,
while the trust gate matched the `NLIVerdict` enum *including* its canonical
aliases. So a `CONTRADICTED` verdict was a contradiction to the gate and a
non-contradiction to Shapley — which then also charged the confidence-band
penalty the gate had skipped, silently diverging the two `overall_score`s.
`NOT_ENOUGH_INFO` diverged the same way, and the empty-premise fix above is
exactly what would have started producing it. New `shared/nli_policy.py` is
now the single source of truth (`nli_penalty_flags`, plus verdict predicates
accepting either spelling and either a string or an enum); both modules call
it. The tests assert the two computations against **each other** across every
verdict spelling and confidence band, so a future divergence fails regardless
of what the penalty constants happen to be.

**`POST /api/verify/` was dead.** `verification/routes.py` unpacked three
values from `verify_all_claims`, which has returned a 4-tuple since W4.3 added
`premise_deletions`. Every call raised `ValueError: too many values to
unpack`. No test touched the route. Fixed, and
`tests/test_verification_routes.py` now pins the arity contract.

**Deliberately NOT shipped: multi-premise verification.** The other half of
finding #10 is that each claim is scored against exactly one passage (its own
cited `source_passage`). The obvious fix — score against all K retrieved
chunks and take the max entailment — was designed, reviewed, and rejected:

* `E[max over K]` rises with K for a noisy model, so the measured
  hallucination rate would become a function of `FINAL_TOP_K`. A retrieval
  change would then show up as a detector improvement — the exact confound the
  Phase 4 plane split exists to *remove*.
* Max-entailment destroys contradiction detection in this repo's flagship
  case: if one chunk entails and an older edition contradicts, max selects the
  entailing chunk and the conflict `verification/edition_conflict.py` exists
  to surface disappears.
* FEVER aggregates over a **fixed** evidence budget with a precedence rule
  (any REFUTES → REFUTED), and RAGTruth treats the whole provided context as
  one premise. Neither takes a raw max over an unbounded retrieved set.

More fundamentally: there is still **no measurement of NLI verdict quality**
in this repo (`scoring.detector_metrics` is written but called from nowhere,
and `schema.reference_claims` is validated but read by nothing). Shipping a
method change with no ground truth means Phase 5's first external benchmark
would measure a method chosen by intuition. The correct order is: build the
detector plane, get labels, run both arms, then choose. Multi-premise is
therefore Phase 4's remaining work, to land behind `NLI_MULTI_PREMISE`
(default off) once it can be justified with a number.

**Known prerequisites for that work, recorded now rather than rediscovered:**
`verify_claims_batch` and `build_entailment_matrix` apply *different* premise
treatment to the same (claim, passage) pair, so the XAI entailment matrix the
UI renders can already disagree with the verdict beside it — unify before
widening. Every NLI/encoder call is synchronous inside `async def` (there is
no `to_thread`/`run_in_executor` anywhere in `backend/`), so under
`EVAL_CONCURRENCY` each query's NLI batch blocks the event loop; multi-premise
multiplies that window by K, and `_encoder`/`_nli` are shared module-level
instances that need a semaphore before any threading.
`extract_relevant_sentences` would run N×K, re-parsing and re-embedding the
same chunks once per claim — it needs a per-query per-chunk sentence/embedding
cache first. And both `stability.py`'s independent NLI path and the persisted
conformal threshold (calibrated against the current `mean_penalty`
distribution) would need to follow.

**Metric-breaking change, by design.** The empty-premise guard was changed
outright rather than config-gated: a softmax over an empty string is not
behavior worth preserving, and gating it would mean the default stays "report
fabricated confidence". `mean_faithfulness_*`, `abstention_rate` and the trust
status distribution will move for any corpus where claims go unattributed —
previously those claims drew a random verdict from the model. Runs from before
this change are not comparable to runs after it.

**Tests:** `tests/test_nli_policy.py` (19) and
`tests/test_verification_routes.py` (3). 302 tests pass (was 280).

### Phase 4 (continued) — the detector plane

**Done: claim-verification quality is now measurable. This is the first time
anything in this repo compares an NLI verdict to a label.**

Before this, `eval/scoring.py::detector_metrics` — AUROC, AUPRC, per-error-type
recall, all correctly implemented — was called from **nowhere**, and
`eval/schema.py`'s `reference_claims` was validated and read by nothing. So the
hallucination detector at the centre of the system had never been scored
against ground truth. Every number the repo could produce described retrieval,
generation or governance; none described *detection*.

**What the plane is.** New `eval/detector.py`. Given rows of
`(claim, premise, label)` it runs claim verification and nothing else — no
retrieval, no generation, no LLM call, no database. That matters for two
reasons: it costs nothing per item (so it can run over a full external
benchmark rather than a handful of queries), and it isolates the detector from
retrieval, so a retrieval change can no longer masquerade as a detection
improvement. It is also the exact shape FEVER, HaluEval and RAGTruth already
ship in, which is what makes the Phase 5 adapters small.

**Three deliberate choices, each of which could have quietly invalidated the
number:**

* **The premise is not normalized** (`profile="none"` by default). The
  normalizer profiles exist to strip PDF artifacts from *this* corpus; running
  them over a benchmark's own evidence text would produce a number nobody
  outside this repo could reproduce. `shared/text_normalize.py` documents the
  `none` profile as existing for precisely this.
* **The prediction is the shipped decision rule.** `y_pred_flagged` comes from
  `verification/mitigation.py::_should_strip` — the same function the live
  pipeline uses to decide whether a claim survives into the mitigated answer.
  Reimplementing "flagged" inside the harness would measure a detector this
  repo does not ship; a test asserts the real function is the one called.
* **The positive class is `label != SUPPORTED`**, grouping REFUTED (evidence
  contradicts) with NEI (evidence is silent), because the system's job is to
  assert neither. They are *also* reported separately: the label is passed
  through as `error_types`, so `recall_by_error_type` says "catches X% of
  contradictions, Y% of unsupported" instead of hiding the difference in one
  aggregate. `verdict_confusion` (gold label × raw NLI verdict) additionally
  separates "the NLI model is wrong" from "the strip threshold is
  miscalibrated" — a single precision/recall pair cannot distinguish those,
  and they have completely different fixes.

**Benchmark label aliases** are mapped up front (`SUPPORTS`/`REFUTES`/`NOT
ENOUGH INFO` and the common binary spellings), and an *unrecognized* label is
rejected rather than coerced — silently bucketing an unknown label would
corrupt the ground truth itself.

**`POST /api/eval/detector`** runs it. Deliberately **not** persisted to
`eval_runs`: that table's columns and metric shape describe an end-to-end run,
and writing a detector run into it would make two incomparable things
indistinguishable rows (and let `resume_from_run_id` resume across planes).
Persistence lands with the rest of the plane split, along with a plane column.

**`eval/detector_smoke.jsonl`** (12 items, balanced 4 SUPPORTED / 4 REFUTED /
4 NEI) ships so the endpoint is runnable immediately. It is a *machinery*
check, not a benchmark — it deliberately includes the error classes that
separate detectors (numeric swap, year swap, negation, unsupported
elaboration) plus one empty-premise item that exercises the guard end to end.
Tests assert it stays loadable, balanced and unique-id'd so it cannot rot.

**Not yet measured, and worth being explicit about:** running the plane under
the test suite exercises the *mocked* NLI model, so the metrics it produces
there are noise (AUROC ≈ 0.5, as expected). A real number requires the real
`cross-encoder/nli-deberta-v3-base` weights and a real labeled dataset. The
plumbing is now in place for both; neither has been run here.

**Tests:** `tests/test_detector.py` (24). 326 tests pass (was 302).

### Phase 4 (continued) — the retrieval plane, and freezing a corpus for real

**Done: retrieval can be scored without generating, and W3.1's manifest is
now reachable and actually used.**

`run_query_pipeline` calls `rag_query` unconditionally, so scoring retrieval
over 1,000 queries meant 1,000+ LLM calls. That is what made a BEIR-scale
sweep unaffordable and a per-commit retrieval regression check impossible. New
`eval/retrieval.py` runs `hybrid_retrieve` and scores it against graded
`relevant_chunk_keys` — **zero LLM calls**, no generation, no verification, no
audit write. A test pins that by making `call_llm` raise.

It reuses the v2 dataset shape rather than inventing a third format, so the
same dataset drives the end-to-end and retrieval planes and the two cannot
disagree about what "relevant" means. v1 datasets still load via the schema's
auto-upgrade.

**Two failure modes it is built to avoid**, both of which this repo has
already shipped once in some form:

* *Identity namespace.* Retrieved ids and labeled ids must be the same kind of
  string. The removed `retrieval_hit_rate` compared bare section ids against
  canonical chunk keys and was therefore pinned at 0.0 forever. The plane
  resolves retrieved chunks through the same `resolve_chunk_key` the retriever
  uses, and the tests assert real hits and rank sensitivity rather than just
  "a number came out".
* *Unlabeled items counted as misses.* An item with no ground truth cannot
  speak to retrieval quality; averaging it in as a zero would understate the
  system in proportion to how incomplete the labels are. Such items are
  excluded and counted separately (`n_unlabeled`).

**Label durability — where the corpus manifest finally pays off.** A
`relevant_chunk_key` encodes a position, so re-chunking moves it and a label
silently starts pointing at different text; the metric keeps producing a
number, just not the one you think. Pass `corpus_id` and labels carrying a
`text_sha256` are resolved by **content** first, following a moved chunk
instead of being scored as a miss, and the response carries a `label_health`
block saying what fraction of labels still resolve. Measured end to end on a
corpus that was frozen and then deliberately re-chunked:

| label | manifest | hit@5 | label_health |
|---|---|---|---|
| key only | none | 0.0 | *(no signal — silently wrong)* |
| key only | frozen | 0.0 | `missing=1` — wrong, but visible |
| key + `text_sha256` | frozen | **1.0** | `moved=1` — followed the content |

The first row is the one that matters: without any of this, a stale label
produces a confident, wrong 0.0 and nothing anywhere says so.

**The manifest is now reachable.** W3.1 built `build_manifest`/`save_manifest`
and nothing ever called them, so no corpus could actually be frozen.
`POST /api/eval/corpus/freeze` snapshots the live corpus under a `corpus_id`
and returns its hash, counts, chunking config and any SQLite/ChromaDB
`inconsistencies` (surfaced, not buried — a manifest over a corpus whose two
stores disagree is a manifest of a problem). `GET /api/eval/corpus/{id}`
returns the frozen summary plus live drift, classified as
moved / content_changed / removed / added. Asking for durability checking
against a manifest that does not exist is a 404, not a quietly weaker
guarantee.

**Endpoints:** `POST /api/eval/retrieval`, `POST /api/eval/corpus/freeze`,
`GET /api/eval/corpus/{corpus_id}`.

**Tests:** `tests/test_retrieval_plane.py` (14). 340 tests pass (was 326).

**Still missing, deliberately:** plane-aware persistence. Neither the
detector nor the retrieval plane writes to `eval_runs`, because that table's
columns and metric shape describe an end-to-end run — mixing planes there
would make incomparable things indistinguishable rows and let
`resume_from_run_id` resume across planes. A `plane` column plus a per-plane
metric envelope is the remaining piece of the split.

### Real-model verification, and Phase 5 — external benchmark adapters

**Done: the real NLI model has now actually been run, a latent model-swap bug
was found and fixed, and HaluEval/FEVER/generic converters exist.**

#### The real model had never been run

Every number this repo had ever produced — including the detector plane's own
— came from `conftest.py`'s random mock. Before building anything on top, the
12-item smoke set was run through the real
`cross-encoder/nli-deberta-v3-base`. It scores **12/12**. That is a
*machinery* result, not a quality result: the items are deliberately
clear-cut. What it genuinely verifies is that the real model produces sane
probabilities through this code path, that the empty-premise guard fires
(`det-s12` → `no_premise`, never scored), and that numeric- and year-swap
claims come back as CONTRADICTION.

#### The label-order landmine it exposed

The 3-way output order is a property of the **checkpoint**, not of NLI. The
code hardcoded deberta's `(contradiction, entailment, neutral)` in three
places — `verify_claims_batch`'s label list and its `[0]/[1]/[2]` index picks,
`build_entailment_matrix`'s returned labels, and `stability.py`'s `s[1]`. But
`NLI_MODEL` is env-overridable, and **roberta-large-mnli — the most natural
swap — emits `(contradiction, neutral, entailment)`**. Changing that one env
var would have silently exchanged "entailment" and "neutral" for every claim
in the system: hallucinations scored as grounded, nothing raising, every
governance artifact downstream quietly wrong.

The order is now resolved once from the checkpoint's own `id2label`
(`xai_matrices.NLI_LABELS` + index constants + `entailment_score_of`). A
config that isn't a real mapping (the test suite's mock) falls back silently;
one that IS a mapping but doesn't describe a three-way NLI head logs an error
stating that verdicts from that model are not trustworthy. Verified against
the real checkpoint's config and three unambiguous probes; the smoke set
returns identical results before and after.

#### Adapters

New `eval/adapters.py` converts external benchmarks into detector items. It
deliberately **does not download anything** — which benchmarks to use is a
licensing and prioritization decision, and a converter that silently fetches
several gigabytes on first call is not something to discover at runtime. Every
converter returns `(items, report)`, and the report says what was dropped and
why: a number computed over an unknown subset of a benchmark is not comparable
to anyone else's number on that benchmark.

* **HaluEval** is self-contained — each row carries the knowledge plus both a
  correct and a hallucinated answer — so each row yields **two** items against
  the *same* premise. That pairing controls for premise difficulty, so a score
  difference is attributable to the claim rather than to the evidence, and the
  resulting set is balanced by construction. `qa`, `dialogue` and
  `summarization` field layouts are supported.
* **FEVER** ships claims plus *pointers* to evidence (wiki page + sentence
  index); the text lives in a separate dump. The converter takes a
  `resolve_evidence(page, sentence_id)` callable, which keeps the large,
  licence-encumbered corpus out of this repo and makes the converter testable.
* **Generic** field-mapping for anything else, with labels still passing
  through `detector.normalize_label` so an unrecognized spelling is reported
  rather than coerced into a class.

#### The FEVER NEI trap — why this is a module and not a ten-line loop

FEVER's NOT ENOUGH INFO claims have **no gold evidence by construction**: the
annotation says the corpus neither supports nor refutes them, so the evidence
field is empty. Convert those naively and every one arrives with an empty
premise — where the empty-premise guard flags it automatically and it scores
as a correct catch. The result is **100% recall on the NEI class with the
detector never having run**, and it looks exactly like a result.

Such rows are therefore skipped by default and counted in the report
(`skipped_nei_without_evidence`). A test demonstrates the failure concretely:
keeping them yields `recall == 1.0` with every item at
`evidence_status == "no_premise"`. Scoring FEVER's NEI class honestly requires
supplying *candidate* evidence — what a retriever actually returned — which is
a retrieval-plane concern the converter cannot invent. `keep_unresolvable_nei=True`
exists for anyone who wants them anyway; the items are marked
`premise_source="none"`.

A SUPPORTS/REFUTES row whose evidence cannot be resolved is also dropped: the
label asserts a relationship to text we do not have.

#### Operator tool

`backend/scripts/convert_benchmark.py` (manual, never pytest-collected) is the
file plumbing — accepts JSONL or a JSON array, indexes a FEVER wiki dump, and
writes a validated detector dataset. It warns when only one label class
survives, since precision/recall then degenerate and AUROC is undefined.
Verified end to end on synthetic HaluEval-shaped rows: convert → 6 balanced
items → real model → clean metrics.

#### FEVER's wiki dump, indexed to disk

`scripts/convert_benchmark.py` originally loaded the wiki dump into a Python
dict to resolve evidence pointers. Measured on a synthetic dump and
extrapolated to FEVER's ~5.4M sentences, that costs **~1.2 GB of RAM** — not
the several GB first assumed, but enough to matter on a development machine,
and paid again on every single run because nothing was cached.

New `eval/fever_wiki.py` streams the dump into a SQLite index instead
(`WITHOUT ROWID`, keyed on `(page, sentence_id)` — the only query this ever
runs is an exact-match lookup on that pair). Measured against the dict on the
same 200k-sentence dump: **45.4 MB → 3.8 MB peak Python heap, a 12x
reduction**, extrapolating to ~104 MB of RAM plus ~0.5 GB on disk for the full
dump. The index is built once and reused, so a second conversion run skips
parsing entirely. `WikiSentenceIndex.as_resolver()` satisfies
`from_fever`'s `resolve_evidence(page, sentence_id)` contract directly.

Duplicate pages (which real dumps contain) no longer abort the build, and
malformed rows are counted rather than fatal.

**Tests:** `tests/test_adapters.py` (19), `tests/test_fever_wiki.py` (12),
plus 5 label-order cases in `tests/test_nli_policy.py`. 376 tests pass
(was 340).

**What is still missing is only the data.** Point the converter at a real
downloaded benchmark and the detector plane produces a real, externally
comparable number with no API key and no retrieval. Nothing further needs to
be built for that to happen.

### The first real external measurement — and why its number is not usable

**HaluEval QA was downloaded, converted and scored against the real NLI model.
The result is a benchmark-validity finding, not a detector score.**

10,000 HaluEval QA rows converted cleanly into 20,000 detector items (perfectly
balanced, nothing skipped). A seeded 500-row / 1,000-item sample scored against
`cross-encoder/nli-deberta-v3-base`:

| metric | value |
|---|---|
| precision | 0.483 [0.449, 0.518] |
| recall | 0.758 [0.719, 0.793] |
| F1 | 0.590 |
| specificity | **0.190** |
| balanced accuracy | **0.474** |
| MCC | **−0.063** |
| AUROC | 0.526 |

Confusion: TP 379, FN 121, TN 95, **FP 405**. The system flagged **405 of 500
correct answers** as hallucinated. Balanced accuracy below 0.5 and a negative
MCC mean the output is very slightly *anti*-correlated with the truth — on this
data the detector is worse than a coin flip.

**That number describes the dataset, not the detector.** HaluEval QA's
`right_answer` is an *answer* — a bare entity, "Sidney Lumet", "Indian
Airlines" — while its `hallucinated_answer` is a *sentence*, "First for Women
was started first." Measured over the full 20,000 items:

| label | median claim length | under 4 words |
|---|---|---|
| SUPPORTED | 2 words | 88.6% |
| REFUTED | 9 words | 3.8% |

The classes are separable by length alone, without reading the evidence at
all. And for an NLI detector specifically the mismatch is fatal: a bare noun
phrase is not a proposition, so nothing can entail it. The model returns
NEUTRAL, and the strip rule flags it. The entailment rate on *correct* answers
rises monotonically with claim length — 17% at 1–3 words, 28% at 4–8, **80% at
9+**. Given an actual claim, the detector behaves.

So the honest reading is: **this run validates the measurement pipeline end to
end and invalidates this particular conversion.** It is also precisely the
failure that would have been published as "F1 0.59 on HaluEval" by anyone who
reported the headline and skipped the confusion matrix.

**A guard now exists so this cannot pass silently again.**
`adapters.dataset_validity_report()` compares claim-length distributions across
label classes and warns when the label is largely predictable from length
(≥3× median divergence) or when a class is mostly sub-four-word fragments. It
runs automatically at the end of every `scripts/convert_benchmark.py`
conversion, and on this dataset it fires both warnings. A shortcut this strong
means any score is a description of the benchmark's shape, so it is reported
at conversion time rather than left to surface as a strange confusion matrix
later.

**What would make HaluEval QA usable:** turn the question and answer into a
declarative claim ("Which magazine was started first...?" + "Arthur's
Magazine" → "Arthur's Magazine was started first"), so both classes are
propositions of comparable length. That needs question-to-statement
conversion, which is an LLM or template job and a change in what is being
measured — worth doing deliberately, not silently. HaluEval's `dialogue` and
`summarization` splits do not have this problem (their responses are already
sentences), and RAGTruth annotates spans inside generated text, so both are
better next targets than a rewrite of the QA split.

**Cost, measured:** 1,000 items took 552s (1.81 items/sec) on 6 CPU threads,
so the full 20,000 is roughly 3 hours on this hardware. The detector plane
makes no LLM calls, so the only cost is CPU time.

**Tests:** 5 validity cases in `tests/test_adapters.py`. 381 tests pass
(was 376).

### Phase 4 (completed) — premise unification, async model calls, plane persistence

Three items that were known, recorded and deferred are now done.

#### One premise preparation, not three

`verify_claims_batch`, `build_entailment_matrix` and `stability.py` each
prepared NLI premises independently, and disagreed for the same
(claim, passage) pair in four ways:

* the matrix applied the `NLI_PREPROCESS_AT` word-count threshold to the
  **raw** passage while verification applied it to the **normalized** one, so
  a passage could be sentence-focused on one path and not the other;
* below that threshold the matrix fed the **raw** passage and verification the
  **normalized** one;
* the matrix never received the caller's profile, so it always used the config
  default even when the caller asked for `none` — which is exactly what the
  detector plane passes for external benchmarks;
* the matrix had no empty-premise guard, so it still sent `("", claim)` to the
  model, reintroducing the fabricated-softmax bug on the XAI path after it had
  been fixed on the verdict path.

The visible consequence: **the entailment matrix rendered in the UI could
contradict the verdict displayed beside it**, because the two numbers were
computed from different premise text. `stability.py` was a third variant again
— it skipped normalization entirely and thresholded on raw text.

All three now call `xai_matrices.prepare_premise(claim, raw_passage, profile)`,
which returns `(focused_text, deleted_lines, evidence_status)`. The regression
test asserts the two paths send byte-identical text to the model, using a
premise containing a `3 | P a g e` marker that the generic normalizer deletes
outright — so the old paths differed in content, not merely whitespace, and
the test genuinely fails against the old behavior.

#### Model calls off the event loop

Every NLI and embedding call is synchronous, and they are invoked from
`async def` functions. A CrossEncoder forward pass therefore blocked the event
loop: under `EVAL_CONCURRENCY` each item's NLI batch stalled every other
in-flight item's LLM HTTP calls, so raising concurrency bought much less than
it appeared to. There was no `to_thread` or `run_in_executor` anywhere in
`backend/`.

New `averify_claims_batch`, `abuild_entailment_matrix` and `anli_predict` wrap
the synchronous functions in `asyncio.to_thread`; `nli_engine` and
`stability.py` use them. The work is guarded by a module-level
`threading.RLock`, and that is not optional: `_encoder` and `_nli` are single
shared torch modules, and running `predict()` on one from several threads at
once multiplies peak memory by the thread count. `RLock` rather than `Lock`
because the guarded functions call each other (`verify_claims_batch` →
`extract_relevant_sentences` both touch the models) and a plain lock would
deadlock on re-entry. Deliberately a threading primitive rather than an
`asyncio.Semaphore`, which binds to the event loop that first awaits it and
breaks the moment a second loop appears — which every `asyncio.run` in the
test suite creates.

#### Deterministic test fakes

`conftest.py`'s ML fakes returned `np.random.rand(...)`, so every assertion
about a verdict, a similarity ranking or a score distribution depended on the
global RNG. A test could pass or fail run to run, and a genuine regression was
indistinguishable from an unlucky draw — which also made it impossible to
assert anything about aggregation behavior, a prerequisite for the
multi-premise work.

The fakes are now seeded from a SHA-256 of their input, so identical input
always produces identical output, and cosine similarity between two texts is a
stable number. SHA-256 rather than Python's `hash()`, which is salted per
process: a `hash()`-seeded fake would be stable within a run and different
across runs, the worst of both worlds. Verified identical across separate
processes.

#### Plane-aware persistence

The three planes report incompatible metric shapes — end-to-end has
faithfulness and trust statuses, detector has AUROC over claim labels,
retrieval has nDCG over chunk labels. `eval_runs` gained a `plane` column
(existing rows backfilled to `end_to_end`), and the detector and retrieval
endpoints now persist their runs through the same create-before/finalize-after
durability pattern `POST /run` uses, so a crash leaves a diagnosable `failed`
row rather than silence.

Two consequences worth stating. `run_eval(..., resume_from_run_id=...)` now
**refuses** to resume a non-end-to-end run — continuing a detector run under
the end-to-end harness would merge two different measurements into one
aggregate that describes neither. And the Streamlit trend charts filter to
end-to-end runs only: plotting a detector run into a faithfulness series would
draw a gap and imply a regression where there is only a different kind of run.
Non-end-to-end runs are labelled in the run history and render their own
metrics rather than end-to-end fields as zeros.

**Tests:** `tests/test_eval_planes.py` (8) plus 5 premise-consistency cases in
`tests/test_nli_policy.py`. 394 tests pass (was 381).

### The first valid external measurement — RAGTruth

**This is the number the whole benchmark-readiness effort existed to produce.**
It is not a good number, and what it says is specific and actionable.

Setup: RAGTruth `test` split, QA + Summary tasks, spans propagated to sentence
labels by overlap (`propagation_rule="sentence_overlaps_any_span"`), premise
normalizer `none`, prediction taken from the shipped
`mitigation._should_strip` rule. 1,502 sentences sampled by whole response so
prevalence is preserved. The dataset passes `dataset_validity_report` — median
claim length 16/17/20 words across SUPPORTED/NEI/REFUTED, no length shortcut.

| metric | value |
|---|---|
| n | 1,502 (97 positives, 6.5% prevalence) |
| precision | 0.102 [0.083, 0.125] |
| recall | 0.835 [0.749, 0.896] |
| F1 | 0.182 |
| specificity | 0.495 |
| balanced accuracy | 0.665 |
| MCC | 0.162 [0.119, 0.202] |
| **AUROC** | **0.727** |
| AUPRC | 0.130 |

Confusion: TP 81, FN 16, TN 695, **FP 710**.

#### Three findings, in order of usefulness

**1. The ranking works; the operating point does not.** *(SUPERSEDED - a threshold sweep later showed precision is flat across the whole range and the cause is a predicate mismatch, not calibration. See the correction entry below.)* AUROC 0.727 says the
entailment score does separate hallucinated sentences from grounded ones —
compare HaluEval QA's 0.526, which was chance. But at the shipped threshold
the system flags **710 of 1,405 grounded sentences**, almost exactly half, for
81 true catches. Precision 0.10. A reviewer would stop reading at that
precision, and an operator would turn the feature off within a day.

This is the distinction `verdict_confusion` was added to expose: "the NLI model
is wrong" versus "the strip threshold is miscalibrated". Here it is decisively
the threshold. `STRIP_ENTAILMENT_FLOOR = 0.5` was chosen by hand and has never
been fitted to data; 600 of those 1,405 grounded sentences came back NEUTRAL,
which on a long article premise is an ordinary outcome for a true sentence
whose supporting detail did not make the top-3 selection. This is exactly what
the conformal calibration machinery (Phase 0/W0.3) exists to set, and it has
never been run against a labeled external set — now it can be.

**2. Recall is inverted across error types, and that is the real bug.**

| RAGTruth error type | recall | n |
|---|---|---|
| Evident Baseless Info | 0.938 | 64 |
| Subtle Baseless Info | 0.857 | 7 |
| **Evident Conflict** | **0.542** | 24 |
| Subtle Conflict | 1.0 | 2 |

A claim the evidence **flatly contradicts** is caught barely half the time,
while a claim the evidence merely fails to mention is caught 94% of the time.
That is backwards: a contradiction is the easier and more dangerous case.

**3. The mechanism is visible in the verdict confusion.** Of 26 REFUTED
sentences the model returned CONTRADICTION for only **4**, NEUTRAL for 11, and
**ENTAILMENT for 11**. A contradicted claim being scored as entailed is not a
threshold problem — the model was shown a premise that genuinely supports the
claim's surface form. That points at premise *selection*:
`extract_relevant_sentences` picks the top-`NLI_TOP_K` sentences most
**similar** to the claim, and for a contradiction the sentence that refutes it
is often lexically *less* similar than nearby supporting context. The
selector systematically hands the model the wrong three sentences.

#### What this settles

Multi-premise NLI has been deferred twice in this effort for want of evidence
to choose an aggregation rule with. There is now evidence, and it redirects the
work: the problem is not primarily aggregating over K retrieved chunks, it is
that **similarity-ranked sentence selection loses contradictions**. A fix
should be evaluated on `Evident Conflict` recall specifically — currently
0.542 — with precision held constant. Candidate directions, now testable
against a real number rather than argued from first principles: select
premises by NLI score rather than embedding similarity, keep a contradiction
-oriented selection alongside the similarity one, or score against the whole
premise when it fits the model's context.

#### Honest limits

Sampled (1,502 of 12,209 converted sentences), single split, two task types,
one NLI checkpoint. `Subtle Conflict` n=2 and `Subtle Baseless Info` n=7 are
too small to read. The sentence-level propagation rule is this repo's choice
and a different rule gives a different number, which is why it travels with
every item. Nothing here is comparable to a published RAGTruth score computed
at response level.

#### Cost

0.6-0.8 items/sec on 6 CPU threads, so ~35 minutes for 1,502 items. Run in
chunks that persist per-item results, because two attempts to score the sample
in one process were killed for memory on a machine with ~2 GB free — the
premise cache and batching reduced the work substantially but the models
themselves need ~3.5 GB resident.

### Correcting the RAGTruth interpretation — precision on this conversion is not a valid detector measure

The entry above reads the low precision (0.102) as a miscalibrated operating
point and calls the threshold "decisively" the cause. **That was wrong**, and
four experiments were needed to establish it. Recording them because the
negative results are worth more than the original claim.

**1. Threshold calibration — no effect.** New `eval/threshold.py` sweeps the
entailment floor while replicating the shipped `_should_strip` rule exactly
(contradiction and missing-evidence branches fire as in production; only the
floor varies — sweeping `score < t` alone would describe a detector this repo
does not ship). Precision is flat at ~0.10 across the whole range:

| floor | precision | recall | FP |
|---|---|---|---|
| 0.10 | 0.108 | 0.835 | 666 |
| **0.50 (shipped)** | **0.102** | **0.835** | **710** |
| 0.90 | 0.102 | 0.876 | 747 |

Best achievable F1 is 0.198 against a current 0.182. There is no operating
point worth moving to, so the threshold is not the problem.

The score distributions say why. Grounded sentences the system keeps have a
median entailment of **0.9951**; grounded sentences it flags have **0.0013**;
genuine hallucinations have **0.0011**. The falsely-flagged grounded sentences
and the real hallucinations are statistically indistinguishable — the score
carries no information separating them, so no cut-off can.

**2. Premise selection — not the cause either.** Hypothesis: top-3 similarity
selection discards the supporting sentence. Two tests, both negative.
Handing the model the whole premise made things *worse* (specificity 0.220 vs
0.534), because NLI models are trained on sentence pairs and degrade on
300-word premises. Raising `NLI_TOP_K` from 3 to 6 moved specificity from
0.534 to 0.559 — three fewer false positives in 118.

**3. Fragment artifacts — real bug, not the cause.** The sentence splitter was
emitting bare list markers ("1.", "2.") as claims: 126 of 1,502 scored items,
and 18% of all false positives. Nothing can entail "2.", so each was a
guaranteed false positive that said nothing about the detector. Fixed
(`_drop_list_markers`, `_strip_leading_markers` advancing `start` so
`text[start:end] == sentence` still holds, and an `is_proposition` filter with
`min_claim_words`, all counted in the conversion report). But excluding them
barely moves precision — 0.102 to 0.098 — because it removes true positives in
the same proportion.

**4. The actual cause: entailment and hallucination are different predicates.**
Tracing one false positive settles it. Claim: *"To make panko crumbs, preheat
the oven to 300 degrees F (150 degrees C)."* The selector worked perfectly and
handed the model *"passage 1:1 Preheat an oven to 300 degrees F (150 degrees
C)... passage 3:Preheat the oven to 300 degrees F."* Entailment: **0.002**.

The word "panko" never appears in the premise. Strict entailment is therefore
*correct* to refuse: the evidence supports "preheat to 300F" but not the panko
framing, which the reader supplies from the question. RAGTruth does not
annotate this as a hallucination, because a human judges it faithful in
context.

RAGTruth marks a span hallucinated when a human reads it as fabricated. Strict
NLI entailment requires the premise to entail every element of the claim,
including topical framing a reader bridges for free. The second predicate is
strictly stronger, so measuring this system's precision against RAGTruth's
sentence labels penalizes it for being stricter than the annotation — not for
being wrong.

#### What survives, and what does not

* **Not usable:** precision, F1, specificity on this conversion. They are
  dominated by the predicate mismatch, and any of them quoted as "our
  hallucination detector's precision" would be misleading.
* **Usable:** recall (0.835) and AUROC (0.727-0.733) — both computed over
  RAGTruth's annotated positives, where the two predicates agree that
  something is wrong.
* **Still the headline finding:** Evident Conflict recall **0.45-0.54** versus
  0.94 for Baseless Info. Direct contradictions are caught half as often as
  merely unsupported claims, and *that* is measured on the annotated positives
  where the predicates do agree. It remains the one clearly actionable defect.

#### The product consequence is real even though the measurement is not

The benchmark cannot fairly score precision here, but the *behavior* it
surfaced is genuine: on RAG-style content this system flags roughly half of
all sentences, including correct ones, because strict entailment rejects
ordinary topical bridging. A user would experience that as noise regardless of
what any benchmark says. Softening it is a design decision — accept
claim-level entailment against a wider premise, or gate on contradiction
strength rather than entailment absence — not a calibration one.

#### Method note

Four hypotheses, three falsified, one confirmed by a single traced example.
The three negative results are recorded because each was individually
plausible and would otherwise be re-proposed: the threshold looks miscalibrated
from the confusion matrix, the selector looks lossy from the top-K design, and
the fragments look like the obvious artifact once seen.

### Pre-dating this effort, but foundational to it

**Gemini → Groq/OpenRouter migration.** `shared/llm.py`: config-driven
provider selection (`LLM_PROVIDER`), one `call_llm()` choke point every
caller already went through. This is *why* the multi-key-pool requirement
(Phase 2) exists — the reviewer explicitly noted `shared/llm.py` has no seam
today for multiple keys per provider (a process-global singleton, no
per-key rate-limit state, key-blind retry).

---

## Not started

- **Phase 3 (remainder, W3.2)**: the labels themselves — hundreds of items with
  graded `relevant_chunk_keys`, and an actually-frozen benchmark corpus.
  Corpus identity (W3.1) is done; see above for the open decisions blocking
  the labeling itself.
- **Phase 4 (remainder)**: premise SELECTION, redirected by the RAGTruth
  result above. The evidence says the weak point is not aggregating over K
  retrieved chunks but that similarity-ranked sentence selection loses
  contradictions (Evident Conflict recall 0.542, and 11 of 26 refuted
  sentences scored ENTAILMENT). Evaluate any fix on Evident Conflict recall
  with precision held constant.
- **Threshold calibration**: done and CLOSED as a non-issue. `eval/threshold.py`
  sweeps the floor against labeled data; precision is flat at ~0.10 across the
  entire range, so there is no operating point worth moving to. The tool stays
  for the domain corpus, where the answer may differ.
- **A claim-level entailment benchmark**: precision cannot be measured against
  RAGTruth's sentence labels, because strict entailment is a stronger predicate
  than its hallucination annotation (see the correction entry). Measuring
  precision honestly needs either response-level evaluation as RAGTruth intends,
  or a set annotated for entailment specifically.
- **Phase 5 (mostly done)**: HaluEval, FEVER and generic adapters exist
  (`eval/adapters.py` + `scripts/convert_benchmark.py`) — see above. What
  remains is **running them on real downloaded data**, which is a licensing
  and prioritization decision, not engineering. RAGTruth (span-level
  annotations) and BEIR (retrieval; needs a corpus and the retrieval plane)
  have no converter yet.
- **Phase 6**: pipeline profile registry (`plain_rag` baseline, ablation
  switches) + paired comparison harness.
- **Phase 7 (remainder)**: `eval/cli.py`, async job API, the nightly live-run
  CI tier with the key pool.
- **Phase 8**: variance/CI reporting on trend charts, seeded test mocks,
  golden-vector regression tests, the benchmark card doc.

## Open decisions (still the project owner's call)

See the roadmap artifact for full detail — API key inventory/plan tiers,
cross-provider failover policy, external-benchmark prioritization and
licensing, the embedding-model unification question (finding #18: Chroma
retrieves with MiniLM, `xai_matrices.py` scores with bge-large — deliberately
kept independently configurable rather than silently unified), dataset
labeling budget, target α for conformal, domain-profile default (already
resolved: `generic`), and the benchmark corpus source (public/scriptable vs.
private).
