# Governance Suite

Five features that make White-Box RAG defensible to an auditor or regulator, not just observable. They build on the existing trust/verification layer and stay **pure-math** — Gemini generates and extracts; math judges. Nothing here adds an LLM to the decision loop.

The through-line: a decision is **explainable** (counterfactuals), **statistically defensible** (conformal abstention), **routed to a human when uncertain** (review queue), **cryptographically un-tamperable** (hash-chained audit log), and **mapped to law** (EU AI Act / NIST AI RMF).

| Feature | Pillar | Endpoint(s) | UI |
|---|---|---|---|
| Tamper-evident audit log | Governance | `GET /api/audit/verify-integrity` | Audit Trail |
| Human-in-the-loop review queue | Responsible / Governance | `GET /api/review/queue`, `POST /api/review/{id}/resolve`, `GET /api/review/{id}/history` | Review Queue, Audit Trail |
| Counterfactual explanations | Explainability | *(on `/api/query` response)* | RAG Query, Audit Trail |
| Conformal abstention | Responsible | `POST /api/eval/calibrate`, `GET /api/eval/calibration` | Evaluation |
| EU AI Act / NIST AI RMF mapping | Governance | `GET /api/compliance/frameworks` | Regulatory Mapping |

---

## 1. Tamper-evident, hash-chained audit log

**Problem.** The audit trail was a plain SQLite table — anyone with DB access could rewrite a verdict after the fact, with no way to detect it.

**Solution.** Every audit record is SHA-256 chained to its predecessor:

```
record_hash = sha256( prev_hash + canonical_json(record) )
```

`canonical_json` sorts keys and strips the chain-metadata fields (`id`, `prev_hash`, `record_hash`, `chain_index`) so a record never hashes over its own hash and the payload reproduces at verify time. Any retroactive edit, deletion, or reordering breaks the chain and is reported by `GET /api/audit/verify-integrity`:

```json
{ "intact": false, "count": 12, "first_break": { "id": 7, "chain_index": 6, "reason": "content_tampered" } }
```

`reason` is `content_tampered` (a record's payload was altered) or `broken_linkage` (a record was deleted or reordered).

**Concurrency.** The eval harness fans out `/query` under `asyncio.Semaphore(3)`. A naïve "read last hash → compute → write" would let two concurrent finalizes read the same predecessor and fork the chain. Finalization is guarded by a module `asyncio.Lock`, and an explicit `chain_index` (assigned under the lock) — not the row `id` — defines chain order, so out-of-order finalization is safe.

**Bonus fix.** The live audit write (`compliance/audit.py`) previously persisted only a *partial* report; the scorecard/XAI/faithfulness fields were assembled later in the pipeline and returned in-memory only, never saved. Finalization now overwrites `audit_data_json` with the **complete** record — so the Audit Trail detail view finally reads back the full scorecard it renders.

**Code.** `backend/shared/audit_chain.py` (pure-stdlib primitives), `backend/shared/database.py` (`finalize_audit_record`, `iter_audit_chain`, `_ensure_column` migration helper), `backend/ingestion/pipeline.py` (finalize step). **Tests:** `backend/tests/test_audit_chain.py`.

---

## 2. Human-in-the-loop (HITL) review queue

**Problem.** The Trust Gate emitted `NEEDS_HUMAN_REVIEW` and then… nothing happened. The same answer was returned; the flag led nowhere.

**Solution.** Flagged audits appear in a review queue and can be resolved by a named reviewer (`approve` / `override` / `reject`) with a note. Resolutions are **append-only** and form their **own hash chain** (`review_actions`) — the audit row is never mutated (that would break its chain), and the current review state is *derived* from the latest action, giving full resolution history rather than just the latest flag.

- `GET  /api/review/queue` — audits flagged `Needs_Human_Review` with no resolution yet.
- `POST /api/review/{audit_id}/resolve` — body `{ reviewer, action, note }`, appends a chained action.
- `GET  /api/review/{audit_id}/history` — full chained history + derived status.

> **Limitation.** There is no authentication in this system, so reviewer identity is supplied in the request body. In a production deployment this would come from an authenticated session.

**Code.** `backend/governance/routes.py`, `backend/shared/database.py` (`insert_review_action`, `list_pending_reviews`, `latest_review_status`, `iter_review_chain`). **Tests:** `backend/tests/test_review_chain.py`.

---

## 3. Counterfactual / contrastive explanations

**Problem.** A non-SAFE verdict told you *that* an answer was risky, not *what* to do about it.

**Solution.** For each penalising claim, show the exact trust outcome **if that claim were removed** — e.g. *"removing this claim raises the score 0.55 → 0.85 and flips Non-Compliant → Needs Review."* The single highest-leverage claim is flagged as the primary driver.

Two correctness points:

- **Status is not a function of the score.** The Trust Gate status is decided by *which penalty types remain* (contradiction, low-confidence, conflict…), not a threshold on the scalar score. So each counterfactual is computed by **leave-one-out recomputation** of the real gate, not by mapping a projected number to a band.
- `compute_shapley_contributions` returns arrays **sorted by φ**, so their indices don't align with the verifications list. Each claim's φ/reasons come from a **single-claim** Shapley call (reusing the canonical penalty tables), never by indexing the sorted arrays.

Edition conflicts pass through unchanged to each recomputation — removing a claim never resolves a conflict, and the counterfactual honestly reflects that (`flips_status: false`).

**Code.** `backend/verification/counterfactual.py`; `Counterfactual` model; wired into `pipeline.py` and part of the hash-chained record. **Tests:** `backend/tests/test_counterfactual.py`.

---

## 4. Conformal-prediction abstention

**Problem.** Abstention used a hand-tuned constant (`ABSTENTION_MEAN_PENALTY_CEIL = 0.25`) — a guess, with no statistical meaning.

**Solution.** Calibrate the threshold to a **risk target** via split conformal prediction. The nonconformity score is the retained-set mean penalty (what `should_abstain` already compares). Calibrating on items labelled *answerable*, the `⌈(n+1)(1-α)⌉`-th smallest score becomes the threshold, so **at most ~α of truly-answerable queries are wrongly abstained** (marginal coverage on the answerable set).

- `POST /api/eval/calibrate` — body `{ alpha, dataset_path? }`; runs the labelled set through the live pipeline and persists the threshold.
- `GET  /api/eval/calibration` — the active threshold, α, n, and coverage note.

Wiring is **non-breaking**: `should_abstain` prefers the calibrated threshold and falls back to the fixed ceiling when none is set, so behaviour is identical until you calibrate.

> **Honest scoping.** This is split conformal for a binary answer/abstain decision — **not** a per-token guarantee. The shipped `calibration_dataset.jsonl` is a demo starter; a production threshold needs a real labelled set over *your* corpus, and calibration runs the live pipeline (needs a Gemini key + ingested docs), so it is an operator action, not a CI step.

**Code.** `backend/verification/conformal.py` (pure math + JSON persistence), `backend/eval/harness.py` (`run_calibration`), `backend/verification/mitigation.py` (threshold lookup). **Tests:** `backend/tests/test_conformal.py`.

---

## 5. EU AI Act / NIST AI RMF control mapping

**Problem.** "Governance" was implicit. Nothing stated *which* regulatory obligations the system actually addresses.

**Solution.** A **static** control catalog mapping each capability to a real control, with an honest status (`satisfied` / `partial` / `planned`) and evidence. `GET /api/compliance/frameworks` returns it with computed per-framework coverage.

| Framework | Example mappings |
|---|---|
| **EU AI Act** (high-risk, Title III Ch. 2) | Art. 12 record-keeping → hash chain · Art. 13 transparency → scorecard/XAI/counterfactuals · Art. 14 human oversight → HITL · Art. 15 accuracy/robustness → conformal abstention |
| **NIST AI RMF 1.0** | Govern / Map / Measure / Manage → audit chain · retrieval provenance · scorecard+eval · HITL+abstention |

`partial` is used deliberately wherever the system addresses part of an obligation but not the full regime (e.g. Art. 9 risk management, Art. 10 data governance) — no over-claiming.

**Code.** `backend/compliance/frameworks.py` (static catalog + coverage), route in `compliance/routes.py`, `streamlit_app/views/regulatory_mapping.py`. **Tests:** `backend/tests/test_frameworks.py`.

---

## Testing

All governance logic is covered by headless unit tests (mocked models, no Gemini):

```bash
cd backend
pytest tests/test_audit_chain.py tests/test_review_chain.py \
       tests/test_counterfactual.py tests/test_conformal.py tests/test_frameworks.py
```

The full backend suite (64 tests) runs with a bare `pytest` from `backend/`. The live-pipeline paths (real `/query` chaining, calibration, and the Streamlit pages) require a running backend + `GEMINI_API_KEY` and are verified manually.
