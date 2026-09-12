# XAI Governance Framework — Mathematical Specification

## Component Matrix

| Component | Mathematical Method | Role in XAI |
|---|---|---|
| **Retriever** | Similarity Matrix **S = QKᵀ / √d** | Explains why documents were chosen |
| **Attribution** | Attribution Matrix **A = sentence_emb · chunk_embᵀ** (cosine) | Explains what content each claim used |
| **Encoder** | SVD / PCA on embeddings | Visualises knowledge structure |
| **Reasoner** | Graph/Adjacency Matrix of claims | Maps relational knowledge |
| **Evaluator** | Shapley/Confidence Matrix | Assigns trust scores to output |

---

## 1. Retriever — Similarity Matrix

### How it works in our system
When you POST a query to `/api/query`, the `hybrid_retrieve()` function executes two simultaneous retrieval strategies that are **mathematically fused**.

### Dense Retrieval (Similarity Matrix)

Let **q ∈ ℝᵈ** be the embedding of the user query produced by `all-MiniLM-L6-v2` (d = 384 dimensions).

Let **K ∈ ℝⁿˣᵈ** be the matrix of all stored chunk embeddings in ChromaDB, where n = total number of stored chunks.

The **Similarity Score** between the query and every document chunk is:

```
S = q · Kᵀ / (‖q‖ · ‖kᵢ‖)    ∈ ℝⁿ
```

This is the **cosine similarity** between the query vector and every stored chunk vector. Each entry Sᵢ ∈ [-1, 1] represents how topically similar chunk i is to the query.

ChromaDB returns the top-20 chunks by highest Sᵢ score.

### Sparse Retrieval (BM25)

BM25 computes a term frequency score. For query term t in chunk d:

```
BM25(t, d) = IDF(t) · [f(t,d) · (k₁ + 1)] / [f(t,d) + k₁ · (1 - b + b · |d|/avgdl)]
```

Where:
- `f(t,d)` = frequency of term t in document d
- `|d|` = document length, `avgdl` = average document length
- `k₁ = 1.5`, `b = 0.75` (standard constants)
- `IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5))` where N = corpus size, n(t) = docs containing t

The total BM25 score for a chunk is the sum over all query terms: **Score(d) = Σ BM25(tᵢ, d)**

### Reciprocal Rank Fusion (RRF)

Both retrieval lists are fused using RRF to produce a single ranked list:

```
RRF_score(d) = Σ [ 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d)) ]
```

Where k = 60 (standard constant). This ensures chunks that rank highly in **both** strategies are strongly preferred. The top-10 chunks by RRF score are returned for generation.

---

## 2. Claim-Level Attribution — Attribution Matrix (cosine similarity)

> **Corrected 2026-08.** This section previously described a forced inline-citation
> mechanism (`[PUB·EDITION·SECTION·CHUNK]` brackets parsed out of the LLM's
> output) framed as "materializing the transformer's attention matrix." That
> was never an accurate description of an LLM's actual internal attention
> weights (a citation string can't reveal those), and it no longer describes
> the code either: `rag.py`'s system prompt now explicitly instructs the
> model **not** to include inline citations, and attribution is computed by
> an entirely separate, deterministic step — not by parsing anything out of
> the generated text at all.

### How it works in our system

The LLM (Groq or OpenRouter, config-driven — see `shared/llm.py`) generates a
plain-prose answer with **no** inline citation syntax. `rag.py::parse_claims()`
then splits that answer into sentences (spaCy) and hands them, together with
the retrieved chunks, to `shared/xai_matrices.py::build_attribution_matrix()` —
a pure-math step with zero LLM involvement.

### Mathematical Model

Let **E ∈ ℝᵃˣᵈ** be the sentence-transformer embeddings (`BAAI/bge-large-en-v1.5`,
d = 1024) of the m generated sentences, and **C ∈ ℝⁿˣᵈ** the embeddings of the
n retrieved chunks. The **Attribution Matrix** is:

```
A = Ê · Ĉᵀ     ∈ ℝ^(m × n),   Ê, Ĉ = row-L2-normalized E, C
```

so `A[i][j] = cos(sentence_i, chunk_j)`. For each sentence i,
`attribute_sentence()` takes the **argmax** over chunks:

```
primary_chunk(i) = argmax_j A[i][j]
attribution_score(i) = max_j A[i][j]
```

If `attribution_score(i) < MIN_ATTRIBUTION_SCORE` (0.45, configurable via
`shared/config.py`), the sentence is left **unattributed** rather than forced
onto a poor match. `compute_primary_attributions()` additionally records the
runner-up chunk and the confidence gap between the top two matches; a small
gap (< `AMBIGUITY_GAP_THRESHOLD`) flags the attribution as **ambiguous**,
which feeds a Trust Gate/Shapley penalty (§4) — a claim isn't just "attributed
to the wrong place," it's specifically flagged as *ambiguously* attributed
when two chunks are near-tied.

### What this replaces

There is no citation-parsing code in the current pipeline — no regex, no
forced bracket syntax, no dependency on the LLM cooperating with a citation
instruction. `Claim.source_section_id`/`source_passage` are populated
entirely from the Attribution Matrix argmax, which is why claim attribution
is reconstructible and auditable independent of what the LLM actually did
internally.

---

## 3. Hallucination Detection — NLI Entailment Score

> **Corrected 2026-08.** This section previously said the generative LLM
> (then Gemini) acts "as an NLI judge." It never did, and this is the single
> most important correction in this document: verification runs on a
> **local, non-generative CrossEncoder model** (`cross-encoder/nli-deberta-v3-base`,
> `shared/xai_matrices.py`), never the LLM that wrote the answer. This is the
> project's core "LLM extracts, math judges" design principle — no
> LLM-as-judge anywhere in the verification loop — and the earlier wording
> directly contradicted it.

### How it works in our system

For **every claim**, `verification/nli_engine.py::verify_all_claims()` batches
`(claim_text, source_passage)` pairs through the local CrossEncoder
(`shared/xai_matrices.py::verify_claims_batch`), producing a structured
verdict with zero LLM calls.

### Mathematical Model

Formally, NLI models a **conditional probability distribution**:

```
P(label | premise p, hypothesis h)    where label ∈ {ENTAILMENT, NEUTRAL, CONTRADICTION}
```

The CrossEncoder returns a 3-way softmax over exactly this distribution:

```
entailment_score ≡ P(ENTAILMENT | source_passage, claim_text)     ∈ [0.0, 1.0]
```

### What this tells us

| entailment_score | Meaning |
|---|---|
| **> 0.8** | Claim is strongly grounded in the source. Low hallucination risk. |
| **0.5 – 0.8** | Ambiguous. Claim may be partially unsupported. Medium risk. |
| **< 0.5** | Claim is weakly or not supported by the source. High hallucination risk. |
| **verdict = CONTRADICTION** | Claim **directly contradicts** the source passage. Definitive hallucination. |

### Faithfulness Score (RAGAS-style)

After verifying all n claims, the **Faithfulness** of the RAG response is:

```
Faithfulness = |{cᵢ : verdict(cᵢ) = ENTAILMENT}| / n     ∈ [0, 1]
```

A score of 1.0 means every claim is perfectly grounded. This is computed in `scorecard.py`.

---

## 4. Trust Gating — Shapley / Weighted Scoring

### How it works in our system

After all claims are verified, `compute_trust_gate()` in `trust_gate.py` aggregates the individual verification scores into a single **overall_score** and maps it to a trust tier.

### Mathematical Model

The **overall trust score** is initialised at maximum (1.0) and **penalties are subtracted** based on violations, analogous to a Shapley-value additive contribution model:

```
Overall_Score = 1.0
             - 0.5  × [unresolved_edition_conflicts > 0]    (binary conflict penalty)
             - Σᵢ 0.3  × [verdict(cᵢ) = CONTRADICTION]     (per contradiction)
             - Σᵢ 0.1  × [verdict(cᵢ) = NEUTRAL]           (per neutral claim)
             - Σᵢ 0.2  × [entailment_score(cᵢ) < 0.5]      (per low-confidence claim)
             - Σᵢ 0.05 × [0.5 ≤ entailment_score(cᵢ) ≤ 0.8] (per medium-confidence)

Overall_Score = max(0.0, min(1.0, Overall_Score))           (clamped to [0, 1])
```

### Gate Decision Boundary (Classification)

The final trust status is determined by a **threshold classifier** on the accumulated penalty flags:

```
Trust_Status =
  "NON_COMPLIANT"      if (any CONTRADICTION) OR (any entailment < 0.5) OR (any unresolved conflict)
  "NEEDS_HUMAN_REVIEW" if (any NEUTRAL) OR (any 0.5 ≤ entailment ≤ 0.8)
  "SAFE"               if (all ENTAILMENT) AND (all entailment > 0.8) AND (no conflicts)
```

This is a **deterministic decision tree**, ensuring full auditability — every gate decision can be fully reconstructed from the individual claim verdicts.

### Shapley-Value Connection

Each claim's contribution to the overall trust reduction is its **marginal contribution** to the score drop — directly analogous to Shapley values from cooperative game theory:

```
φᵢ = Overall_Score_without_claim_i - Overall_Score_with_claim_i
```

The claims with the largest φᵢ are the most impactful on the final gate decision, and are highlighted in the audit report as **key risk contributors**.

---

## 5. BRD Compliance Engine — Alignment Score & Gap Analysis

> **Corrected 2026-08.** This section previously described `alignment_score`
> as a computed average cosine similarity, with a derived
> `Compliance_Score = 100 × Alignment_Score × (1 - violation_penalty)`
> formula. Neither formula is computed anywhere in `mapper.py` — unlike
> retrieval, NLI verification, and the Trust Gate (all pure math), the BRD
> alignment/gap/violation/risk-level judgment is **produced directly by the
> LLM's structured JSON output**, not derived from a similarity score. This
> is a deliberate scope note, not just a correction: `compliance/mapper.py`
> is the one place in the pipeline where "math judges" doesn't hold — the
> LLM extracts *and* judges here. Wording below is also now domain-generic
> (the collection label is free text, not tied to any regulator).

### How it works in our system

`map_requirement()` in `mapper.py` uses hybrid retrieval (§1 — real cosine +
BM25 + RRF math) to find the top-5 most relevant retrieved sections for a
requirement, then prompts the LLM (Groq or OpenRouter) with those sections
and asks for a structured compliance evaluation.

### What the retrieval step computes (real math)

Let **r ∈ ℝᵈ** be the embedding of the requirement text and
**s₁, ..., s₅ ∈ ℝᵈ** the embeddings of the top-5 retrieved sections
(`hybrid_retrieve`, §1's dense+BM25+RRF math — this part is deterministic).

### What the LLM produces (not a formula)

Given the requirement and the retrieved section texts, the LLM returns
structured JSON containing:
- **alignment_score**: the model's own judgment of how well the requirement
  is covered by the retrieved sections
- **gaps**: requirements present in the BRD but not covered by any retrieved section
- **violations**: requirements that directly conflict with the retrieved sections
- **risk_level**: HIGH / MEDIUM / LOW based on the severity of gaps/violations
- **remediation_suggestions**: explicit corrective actions

None of these are recomputed or cross-checked against the retrieval
similarity scores — they are read directly from the LLM's response (with a
fixed fallback value on a parse/API failure; see `mapper.py`).

---

## 6. Audit Report — Explainability Artifacts

### How it works in our system

`generate_audit_report()` in `audit.py` compiles every mathematical intermediate into a traceable JSON audit log stored in SQLite.

### What is recorded and why

| Field | Mathematical Source | Explainability Purpose |
|---|---|---|
| `claims[].source_section_id` | Attribution Matrix argmax (cosine similarity, §2) | Which chunk drove each claim |
| `claims[].source_passage` | Chunk text at the argmax index | Exact text the attribution points to |
| `verifications[].entailment_score` | P(ENTAILMENT \| premise, hypothesis) — local CrossEncoder (§3) | Probability the claim is grounded |
| `verifications[].verdict` | NLI classification label | Discrete grounding decision |
| `trust_gate.overall_score` | Shapley penalty summation | Composite hallucination risk measure |
| `trust_gate.status` | Decision boundary classifier | Final compliance ruling |
| `trust_gate.reasoning` | Per-claim penalty trace | Full decision audit trail |
| `compliance_evidence[].alignment_score` | LLM judgment over retrieved sections (§5 — not a computed formula) | How well the BRD maps to the retrieved corpus |
| `compliance_evidence[].gaps` | LLM gap analysis | What is missing from the BRD |
| `edition_traceability` | Cross-edition similarity comparison | Which edition is authoritative |

### Full Pipeline Math Summary

```
Query q
  │
  ▼
[Retriever]    S = q·Kᵀ/‖q‖‖K‖  +  BM25 → RRF → top-10 chunks C
  │
  ▼
[Generation]   LLM(Groq/OpenRouter) generates plain-prose Answer (no citations)
  │
  ▼
[Attribution]  A = sentence_embs·chunk_embsᵀ (cosine) → argmax → claims {c₁,...,cₙ} + source chunk
  │
  ▼
[NLI Verifier] ∀cᵢ → local CrossEncoder: P(ENTAILMENT|cᵢ.passage, cᵢ.text) = entailment_score(cᵢ)
  │
  ▼
[Trust Gate]   Overall_Score = 1 - Σ penalties(cᵢ)  → Status ∈ {SAFE, REVIEW, NON_COMPLIANT}
  │
  ▼
[BRD Mapper]   retrieval(r,sⱼ) real cosine+BM25+RRF math → LLM judges alignment_score + gaps + violations
  │
  ▼
[Audit Store]  Full JSON artifact → SQLite, SHA-256 hash-chained  (complete mathematical trace)
```

Every step except the BRD Mapper's compliance judgment and the answer
generation itself is **deterministic**, **traceable**, and **reconstructible**
from the stored audit record — that's the actual scope of "no LLM in the
decision loop": generation and BRD compliance judgment are LLM steps;
retrieval, attribution, NLI verification, and trust gating are not.
