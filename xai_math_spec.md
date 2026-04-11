# XAI Governance Framework — Mathematical Specification

## Component Matrix

| Component | Mathematical Method | Role in XAI |
|---|---|---|
| **Retriever** | Similarity Matrix **S = QKᵀ / √d** | Explains why documents were chosen |
| **Generator** | Attention Matrix **A = softmax(QKᵀ / √dₖ) · V** | Explains what content was used |
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

## 2. Claim-Level Attribution — Attention Matrix

### How it works in our system

When Gemini Pro generates the RAG answer, the response is forced (via prompt engineering) to include citation keys like `[FSR·Jun2024·1.1·c0]` after every claim sentence. This directly **materialises the Attention Matrix**.

### Theoretical Basis

Inside Gemini's transformer, for each generated token, the **scaled dot-product attention** over all input (source chunk) tokens is:

```
A = softmax(Q · Kᵀ / √dₖ) · V     ∈ ℝ^(output_len × d_v)
```

Where:
- **Q** = query matrix (generated tokens being predicted)
- **K** = key matrix (all input chunk tokens)
- **V** = value matrix (semantic content of input chunks)
- **√dₖ** = scaling factor to prevent softmax saturation

Each row Aᵢ of the attention matrix tells us which input tokens (i.e., which source chunks) had the **highest attention weight** when generating output token i.

### What we capture

Our `parse_claims()` function in `rag.py` extracts this implicitly:

```python
# Regex extracts [PUB·EDITION·SECTION·CHUNK] citations from each sentence
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")
claims = []
for sentence in sentences:
    citations = CITATION_PATTERN.findall(sentence)
    source_chunk = _find_chunk_by_key(citations[0], chunks)
    claims.append(Claim(
        text=sentence,
        source_passage=source_chunk.chunk_text[:500],
        ...
    ))
```

The citation in each claim sentence is a **symbolic proxy for the maximum attention weight** — Gemini is forced to declare which chunk most strongly influenced each generated claim, recreating interpretable attribution from the otherwise opaque attention matrix.

---

## 3. Hallucination Detection — NLI Entailment Score

### How it works in our system

For **every claim** produced by BP1, BP2's `verify_claim()` calls Gemini as an NLI judge, producing a structured verdict.

### Mathematical Model

Formally, NLI models a **conditional probability distribution**:

```
P(label | premise p, hypothesis h)    where label ∈ {ENTAILMENT, NEUTRAL, CONTRADICTION}
```

Gemini returns a confidence score which we interpret as:

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

### How it works in our system

When `map_requirement()` in `mapper.py` processes a BRD requirement, it uses hybrid retrieval to find the top-5 most relevant RBI sections, then prompts Gemini to produce a structured compliance evaluation.

### Mathematical Model

Let **r ∈ ℝᵈ** be the embedding of the BRD requirement text.  
Let **s₁, s₂, ..., s₅ ∈ ℝᵈ** be the embeddings of the top-5 retrieved RBI sections.

The **semantic alignment score** for each section sⱼ is:

```
alignment(r, sⱼ) = (r · sⱼᵀ) / (‖r‖ · ‖sⱼ‖)     ∈ [-1, 1]
```

The **overall alignment score** for the requirement against all retrieved sections is:

```
Alignment_Score = (1/5) Σⱼ alignment(r, sⱼ)     ∈ [0, 1]
```

But Gemini goes further — given the actual requirement and section texts, it produces:
- **gaps**: requirements present in the BRD but not covered by any RBI section
- **violations**: requirements in the BRD that directly conflict with RBI guidance
- **risk_level**: HIGH / MEDIUM / LOW based on the severity of gaps/violations
- **remediation_suggestions**: explicit corrective actions

### Compliance Score Formula (0-100)

```
Compliance_Score = 100 × Alignment_Score × (1 - violation_penalty)

where violation_penalty = min(1.0, |violations| × 0.25)
```

A requirement with no violations and perfect alignment scores 100. Each violation reduces the score by 25 points up to a floor of 0.

---

## 6. Audit Report — Explainability Artifacts

### How it works in our system

`generate_audit_report()` in `audit.py` compiles every mathematical intermediate into a traceable JSON audit log stored in SQLite.

### What is recorded and why

| Field | Mathematical Source | Explainability Purpose |
|---|---|---|
| `claims[].source_section_id` | RRF top-1 chunk ID | Which document drove each claim |
| `claims[].source_passage` | Raw chunk text (attention arg-max) | Exact text the model "attended to" |
| `verifications[].entailment_score` | P(ENTAILMENT \| premise, hypothesis) | Probability the claim is grounded |
| `verifications[].verdict` | NLI classification label | Discrete grounding decision |
| `trust_gate.overall_score` | Shapley penalty summation | Composite hallucination risk measure |
| `trust_gate.status` | Decision boundary classifier | Final compliance ruling |
| `trust_gate.reasoning` | Per-claim penalty trace | Full decision audit trail |
| `compliance_evidence[].alignment_score` | Cosine similarity of req vs sections | How well BRD maps to RBI law |
| `compliance_evidence[].gaps` | Gemini gap analysis | What is missing from the BRD |
| `edition_traceability` | Cross-edition similarity comparison | Which RBI edition is authoritative |

### Full Pipeline Math Summary

```
Query q
  │
  ▼
[Retriever]    S = q·Kᵀ/‖q‖‖K‖  +  BM25 → RRF → top-10 chunks C
  │
  ▼
[Generator]    A = softmax(QKᵀ/√dₖ)V  → Answer + claims {c₁,...,cₙ} with citations
  │
  ▼
[NLI Verifier] ∀cᵢ → P(ENTAILMENT|cᵢ.passage, cᵢ.text) = entailment_score(cᵢ)
  │
  ▼
[Trust Gate]   Overall_Score = 1 - Σ penalties(cᵢ)  → Status ∈ {SAFE, REVIEW, NON_COMPLIANT}
  │
  ▼
[BRD Mapper]   alignment(r,sⱼ) = r·sⱼᵀ/‖r‖‖sⱼ‖  → Compliance_Score + gaps + violations
  │
  ▼
[Audit Store]  Full JSON artifact → SQLite  (complete mathematical trace)
```

Every step is **deterministic**, **traceable**, and **reconstructible** from the stored audit record. This is what makes the framework genuinely "Explainable AI" rather than a black box.
