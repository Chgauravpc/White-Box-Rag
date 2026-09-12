"""
Query Pipeline — the full RAG + Verification + Compliance Audit orchestration.

Extracted from ingestion/routes.py so it has one canonical call path usable
both from the HTTP route and directly (no HTTP round-trip) by the offline
evaluation harness (backend/eval/harness.py).
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from shared.models import (
    AuditReport,
    BRDRequirement,
    XAIArtifacts,
    RetrievalMatrix,
    EntailmentMatrix,
    AttributionMatrix,
    ShapleyContributions,
)
from ingestion.rag import rag_query
from ingestion.retriever import hybrid_retrieve

from verification.nli_engine import verify_all_claims
from verification.trust_gate import compute_trust_gate
from verification.mitigation import filter_claims, should_abstain, ABSTENTION_MESSAGE_TEMPLATE
from verification.counterfactual import compute_counterfactuals
from verification.scorecard import generate_scorecard
from verification.stability import compute_paraphrase_stability
from verification.edition_conflict import discover_and_check_conflicts
from compliance.mapper import map_requirement
from compliance.audit import generate_audit_report
from shared.xai_matrices import (
    build_retrieval_similarity_matrix,
    compute_shapley_contributions,
    compute_primary_attributions,
    find_related_queries,
    get_encoder,
    compute_answer_relevancy,
    compute_context_utilization,
    compute_context_diversity,
)
from shared.database import store_query_embedding, get_past_query_embeddings, finalize_audit_record
from shared.models import NLIVerdict
from shared.chunk_key import resolve_chunk_key
from shared.llm import llm_call_scope, get_current_llm_calls

logger = logging.getLogger(__name__)

# Serializes the audit-chain finalize step. The backend is a single uvicorn
# process, and the eval harness fans out /query under asyncio.Semaphore(3);
# without this lock two concurrent finalizes could read the same predecessor
# and fork the hash chain.
_chain_lock = asyncio.Lock()


def _faithfulness(verifications: list) -> float:
    """entailed/total — 1.0 (vacuously faithful) when there's nothing to verify."""
    if not verifications:
        return 1.0
    entailed = sum(1 for v in verifications if v.verdict in (NLIVerdict.ENTAILMENT, NLIVerdict.SUPPORTED))
    return round(entailed / len(verifications), 6)


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate retrieved chunks: keep one (highest-index = highest RRF) per
    chunk identity.

    Keys on the canonical chunk_key (falling back to a content-addressed key
    for legacy pre-migration data), NOT bare section_id — two documents that
    happen to share a section_id (e.g. both have a "1.1") must not collide
    and silently drop one another's chunk out of context.
    """
    seen: dict[str, dict] = {}
    dropped = 0
    for chunk in chunks:
        key = resolve_chunk_key(
            chunk["publication_name"], chunk["edition_date"], chunk["section_id"],
            chunk["chunk_text"], chunk.get("chunk_key", ""),
        )
        if key not in seen:
            seen[key] = chunk  # first occurrence = highest RRF rank
        else:
            dropped += 1
    if dropped:
        logger.debug(f"_deduplicate_chunks: collapsed {dropped} true duplicate chunk(s)")
    return list(seen.values())


async def run_query_pipeline(
    query: str,
    filters: dict | None = None,
    eval_mode: bool = False,
    abstention_threshold: Optional[float] = None,
) -> AuditReport:
    """Thin wrapper: opens a Phase-2 llm_call_scope() around the whole query
    so every call_llm() made anywhere in the pipeline (rag.py, mapper.py,
    audit.py, stability.py) is collected into the returned report's
    llm_calls, then delegates to _run_query_pipeline for the actual work."""
    with llm_call_scope():
        report = await _run_query_pipeline(query, filters, eval_mode, abstention_threshold)
        report.llm_calls = get_current_llm_calls()
        return report


async def _run_query_pipeline(
    query: str,
    filters: dict | None = None,
    eval_mode: bool = False,
    abstention_threshold: Optional[float] = None,
) -> AuditReport:
    """Ask a question and get a fully integrated RAG + Verification + Compliance Audit response.

    The output includes all XAI mathematical artifacts:
    - retrieval_similarity_matrix: S = q·Kᵀ/(‖q‖·‖k‖) for every retrieved chunk
    - shapley: Shapley-style penalty decomposition
    - faithfulness: |ENTAILMENT| / total claims
    - related_queries: past queries with cosine_sim > 0.70

    Args:
        eval_mode: Skips Step 5 (requirement mapping) and Step 6 (LLM
            audit-report generation) — governance bookkeeping that no
            benchmark metric reads (see eval/harness.py::_run_one), and that
            would otherwise burn 2 of the 4 LLM calls per query and write a
            row into the production audit trail for every benchmark item.
            The live `/query` HTTP route never sets this — default is False,
            live-query behavior is unchanged.
        abstention_threshold: When set, passed straight to
            `should_abstain(..., threshold=...)` instead of letting it read
            `verification/conformal.py::load_active_threshold()` itself. Lets
            an eval run freeze the threshold once at run start so a
            concurrent recalibration can't change results mid-run (finding #20).
    """
    latency_ms: dict[str, float] = {}
    llm_call_count = 0

    # ── Step 1: Hybrid Retrieval & Retrieval Matrix (S) ──
    logger.info(f"Step 1: Retrieving chunks for: '{query[:80]}'")
    _t0 = time.monotonic()
    raw_chunks = hybrid_retrieve(query=query, filters=filters)

    # Convert to dicts, deduplicate by chunk identity (not bare section_id —
    # see _deduplicate_chunks)
    dict_chunks = [{
        "chunk_text":       c.chunk_text,
        "section_id":       c.section_id,
        "publication_name": c.publication_name,
        "edition_date":     c.edition_date,
        "chunk_key":        c.chunk_key,
    } for c in raw_chunks]
    dict_chunks = _deduplicate_chunks(dict_chunks)

    # Matrix 1 (S) — Cosine similarity of query vs each unique chunk
    chunk_ids, S_scores = build_retrieval_similarity_matrix(query, dict_chunks)
    latency_ms["retrieval_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 2: Generation & Attribution Matrix (A) ──
    _t0 = time.monotonic()
    rag_response, A_matrix = await rag_query(query, dict_chunks)
    llm_call_count += 1
    latency_ms["generation_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 3: NLI Verification, Entailment Matrix (E), focused passages ──
    logger.info("Step 3: Batched NLI verification via CrossEncoder")
    _t0 = time.monotonic()
    verifications, E_matrix, focused_passages, premise_deletions = await verify_all_claims(rag_response.claims)
    latency_ms["verification_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # Stamp focused_passage/premise_deletions onto each Claim for full audit traceability
    for i, claim in enumerate(rag_response.claims):
        if i < len(focused_passages):
            claim.focused_passage = focused_passages[i]
        if i < len(premise_deletions):
            claim.premise_deletions = premise_deletions[i]

    # ── Step 4: XAI Math — Shapley (φ), primary attributions, assemble artifacts ──
    logger.info("Step 4: Computing Shapley and attribution artifacts")

    A_scores = A_matrix.tolist() if isinstance(A_matrix, np.ndarray) and A_matrix.size > 0 else []
    E_scores = E_matrix.tolist() if isinstance(E_matrix, np.ndarray) and E_matrix.size > 0 else []

    # Primary attribution per sentence (argmax + runner-up)
    prim_attrs = compute_primary_attributions(A_matrix, chunk_ids) if A_scores else []

    # Edition conflicts — real cross-edition contradiction discovery. Guarded
    # internally against UNSTRUCTURED (page-level) sections and only fires for
    # sections with genuine multi-edition versioning in the knowledge base.
    _t0 = time.monotonic()
    conflicts = await discover_and_check_conflicts(rag_response.claims)
    latency_ms["conflict_detection_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # Trust gate — computed once prim_attrs are available (NLI + attribution quality)
    trust_gate = compute_trust_gate(verifications, conflicts, prim_attrs)

    # Shapley: mirrors trust_gate penalties exactly (NLI + attribution)
    shapley = compute_shapley_contributions(
        [v.model_dump() for v in verifications],
        prim_attrs,
    )

    # Counterfactuals: per-claim "what would change the verdict" (pure math).
    counterfactuals = compute_counterfactuals(verifications, prim_attrs, conflicts, trust_gate)

    retrieval_mat = RetrievalMatrix(
        chunk_ids=chunk_ids,
        similarity_scores=S_scores,
    )
    attr_mat = AttributionMatrix(
        sentence_texts=[c.text for c in rag_response.claims],
        chunk_ids=chunk_ids,
        scores=A_scores,
        primary_attributions=prim_attrs,
    )
    # Keyed by publication|edition|section, not bare section_id — two documents
    # citing their own "1.1" must not overwrite each other's passage here.
    seen_passages = {}
    for c in rag_response.claims:
        passage_key = f"{c.source_publication}|{c.source_edition}|{c.source_section_id}"
        if c.source_passage and passage_key not in seen_passages:
            seen_passages[passage_key] = c.source_passage
    entail_mat = EntailmentMatrix(
        claim_texts=[c.text for c in rag_response.claims],
        passage_ids=list(seen_passages.keys()),
        passage_texts=list(seen_passages.values()),
        scores=E_scores,
        labels=["contradiction", "entailment", "neutral"],
    )
    shapley_mat = ShapleyContributions(
        claim_texts=shapley["claim_texts"],
        shapley_values=shapley["shapley_values"],
        penalty_reasons=shapley["penalty_reasons"],
        overall_score=shapley["overall_score"],
    )
    xai_artifacts = XAIArtifacts(
        retrieval=retrieval_mat,
        entailment=entail_mat,
        attribution=attr_mat,
        conflict=None,
        shapley=shapley_mat,
    )

    # ── Step 4b: Hallucination mitigation — claim-level filtering + abstention ──
    logger.info("Step 4b: Applying claim-level filtering and abstention checks")
    retained_texts, retained_flags, filter_reasons = filter_claims(rag_response.claims, verifications, prim_attrs)
    for i, claim in enumerate(rag_response.claims):
        if i < len(retained_flags):
            claim.retained = retained_flags[i]
            claim.filter_reason = filter_reasons[i]

    abstained, raw_abstain_reason, retained_trust_score = should_abstain(
        rag_response.claims, verifications, prim_attrs, retained_flags, conflicts,
        threshold=abstention_threshold,
    )
    abstention_reason = ABSTENTION_MESSAGE_TEMPLATE.format(reason=raw_abstain_reason) if abstained else ""

    faithfulness_raw = _faithfulness(verifications)
    retained_verifications = [v for v, keep in zip(verifications, retained_flags) if keep]
    faithfulness_post = _faithfulness(retained_verifications)

    # ── Step 4c: Trust Scorecard (real metrics, pure math + conditional self-consistency) ──
    logger.info("Step 4c: Computing trust scorecard")
    _t0 = time.monotonic()
    paraphrase_stability, stability_note = await compute_paraphrase_stability(
        query, dict_chunks, rag_response.claims, trust_gate.status
    )
    latency_ms["stability_check_ms"] = round((time.monotonic() - _t0) * 1000, 2)
    if not stability_note.startswith("Skipped"):
        llm_call_count += 1
    answer_relevancy = compute_answer_relevancy(query, rag_response.answer)
    context_utilization = compute_context_utilization(prim_attrs, chunk_ids)
    context_diversity = compute_context_diversity(dict_chunks)

    scorecard = generate_scorecard(
        query=query,
        response=rag_response.answer,
        verifications=retained_verifications,
        conflicts=conflicts,
        claims=rag_response.claims,
        retrieval_scores=S_scores,
        paraphrase_stability=paraphrase_stability,
        answer_relevancy=answer_relevancy,
        context_utilization=context_utilization,
        context_diversity=context_diversity,
    )

    # ── Step 5 & 6: Requirement/Compliance Mapping + Audit Report (BP3) ──
    # Skipped in eval_mode — see run_query_pipeline's docstring. Neither
    # step's output feeds anything eval/harness.py::_run_one reads, and
    # skipping them halves LLM calls/query for a benchmark run and keeps
    # eval queries out of the production audit trail entirely (log_id stays
    # None below, so Step 7/8 — embedding storage, chain finalization — are
    # skipped too, exactly as if this were a normal "nothing to persist" path).
    if eval_mode:
        logger.info("Step 5/6: eval_mode — skipping requirement mapping and audit-report generation")
        audit_report_dict: dict = {"query": query}
    else:
        logger.info("Step 5: Mapping the query itself as a requirement-alignment gap check")
        _t0 = time.monotonic()
        brd_req = BRDRequirement(id="ASK", text=query)
        mapped_dict = await map_requirement(brd_req)
        llm_call_count += 1
        brd_req.mapped_sections = [c.chunk_text for c in mapped_dict.get("relevant_chunks", [])]
        brd_req.alignment_score = mapped_dict.get("alignment_score", 0.0)
        brd_req.gaps = mapped_dict.get("gaps", [])
        brd_req.risk_flags = mapped_dict.get("violations", [])
        brd_req.risk_level = mapped_dict.get("risk_level", "LOW")
        brd_req.remediation = mapped_dict.get("remediation_suggestions", "")
        latency_ms["compliance_mapping_ms"] = round((time.monotonic() - _t0) * 1000, 2)

        logger.info("Step 6: Compiling Audit Report")
        _t0 = time.monotonic()
        audit_report_dict = await generate_audit_report(
            query=query,
            rag_response=rag_response.answer,
            claims=rag_response.claims,
            verifications=verifications,
            trust_gate=trust_gate,
            edition_conflicts=conflicts,
            brd_results=[brd_req],
        )
        llm_call_count += 1
        latency_ms["audit_report_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 7: Store embedding + find related queries (pure cosine, no LLM) ──
    log_id = audit_report_dict.get("id")
    q_emb = get_encoder().encode(query).tolist()
    if log_id:
        store_query_embedding(log_id, q_emb)

    past = get_past_query_embeddings(exclude_id=log_id)
    related = find_related_queries(query, past)
    # Exclude near-exact matches (same query re-run) from related list
    related = [r for r in related if r.get("cosine_similarity", 0) < 0.99]

    audit_report_dict["xai_artifacts"] = xai_artifacts.model_dump()
    audit_report_dict["related_queries"] = related
    audit_report_dict["response"] = rag_response.answer
    audit_report_dict["claims"] = [c.model_dump() for c in rag_response.claims]
    audit_report_dict["verifications"] = [v.model_dump() for v in verifications]
    audit_report_dict["trust_gate"] = trust_gate.model_dump() if trust_gate else None
    audit_report_dict["edition_conflicts"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in conflicts]
    audit_report_dict["filtered_claims"] = retained_texts
    audit_report_dict["faithfulness_raw"] = faithfulness_raw
    audit_report_dict["faithfulness_post"] = faithfulness_post
    audit_report_dict["retained_trust_score"] = retained_trust_score
    audit_report_dict["abstained"] = abstained
    audit_report_dict["abstention_reason"] = abstention_reason
    audit_report_dict["scorecard"] = scorecard.model_dump()
    audit_report_dict["counterfactuals"] = counterfactuals
    audit_report_dict["latency_ms"] = latency_ms
    audit_report_dict["llm_call_count"] = llm_call_count

    # ── Step 8: Tamper-evident finalization ──
    # Overwrite the partial row written by generate_audit_report with the COMPLETE
    # record and stamp its hash-chain link. Serialized so the chain can't fork.
    if log_id:
        async with _chain_lock:
            prev_hash, record_hash, _ = finalize_audit_record(log_id, audit_report_dict)
        audit_report_dict["prev_hash"] = prev_hash
        audit_report_dict["record_hash"] = record_hash

    return AuditReport(**audit_report_dict)
