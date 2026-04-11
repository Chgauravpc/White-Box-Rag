export interface ChunkMetadata {
  publication_name: string;
  edition_date: string;
  section_id: string;
  section_title: string;
  page_number: number;
  chunk_text: string;
}

export interface DocumentInfo {
  id: number;
  filename: string;
  publication_name: string;
  edition_date: string;
  chunk_count: number;
  ingested_at: string;
}

export interface Claim {
  text: string;
  source_publication: string;
  source_edition: string;
  source_section_id: string;
  source_passage: string;
  confidence: number;
}

export type NLIVerdict = "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO" | "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL";

export interface VerificationResult {
  claim_text: string;
  verdict: NLIVerdict;
  entailment_score: number;
  explanation: string;
}

export interface EditionConflict {
  publication: string;
  section_id: string;
  older_edition: string;
  newer_edition: string;
  has_conflict: boolean;
  conflict_description: string;
  superseding_edition: string;
  details?: string;
}

export type TrustStatus = "Safe" | "Needs_Human_Review" | "Non_Compliant";

export interface TrustGate {
  status: TrustStatus;
  reasoning: string;
  overall_score: number;
}

export interface TrustScorecard {
  context_relevance: number;
  faithfulness: number;
  citation_precision: number;
  edition_conflict_risk: boolean;
  paraphrase_stability: number;
}

export interface BRDRequirement {
  id: string;
  text: string;
  category?: string;
  regulatory_relevance?: string;
  mapped_sections: string[];
  alignment_score: number;
  gaps: string[];
  risk_flags: string[];
  risk_level: "HIGH" | "MEDIUM" | "LOW";
  remediation: string;
}

export interface AuditReport {
  id: number;
  timestamp: string;
  query: string;
  response: string; // The RAG answer text
  claims: Claim[];
  verifications: VerificationResult[];
  trust_gate?: TrustGate;
  edition_conflicts: EditionConflict[];
  
  // Gemini-generated audit fields (loose JSON from backend)
  decision_log?: any[];
  section_reference_registry?: any[];
  edition_traceability?: any;
  compliance_evidence?: BRDRequirement[];
  final_audit_summary?: {
    overall_trust_status: TrustStatus;
    key_risks: string[];
    compliance_score_summary: string;
    reasoning: string;
  };
}

export interface AuditLogItem {
  id: number;
  timestamp: string;
  query: string;
  risk_level: string;
  compliance_score: string;
}
