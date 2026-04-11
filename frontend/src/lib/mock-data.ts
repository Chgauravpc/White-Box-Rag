import type { VerifiedResponse, DocumentInfo, BRDValidation, AuditLog } from "./types";

export const MOCK_DOCUMENTS: DocumentInfo[] = [
  { id:"1", publication_name:"FSR", edition_date:"June 2024", filename:"FSR_June_2024.pdf",
    total_sections:42, total_chunks:186, ingested_at:"2024-06-15T10:00:00Z" },
  { id:"2", publication_name:"MPR", edition_date:"December 2024", filename:"MPR_Dec_2024.pdf",
    total_sections:38, total_chunks:154, ingested_at:"2024-12-20T10:00:00Z" },
];

export const MOCK_VERIFIED: VerifiedResponse = {
  rag: {
    answer: "The RBI's Financial Stability Report highlights that the gross NPA ratio of SCBs declined to 3.2% in March 2024 [FSR·June 2024·§2.1.1]. The provision coverage ratio improved to 76.3% [FSR·June 2024·§2.1.3].",
    claims: [
      { text:"The gross NPA ratio of SCBs declined to 3.2% in March 2024", source_publication:"FSR",
        source_edition:"June 2024", source_section_id:"2.1.1",
        source_passage:"The GNPA ratio of SCBs continued its declining trend to reach 3.2 per cent in March 2024...",
        confidence:0.94 },
      { text:"The provision coverage ratio improved to 76.3%", source_publication:"FSR",
        source_edition:"June 2024", source_section_id:"2.1.3",
        source_passage:"The PCR of SCBs improved to 76.3 per cent at end-March 2024...", confidence:0.91 }
    ], retrieved_sections: []
  },
  verifications: [
    { claim_text:"The gross NPA ratio of SCBs declined to 3.2%", verdict:"ENTAILMENT", entailment_score:0.94,
      explanation:"Source directly confirms GNPA ratio of 3.2% for March 2024." },
    { claim_text:"The provision coverage ratio improved to 76.3%", verdict:"ENTAILMENT", entailment_score:0.91,
      explanation:"Source explicitly states PCR of 76.3%." }
  ],
  trust_gate: { status:"Safe", reasoning:"All claims ENTAILMENT. High confidence. No edition conflicts.", overall_score:92 },
  conflicts: [],
  scorecard: { context_relevance:0.95, faithfulness:0.93, citation_precision:1.0, edition_conflict_risk:0.0, paraphrase_stability:0.88 }
};

export const MOCK_BRD: BRDValidation = {
  requirements: [
    { id:"REQ-001", text:"System shall calculate NPA ratios quarterly", category:"Asset Quality",
      regulatory_relevance:"FSR", mapped_sections:[], alignment_score:0.87,
      gaps:["Does not specify stressed assets methodology"], violations:[], risk_flags:["MEDIUM"],
      remediation:"Add clause referencing FSR §2.1 stressed assets framework." },
    { id:"REQ-002", text:"Payment system uptime must exceed 99.5%", category:"Payment Systems",
      regulatory_relevance:"PSR", mapped_sections:[], alignment_score:0.62,
      gaps:["RBI mandates 99.9% for critical payment infra"], violations:["Threshold below RBI minimum"],
      risk_flags:["HIGH"], remediation:"Update uptime to 99.9% per PSR §4.2.1." }
  ],
  overall_compliance_score:74, summary:"2 requirements analyzed. 1 gap and 1 violation found."
};

export const MOCK_AUDITS: AuditLog[] = [
  { id:"a1", timestamp:"2024-06-15T10:30:00Z", query:"Key risks to financial stability?", trust_status:"Safe", overall_score:92 },
  { id:"a2", timestamp:"2024-06-15T11:15:00Z", query:"RBI stance on digital lending?", trust_status:"Needs_Human_Review", overall_score:68 },
  { id:"a3", timestamp:"2024-06-15T12:00:00Z", query:"Compare NPA trends across years", trust_status:"Non_Compliant", overall_score:31 },
];
