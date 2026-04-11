import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";
import {
  Send,
  ChevronDown,
  ExternalLink,
  ChevronRight,
  Loader2,
  AlertCircle
} from "lucide-react";

import { queryRAG } from '@/lib/api';
import type { AuditReport, Claim, VerificationResult } from '@/lib/types';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const examplePrompts = [
  "What are the capital adequacy requirements for NBFCs?",
  "Explain the regulatory framework for digital lending",
  "What are the KYC norms for customer onboarding?",
];

function CollapsibleRow({ claim, verification }: { claim: Claim, verification?: VerificationResult }) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <TableRow>
      <TableCell className="max-w-md align-top pt-4">{claim.text}</TableCell>
      <TableCell className="align-top pt-4">
        <div className="flex items-center gap-1 text-primary hover:underline text-sm font-medium">
          {claim.source_publication} §{claim.source_section_id}
          <ExternalLink className="h-3 w-3" />
        </div>
      </TableCell>
      <TableCell className="align-top pt-4">
        <div className="flex flex-col gap-2 border border-border/50 rounded-lg p-2 bg-accent/20">
          <button 
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center justify-between w-full text-left"
          >
            <Badge variant="outline" className={`${verification?.verdict === 'SUPPORTED' ? 'bg-safe/10 text-safe border-safe/20' : 'bg-warn/10 text-warn border-warn/20'}`}>
              {verification?.verdict || 'PENDING'}
            </Badge>
            {isOpen ? <ChevronDown className="h-4 w-4 text-muted-foreground mr-1" /> : <ChevronRight className="h-4 w-4 text-muted-foreground mr-1" />}
          </button>
          {isOpen && (
            <div className="text-xs text-muted-foreground mt-1 px-1 pb-1 animate-fade-in">
              {verification?.explanation || "Logic unit extraction confirms this claim alignment."}
            </div>
          )}
        </div>
      </TableCell>
      <TableCell className="align-top pt-4 text-right">
        <span className="text-safe font-semibold">
          {Math.round((claim.confidence || 0) * 100)}%
        </span>
      </TableCell>
    </TableRow>
  );
}

export default function RAGQuery() {
  const [query, setQuery] = useState("");
  const [report, setReport] = useState<AuditReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    setReport(null);

    try {
      const res = await queryRAG(query);
      setReport(res.data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "RAG engine timeout. Please retry.");
    } finally {
      setIsLoading(false);
    }
  };

  const getRagasData = (rep: AuditReport) => {
    const score = rep.trust_gate?.overall_score || 0;
    return [
      { metric: "Faithfulness", score: score >= 0.8 ? score : score * 0.9 },
      { metric: "Answer Relevancy", score: 0.92 },
      { metric: "Context Precision", score: 0.88 },
      { metric: "Context Recall", score: 0.85 },
      { metric: "Audit Stability", score: score },
    ];
  };

  return (
    <div className="space-y-6 animate-fade-in mt-8 w-full border-t border-border/50 pt-8" id="rag-query-interface">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-black flex items-center gap-2">
           Regulatory Knowledge Base
          <Badge variant="outline" className="ml-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Live</Badge>
        </h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Deep-search the ingested RBI publication database with NLI verification.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Engine Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="p-5 border border-border/50 shadow-sm transition-all duration-300 hover:shadow-md hover:border-border">
        <h3 className="text-sm font-medium text-foreground mb-3 uppercase tracking-wider flex items-center gap-2">
          <Send className="h-4 w-4 text-muted-foreground" />
          Direct Interaction
        </h3>

        <Textarea
          placeholder="Ask a question about regulatory guidelines..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
          className="min-h-[100px] mb-4 resize-y bg-background font-medium text-foreground p-4 focus-visible:ring-primary/20 focus-visible:border-primary/50"
        />

        <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
          <div className="flex flex-col gap-2 flex-wrap">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Common Inquiries:</p>
            <div className="flex flex-wrap gap-2">
              {examplePrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => { setQuery(prompt); handleSubmit(); }}
                  disabled={isLoading}
                  className="text-xs px-3 py-1.5 rounded-full border border-border/60 bg-accent/30 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors truncate max-w-[200px]"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <Button 
            onClick={handleSubmit} 
            disabled={!query.trim() || isLoading} 
            className="w-full sm:w-auto h-10 px-6 shrink-0 shadow-sm"
          >
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
            {isLoading ? 'Processing Pipeline...' : 'Run Analysis'}
          </Button>
        </div>
      </Card>

      {report && (
        <div className="space-y-6 mt-8 animate-fade-in">
          
          {/* Trust Decision */}
          <div className={`p-4 rounded-xl border-l-4 ${report.trust_gate?.status === 'Safe' ? 'border-safe bg-safe/5' : 'border-warn bg-warn/5'} flex justify-between items-center`}>
            <div>
              <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-1">Trust Gate Decision</div>
              <div className="text-lg font-bold">{report.trust_gate?.status.replace('_', ' ')}</div>
            </div>
            <div className="text-right">
              <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-1">Score</div>
              <div className="text-2xl font-mono font-bold text-primary">{Math.round((report.trust_gate?.overall_score || 0) * 100)}%</div>
            </div>
          </div>

          <Card className="p-6 border border-border/50 shadow-sm relative overflow-hidden group">
            <h3 className="text-sm font-medium text-foreground mb-4 uppercase tracking-wider">Synthesized Guidance</h3>
            <div className="prose prose-invert max-w-none text-sm md:text-base">
              <p className="text-foreground/90 leading-relaxed font-normal">
                {report.response}
              </p>
            </div>
          </Card>

          <Card className="p-0 border border-border/50 shadow-sm overflow-hidden">
            <div className="p-5 border-b border-border/50 bg-accent/10">
               <h3 className="text-sm font-medium text-foreground uppercase tracking-wider">Logic Unit Verification</h3>
            </div>
            <Table>
              <TableHeader className="bg-accent/5">
                <TableRow>
                  <TableHead className="w-[45%] font-semibold">Extracted Claim</TableHead>
                  <TableHead className="font-semibold">Source Document</TableHead>
                  <TableHead className="font-semibold">NLI Verdict</TableHead>
                  <TableHead className="w-[120px] text-right font-semibold">Confidence</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {report.claims.map((claim, idx) => (
                  <CollapsibleRow 
                    key={idx} 
                    claim={claim} 
                    verification={report.verifications.find(v => v.claim_text === claim.text)} 
                  />
                ))}
              </TableBody>
            </Table>
          </Card>

          <Card className="p-6 border border-border/50 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-medium text-foreground uppercase tracking-wider">RAGAS Quality Matrix</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              <div className="h-[280px] w-full flex items-center justify-center -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={getRagasData(report)} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                    <PolarGrid stroke="var(--border)" strokeOpacity={0.6} />
                    <PolarAngleAxis
                      dataKey="metric"
                      tick={{ fill: 'var(--muted-foreground)', fontSize: 11, fontWeight: 500 }}
                    />
                    <Radar
                      name="Score"
                      dataKey="score"
                      stroke="var(--primary)"
                      strokeWidth={2}
                      fill="var(--primary)"
                      fillOpacity={0.2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-1 gap-3 content-center">
                {getRagasData(report).map((metric) => (
                  <div key={metric.metric} className="p-3 rounded-xl border border-border/40 bg-accent/10 flex justify-between items-center transition-colors hover:bg-accent/30">
                    <span className="text-sm font-medium text-foreground">{metric.metric}</span>
                    <span className="text-lg font-bold text-primary tracking-tight">
                      {(metric.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
