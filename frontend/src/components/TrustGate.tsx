import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { CheckCircle2, AlertTriangle, XCircle, ShieldCheck } from "lucide-react";
import { Progress } from "./ui/progress";

export interface TrustGateResult {
  status: "safe" | "review" | "noncompliant";
  score: number;
  reasoning: string;
  verificationDetails: {
    totalClaims: number;
    verifiedClaims: number;
    conflictingClaims: number;
    unsupportedClaims: number;
  };
  checks: Array<{
    name: string;
    status: "passed" | "failed" | "warning";
    message: string;
  }>;
  editionConsistency: {
    consistent: boolean;
    conflictCount: number;
    details: string[];
  };
  confidenceMetrics: {
    avgConfidence: number;
    minConfidence: number;
    maxConfidence: number;
  };
}

export function TrustGate({ result }: { result: TrustGateResult }) {
  const isSafe = result.status === 'safe';
  
  return (
    <Card className="p-6 border-l-4 bg-background" style={{ borderLeftColor: isSafe ? 'var(--safe)' : 'var(--warn)' }}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold flex items-center gap-2">
            {isSafe ? <ShieldCheck className="text-safe" /> : <AlertTriangle className="text-warn" />}
            TrustGate Validation
            <Badge variant="outline" className={isSafe ? "text-safe border-safe/30 bg-safe/10" : "text-warn border-warn/30 bg-warn/10"}>
              {result.score}/100
            </Badge>
          </h3>
          <p className="text-muted-foreground mt-2">{result.reasoning}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Verification Checks</h4>
          <div className="space-y-3">
            {result.checks.map(check => (
              <div key={check.name} className="flex items-start gap-2 text-sm">
                <div className="mt-0.5">
                  {check.status === 'passed' ? <CheckCircle2 className="h-4 w-4 text-safe" /> : <XCircle className="h-4 w-4 text-danger" />}
                </div>
                <div>
                  <span className="font-medium text-foreground block">{check.name}</span>
                  <span className="text-muted-foreground">{check.message}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Confidence Metrics</h4>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-muted-foreground">Average Confidence</span>
                <span className="text-foreground">{(result.confidenceMetrics.avgConfidence * 100).toFixed(1)}%</span>
              </div>
              <Progress value={result.confidenceMetrics.avgConfidence * 100} className="h-2" />
            </div>
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="bg-accent/30 border border-border p-3 rounded-lg text-center">
                <div className="text-xs text-muted-foreground mb-1">Detailed Claims</div>
                <div className="text-2xl font-bold text-foreground">
                  {result.verificationDetails.verifiedClaims} <span className="text-sm font-normal text-muted-foreground">/ {result.verificationDetails.totalClaims}</span>
                </div>
              </div>
              <div className="bg-accent/30 border border-border p-3 rounded-lg text-center">
                <div className="text-xs text-muted-foreground mb-1">Conflicts</div>
                <div className="text-2xl font-bold text-foreground">{result.verificationDetails.conflictingClaims}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
