import React from 'react';
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertTriangle, XCircle, Clock } from "lucide-react";

const statusConfig = {
  Safe: { icon: CheckCircle2, color: "text-emerald-500", bg: "bg-emerald-50", badge: "border-emerald-200 text-emerald-700 bg-emerald-50" },
  Review: { icon: AlertTriangle, color: "text-amber-500", bg: "bg-amber-50", badge: "border-amber-200 text-amber-700 bg-amber-50" },
  "Non-Compliant": { icon: XCircle, color: "text-rose-500", bg: "bg-rose-50", badge: "border-rose-200 text-rose-700 bg-rose-50" },
};

const recentQueries = [
  { id: "1", query: "Capital Adequacy Ratio requirements for FY24", status: "Safe", score: 98, time: "2 mins ago" },
  { id: "2", query: "LCR reporting window for small finance banks", status: "Review", score: 72, time: "15 mins ago" },
  { id: "3", query: "KYC non-compliance penalties for digital lending", status: "Non-Compliant", score: 45, time: "1 hour ago" },
  { id: "4", query: "Priority sector lending targets for urban co-ops", status: "Safe", score: 92, time: "3 hours ago" },
  { id: "5", query: "NPA classification norms for agricultural loans", status: "Review", score: 68, time: "5 hours ago" },
];

export function RecentQueries() {
  const navigate = useNavigate();

  return (
    <Card className="border-none bg-card shadow-sm shadow-primary/5 h-full">
      <CardHeader className="p-8 pb-4">
        <CardTitle className="text-2xl font-bold font-heading tracking-tight">Recent Queries</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border/10">
                <th className="px-8 py-4 text-[11px] font-bold text-muted-foreground uppercase tracking-[0.2em]">Query Details</th>
                <th className="px-8 py-4 text-[11px] font-bold text-muted-foreground uppercase tracking-[0.2em]">Compliance</th>
                <th className="px-8 py-4 text-[11px] font-bold text-muted-foreground uppercase tracking-[0.2em] text-right">Trust Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/5">
              {recentQueries.map((query) => {
                const config = statusConfig[query.status as keyof typeof statusConfig];
                const Icon = config.icon;

                return (
                  <tr 
                    key={query.id} 
                    onClick={() => navigate(`/audit/${query.id}`)}
                    className="group cursor-pointer hover:bg-secondary/30 transition-all duration-300"
                  >
                    <td className="px-8 py-6">
                      <p className="text-base font-semibold text-foreground leading-tight tracking-tight group-hover:text-primary transition-colors line-clamp-1">
                        {query.query}
                      </p>
                      <div className="flex items-center gap-2 mt-2 text-muted-foreground/50">
                        <Clock className="h-3 w-3" />
                        <span className="text-[10px] font-bold uppercase tracking-wider">{query.time}</span>
                      </div>
                    </td>
                    <td className="px-8 py-6">
                      <Badge variant="outline" className={cn("px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest", config.badge)}>
                        <Icon className="h-3 w-3 mr-2" />
                        {query.status}
                      </Badge>
                    </td>
                    <td className="px-8 py-6 text-right">
                      <div className="inline-flex flex-col items-center justify-center p-3 rounded-2xl bg-white shadow-sm border border-border/5 min-w-[64px]">
                        <span className="text-xl font-bold text-foreground leading-none font-heading">{query.score}</span>
                        <span className="text-[8px] font-bold uppercase tracking-widest text-muted-foreground mt-1">Index</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
