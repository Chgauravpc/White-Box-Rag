import React from "react";
import { cn } from "@/lib/utils";
import { ShieldCheck, AlertTriangle, ShieldX } from "lucide-react";
import type { TrustStatus } from "@/lib/types";

const CFG: Record<TrustStatus, { icon: React.ElementType; label: string; cls: string; glow: string }> = {
  Safe: { icon: ShieldCheck, label: "Safe", cls: "bg-safe/15 text-safe border-safe/30", glow: "glow-safe" },
  Needs_Human_Review: { icon: AlertTriangle, label: "Needs Review", cls: "bg-review/15 text-review border-review/30", glow: "glow-review" },
  Non_Compliant: { icon: ShieldX, label: "Non-Compliant", cls: "bg-noncompliant/15 text-noncompliant border-noncompliant/30", glow: "glow-danger" },
};

export default function TrustBadge({ status, size="md" }: { status: TrustStatus; size?: "sm"|"md"|"lg" }) {
  const c = CFG[status]; const Icon = c.icon;
  const sz = { sm:"px-2 py-1 text-xs", md:"px-3 py-1.5 text-sm", lg:"px-5 py-3 text-base" };
  return (
    <span className={cn("inline-flex items-center gap-2 rounded-full border font-medium", c.cls, c.glow, sz[size])}>
      <Icon className={size==="lg"?"h-5 w-5":"h-4 w-4"} /> {c.label}
    </span>
  );
}
