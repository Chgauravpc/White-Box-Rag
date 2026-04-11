import React, { useEffect, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { FileText, MessageSquare, ShieldCheck, AlertCircle } from "lucide-react";

const useAnimatedCounter = (target: number, duration: number = 2000) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = target;
    if (start === end) return;

    const timer = setInterval(() => {
      start += 1;
      setCount(prev => Math.min(prev + Math.ceil(end / (duration / 16)), end));
      if (start >= end) clearInterval(timer);
    }, 16);

    return () => clearInterval(timer);
  }, [target, duration]);

  return count;
};

const AnimatedNumber = ({ value, suffix = "" }: { value: number; suffix?: string }) => {
  const displayValue = useAnimatedCounter(value);
  return <>{displayValue.toLocaleString()}{suffix}</>;
};

const stats = [
  {
    label: "Documents Ingested",
    value: 1284,
    icon: FileText,
    color: "text-blue-500",
  },
  {
    label: "Queries Processed",
    value: 4592,
    icon: MessageSquare,
    color: "text-purple-500",
  },
  {
    label: "Avg Trust Score",
    value: 94,
    icon: ShieldCheck,
    color: "text-emerald-500",
    suffix: "%"
  },
  {
    label: "Active Conflicts",
    value: 12,
    icon: AlertCircle,
    color: "text-rose-500",
  },
];

export function StatsCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card 
            key={stat.label} 
            className="p-0 border-none bg-card shadow-sm shadow-primary/5 group transition-all duration-500 hover:shadow-xl hover:shadow-primary/10 overflow-hidden relative"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <CardContent className="p-8 relative z-10">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-[11px] font-bold text-muted-foreground uppercase tracking-[0.2em]">{stat.label}</p>
                  <p className="text-5xl font-bold mt-4 text-foreground tracking-tighter font-heading">
                    <AnimatedNumber value={stat.value} suffix={stat.suffix} />
                  </p>
                </div>
                <div className={cn("p-5 rounded-3xl bg-secondary/50 transition-all duration-500 group-hover:bg-primary/10 group-hover:scale-110", stat.color)}>
                  <Icon className="h-7 w-7" />
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
