import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

const complianceData = [
  { name: "Safe", value: 1245, color: "#10b981" },
  { name: "Review", value: 456, color: "#f59e0b" },
  { name: "Non-Compliant", value: 133, color: "#ef4444" },
];

export function TrustDistribution() {
  return (
    <Card className="border-none bg-card shadow-sm shadow-primary/5 flex flex-col h-full">
      <CardHeader className="p-8 pb-4">
        <CardTitle className="text-2xl font-bold font-heading tracking-tight">Trust Distribution</CardTitle>
      </CardHeader>
      <CardContent className="p-8 pt-4 flex-1 flex flex-col justify-center">
        <div className="h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={complianceData}
                cx="50%"
                cy="50%"
                innerRadius={80}
                outerRadius={110}
                paddingAngle={10}
                dataKey="value"
                stroke="none"
              >
                {complianceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: 'none',
                  borderRadius: '24px',
                  boxShadow: '0 20px 50px rgba(0,0,0,0.1)',
                  padding: '20px'
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={40}
                iconType="circle"
                formatter={(value) => <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground ml-2">{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="grid grid-cols-3 gap-6 mt-10 pt-10 border-t border-secondary/50 text-center">
          {complianceData.map((item) => (
            <div key={item.name}>
              <p className="text-3xl font-bold tracking-tighter font-heading" style={{ color: item.color }}>
                {((item.value / 1834) * 100).toFixed(0)}%
              </p>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground mt-2">{item.name}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
