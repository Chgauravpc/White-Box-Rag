export default function ScoreGauge({ score, label }: { score: number; label: string }) {
  const r=45, c=2*Math.PI*r, offset=c-(score/100)*c;
  const color = score>=80?"var(--safe)":score>=50?"var(--review)":"var(--noncompliant)";
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--border)" strokeWidth="8"/>
        <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 60 60)" style={{transition:"stroke-dashoffset 1s ease"}}/>
        <text x="60" y="55" textAnchor="middle" className="fill-foreground text-2xl font-bold">{score}</text>
        <text x="60" y="72" textAnchor="middle" className="fill-muted-foreground text-xs">/100</text>
      </svg>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  );
}
