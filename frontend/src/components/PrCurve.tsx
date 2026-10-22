export function PrCurve({ curve, width = 420, height = 260 }: { curve: { fpr: number; recall: number; threshold: number }[]; width?: number; height?: number }) {
  if (!curve || curve.length === 0) return <div className="text-sm text-slate-400">No PR curve data.</div>;
  const points = [...curve].sort((a, b) => a.fpr - b.fpr);
  const maxX = Math.max(0.02, Math.min(1, Math.max(...points.map((p) => p.fpr))));
  const path = points
    .map((p, i) => {
      const x = (Math.min(p.fpr, maxX) / maxX) * width;
      const y = height - p.recall * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} className="rounded border border-slate-700 bg-slate-900">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={0} x2={width} y1={height - g * height} y2={height - g * height} stroke="#1e293b" />
          <text x={4} y={height - g * height - 3} fill="#64748b" fontSize={10}>
            {(g * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke="#f97316" strokeWidth={2} />
      <text x={width / 2} y={height - 4} fill="#94a3b8" fontSize={11} textAnchor="middle">
        false-alarm rate (max {(maxX * 100).toFixed(1)}%)
      </text>
      <text x={4} y={12} fill="#94a3b8" fontSize={11}>
        recall
      </text>
    </svg>
  );
}
