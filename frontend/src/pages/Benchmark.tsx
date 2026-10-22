import { useEffect, useState } from "react";
import { api } from "../api";
import { PrCurve } from "../components/PrCurve";

export default function Benchmark() {
  const [data, setData] = useState<any>(null);
  const [selected, setSelected] = useState("stage1_full_test_bench");
  const [curve, setCurve] = useState<any[]>([]);

  useEffect(() => {
    api.benchmark().then(setData).catch(() => undefined);
  }, []);

  useEffect(() => {
    const model = "ssprod_best_fire_v3";
    fetch(`/api/artifacts/benchmark_stage1/${model}/metrics.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => setCurve(m?.pr_curve ?? []))
      .catch(() => undefined);
  }, []);

  const table: any[] = Array.isArray(data?.[selected]) ? data[selected] : [];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Run benchmark</h1>
      <div className="flex items-center gap-2">
        <select className="rounded bg-slate-800 px-2 py-1 text-sm" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {Object.keys(data ?? {}).map((k) => (
            <option key={k}>{k}</option>
          ))}
        </select>
        <span className="text-sm text-slate-400">Thresholds calibrated on val only; frozen test reported once.</span>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="overflow-auto rounded border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-300">
              <tr>
                {table[0] && Object.keys(table[0]).map((k) => <th key={k} className="px-2 py-1">{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {table.map((row, i) => (
                <tr key={i} className="border-t border-slate-800">
                  {Object.values(row).map((v, j) => (
                    <td key={j} className="px-2 py-1 text-slate-300">
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mb-2 text-sm text-slate-300">ssprod_best_fire_v3: image recall vs false-alarm rate (frozen test_bench)</div>
          <PrCurve curve={curve} />
        </div>
      </div>
    </div>
  );
}
