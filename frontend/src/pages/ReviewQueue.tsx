import { useEffect, useState } from "react";
import { api } from "../api";

export default function ReviewQueue() {
  const [rows, setRows] = useState<any[]>([]);

  const load = () => api.reviewQueue().then((r) => setRows(r.items)).catch(() => undefined);
  useEffect(() => {
    load();
  }, []);

  const decide = async (row: any, verdict: string) => {
    await api.feedback({ record_id: row.record_id, verdict, note: row.note ?? "", model: row.model, label: row.label });
    load();
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Review queue</h1>
      <p className="text-sm text-slate-400">
        Operator decisions are appended to <code>data/feedback/queue.jsonl</code> and feed the retraining pipeline.
      </p>
      <div className="overflow-auto rounded border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-800/60">
            <tr>
              <th className="px-3 py-2">record</th>
              <th className="px-3 py-2">verdict</th>
              <th className="px-3 py-2">model</th>
              <th className="px-3 py-2">note</th>
              <th className="px-3 py-2">actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-slate-800">
                <td className="px-3 py-2">{r.record_id}</td>
                <td className="px-3 py-2">{r.verdict}</td>
                <td className="px-3 py-2">{r.model ?? "-"}</td>
                <td className="px-3 py-2">{r.note ?? "-"}</td>
                <td className="px-3 py-2 space-x-2">
                  <button className="rounded bg-emerald-700 px-2 py-0.5 text-xs" onClick={() => decide(r, "approved")}>
                    approve
                  </button>
                  <button className="rounded bg-red-800 px-2 py-0.5 text-xs" onClick={() => decide(r, "rejected")}>
                    reject
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
