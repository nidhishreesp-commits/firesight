import { useEffect, useState } from "react";
import { api } from "../api";

export default function ModelComparison() {
  const [models, setModels] = useState<any[]>([]);
  const [bench, setBench] = useState<any>(null);
  const [video, setVideo] = useState<any>(null);

  useEffect(() => {
    api.models().then((r) => setModels(r.models ?? [])).catch(() => undefined);
    api.benchmark().then(setBench).catch(() => undefined);
    fetch("/api/artifacts/video_ablation/summary.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setVideo)
      .catch(() => undefined);
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Model comparison</h1>
      <section>
        <h2 className="mb-2 text-lg">Registry</h2>
        <div className="overflow-auto rounded border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60">
              <tr>
                <th className="px-2 py-1">id</th>
                <th className="px-2 py-1">family</th>
                <th className="px-2 py-1">license</th>
                <th className="px-2 py-1">trained on / notes</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} className="border-t border-slate-800">
                  <td className="px-2 py-1">{m.id}</td>
                  <td className="px-2 py-1">{m.family}</td>
                  <td className="px-2 py-1">{m.license}</td>
                  <td className="px-2 py-1 text-slate-400">{m.trained_on} — {m.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section>
        <h2 className="mb-2 text-lg">Video ablation (time-to-detection / false alarms)</h2>
        <pre className="overflow-auto rounded border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-300">
          {JSON.stringify(video, null, 2)}
        </pre>
      </section>
      <section>
        <h2 className="mb-2 text-lg">Latest benchmark tables</h2>
        <pre className="max-h-96 overflow-auto rounded border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-300">
          {JSON.stringify(bench, null, 2)}
        </pre>
      </section>
    </div>
  );
}
