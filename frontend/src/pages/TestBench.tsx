import { useEffect, useState } from "react";
import { api } from "../api";
import { BoxOverlay } from "../components/BoxOverlay";
import { TestImage } from "../types";

const filters = ["all", "true_positive", "false_positive", "partial", "true_negative"];

export default function TestBench() {
  const [testset, setTestset] = useState("test_bench");
  const [filter, setFilter] = useState("all");
  const [threshold, setThreshold] = useState(0.5);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<any>(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    setPage(0);
  }, [testset, filter, threshold]);

  useEffect(() => {
    api.testImages(testset, filter, page * 24, 24, threshold).then(setData).catch((e) => setMessage(String(e)));
  }, [testset, filter, page, threshold]);

  const feedback = async (item: TestImage, verdict: string) => {
    await api.feedback({ record_id: item.id, verdict, model: "ssprod_best_fire_v3", label: item.category });
    setMessage(`Recorded feedback '${verdict}' for ${item.id}`);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Test bench</h1>
        <select className="rounded bg-slate-800 px-2 py-1 text-sm" value={testset} onChange={(e) => setTestset(e.target.value)}>
          {["test_bench", "test_mini", "val_bench", "external_cctv_nc", "testbench100"].map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <select className="rounded bg-slate-800 px-2 py-1 text-sm" value={filter} onChange={(e) => setFilter(e.target.value)}>
          {filters.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>
        <label className="text-sm text-slate-300">
          conf {threshold.toFixed(2)}
          <input type="range" min={0.05} max={0.95} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="ml-2 align-middle" />
        </label>
        {data && <span className="text-sm text-slate-400">{data.total} images · predictions {data.predictions_available ? "cached" : "unavailable"}</span>}
      </div>
      {message && <div className="rounded bg-slate-800 px-3 py-2 text-sm">{message}</div>}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {(data?.items ?? []).map((item: TestImage) => (
          <div key={item.id} className="rounded-lg border border-slate-800 bg-slate-900/60 p-2">
            <div className="relative aspect-video overflow-hidden rounded bg-black">
              <img src={`/api/image/${encodeURIComponent(item.id)}?manifest=${testset}`} className="h-full w-full object-contain" alt={item.id} />
              <BoxOverlay width={item.width} height={item.height} detections={item.detections} groundTruth={item.ground_truth} />
            </div>
            <div className="mt-2 flex items-center justify-between text-xs">
              <span className={item.status.includes("false") ? "text-red-400" : "text-emerald-400"}>{item.status}</span>
              <span className="text-slate-500">{item.source}/{item.category}</span>
            </div>
            <div className="mt-1 flex gap-1 text-[11px]">
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => feedback(item, "false_positive")}>FP</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => feedback(item, "missed")}>FN</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => feedback(item, "correct")}>ok</button>
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <button className="rounded bg-slate-800 px-3 py-1 text-sm disabled:opacity-40" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          prev
        </button>
        <button className="rounded bg-slate-800 px-3 py-1 text-sm" onClick={() => setPage((p) => p + 1)}>
          next
        </button>
      </div>
    </div>
  );
}
