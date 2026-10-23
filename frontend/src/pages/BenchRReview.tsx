import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { BoxOverlay } from "../components/BoxOverlay";

const CATEGORIES = [
  "uncontrolled_fire", "controlled_fire", "spark_screen", "smoke_only", "both",
  "sunset_orange_sky", "red_orange_neon_lights", "vehicle_lights_streetlights", "chandelier_stage_lights",
  "reflections_lens_flare_glare", "orange_ppe_vests_cones", "flowers_autumn_leaves", "clouds", "fog_haze_mist",
  "steam", "exhaust", "dust_construction_plaster", "snow", "overexposed_white", "camera_artifacts",
  "ordinary_indoor", "ordinary_streets_warehouses_offices", "ordinary_other"
];
const FLAME_TYPES = ["uncontrolled", "controlled", "spark", "screen", "unknown"];

export default function BenchRReview() {
  const [data, setData] = useState<any>(null);
  const [idx, setIdx] = useState(0);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("firesight.reviewer") ?? "");
  const [form, setForm] = useState<any>({});
  const [boxesFire, setBoxesFire] = useState<number[][]>([]);
  const [boxesSmoke, setBoxesSmoke] = useState<number[][]>([]);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [drawMode, setDrawMode] = useState<"fire" | "smoke" | null>(null);
  const [drag, setDrag] = useState<any>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [spotMode, setSpotMode] = useState(false);
  const [spot, setSpot] = useState<any>(null);
  const [spotIdx, setSpotIdx] = useState(0);
  const [proposals, setProposals] = useState<any>(null);
  const [minProposalScore, setMinProposalScore] = useState(0.02);
  const [proposed, setProposed] = useState(false);
  const [blind, setBlind] = useState(false);
  const startedAt = useRef<number>(Date.now());
  const canvasRef = useRef<HTMLDivElement | null>(null);

  const load = () => api.benchR().then(setData).catch((e) => setError(String(e)));
  useEffect(() => { load(); api.benchRProposals().then(setProposals).catch(() => undefined); }, []);
  useEffect(() => { localStorage.setItem("firesight.reviewer", reviewer); }, [reviewer]);

  const rows: any[] = data?.images ?? [];
  const row = rows[idx];
  useEffect(() => {
    if (!row) return;
    const prop = proposals?.images?.[row.id];
    const isBlind = Boolean(prop?.blind);
    const useProposal = row.review_status !== "reviewed" && Boolean(prop) && !isBlind;
    setProposed(useProposal);
    setBlind(isBlind);
    if (useProposal) {
      setForm({
        has_fire: prop.suggested_has_fire ?? 0, has_smoke: prop.suggested_has_smoke ?? 0,
        flame_type: prop.suggested_flame_type ?? "unknown", category: row.family,
        difficulty: row.difficulty ?? "medium", ambiguous: 0, notes: ""
      });
      const pf = (prop.proposals ?? []).filter((p: any) => p.label === "fire" && p.score >= minProposalScore).map((p: any) => p.box);
      const ps = (prop.proposals ?? []).filter((p: any) => p.label === "smoke" && p.score >= minProposalScore).map((p: any) => p.box);
      setBoxesFire(pf);
      setBoxesSmoke(ps);
    } else {
      setForm({
        has_fire: row.has_fire ?? 0, has_smoke: row.has_smoke ?? 0, flame_type: row.flame_type ?? "unknown",
        category: row.category ?? row.family, difficulty: row.difficulty ?? "medium",
        ambiguous: row.ambiguous ?? 0, notes: row.notes ?? ""
      });
      setBoxesFire(row.boxes_fire ?? []);
      setBoxesSmoke(row.boxes_smoke ?? []);
    }
    setDims({ w: row.width ?? 0, h: row.height ?? 0 });
    setDrawMode(null);
    setDrag(null);
    startedAt.current = Date.now();
  }, [idx, row?.id, proposals, minProposalScore]);

  const reviewed = useMemo(() => rows.filter((r) => r.review_status === "reviewed").length, [rows]);
  const comp = data?.composition ?? {};
  const spotInfo = data?.spot_check ?? {};
  const locked = data?.locked;

  const toImage = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !dims.w || !dims.h) return null;
    return { x: Math.max(0, Math.min(dims.w, ((e.clientX - rect.left) / rect.width) * dims.w)),
             y: Math.max(0, Math.min(dims.h, ((e.clientY - rect.top) / rect.height) * dims.h)) };
  };
  const onDown = (e: React.MouseEvent) => { if (!drawMode) return; const p = toImage(e); if (p) setDrag({ ...p, x2: p.x, y2: p.y }); };
  const onMove = (e: React.MouseEvent) => { if (!drag) return; const p = toImage(e); if (p) setDrag({ ...drag, x2: p.x, y2: p.y }); };
  const onUp = () => {
    if (!drag || !drawMode) { setDrag(null); return; }
    const box = [Math.min(drag.x, drag.x2), Math.min(drag.y, drag.y2), Math.max(drag.x, drag.x2), Math.max(drag.y, drag.y2)];
    if (box[2] - box[0] > 5 && box[3] - box[1] > 5) {
      if (drawMode === "fire") setBoxesFire([...boxesFire, box]); else setBoxesSmoke([...boxesSmoke, box]);
    }
    setDrag(null);
  };

  const save = async (advance = true) => {
    if (!row) return;
    if (!reviewer.trim()) { setError("Enter your name."); return; }
    setError("");
    const prop = proposals?.images?.[row.id];
    const pFire = (prop?.proposals ?? []).filter((p: any) => p.label === "fire" && p.score >= minProposalScore).map((p: any) => p.box);
    const pSmoke = (prop?.proposals ?? []).filter((p: any) => p.label === "smoke" && p.score >= minProposalScore).map((p: any) => p.box);
    const same = (a: number[][], b: number[][]) =>
      a.length === b.length && a.every((box, i) => b[i] && box.every((v, j) => Math.abs(v - b[i][j]) <= 2));
    const editFire = !same(boxesFire, pFire);
    const editSmoke = !same(boxesSmoke, pSmoke);
    const editLabels = Number(form.has_fire) !== Number(prop?.suggested_has_fire ?? 0)
      || Number(form.has_smoke) !== Number(prop?.suggested_has_smoke ?? 0)
      || (form.flame_type ?? "unknown") !== (prop?.suggested_flame_type ?? "unknown");
    const editsMade = Number(editFire) + Number(editSmoke) + Number(editLabels);
    try {
      await api.benchRReview({ id: row.id, ...form, reviewer, boxes_fire: boxesFire, boxes_smoke: boxesSmoke,
                               width: dims.w || row.width, height: dims.h || row.height,
                               accepted_as_proposed: !blind && !!prop && editsMade === 0,
                               edits_made: editsMade,
                               time_spent_s: Math.round((Date.now() - startedAt.current) / 1000),
                               blind });
      setMessage(`Saved ${reviewed + 1}/1000`);
      await load();
      if (advance && idx < rows.length - 1) setIdx(idx + 1);
    } catch (e) { setError(String(e)); }
  };

  const resume = () => {
    const pending = rows.findIndex((r) => r.review_status !== "reviewed");
    if (pending >= 0) setIdx(pending);
  };

  const startSpot = async () => {
    try {
      setSpot(await api.benchRSpotcheck());
      setSpotIdx(0);
      setSpotMode(true);
    } catch (e) { setError(String(e)); }
  };

  const saveSpot = async () => {
    if (!spot) return;
    const item = spot.items[spotIdx];
    try {
      const res = await api.benchRSpotSave({ id: item.id, has_fire: item.has_fire, has_smoke: item.has_smoke,
                                             flame_type: item.flame_type, reviewer });
      setMessage(`Spot check ${res.spot_reviewed}/100 · agreement ${(res.agreement_rate * 100).toFixed(1)}%`);
      if (spotIdx < spot.items.length - 1) setSpotIdx(spotIdx + 1);
      await load();
    } catch (e) { setError(String(e)); }
  };

  const lock = async () => {
    if (!reviewer.trim()) { setError("Enter your name."); return; }
    if (!confirm("Lock bench_R? Labels, manifest and DEV/SEALED split become immutable.")) return;
    try {
      const res = await api.benchRLock(reviewer);
      setMessage(`LOCKED. Manifest ${res.manifest_sha256}`);
      await load();
    } catch (e) { setError(String(e)); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.key === "ArrowRight") setIdx((i) => Math.min(i + 1, rows.length - 1));
      else if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, i - 1));
      else if (e.key.toLowerCase() === "f") setForm((f: any) => ({ ...f, has_fire: f.has_fire ? 0 : 1 }));
      else if (e.key.toLowerCase() === "s") setForm((f: any) => ({ ...f, has_smoke: f.has_smoke ? 0 : 1 }));
      else if (e.key.toLowerCase() === "n") setForm((f: any) => ({ ...f, has_fire: 0, has_smoke: 0, flame_type: null }));
      else if (e.key.toLowerCase() === "a") setForm((f: any) => ({ ...f, ambiguous: f.ambiguous ? 0 : 1 }));
      else if (e.key.toLowerCase() === "x") { if (boxesFire.length) setBoxesFire(boxesFire.slice(0, -1)); else setBoxesSmoke(boxesSmoke.slice(0, -1)); }
      else if (e.key === "Enter") save(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows.length, idx, form, boxesFire, boxesSmoke, dims, reviewer]);

  if (!data) return <div className="text-slate-400">{error || "Loading bench_R…"}</div>;
  if (spotMode && spot) {
    const item = spot.items[spotIdx];
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">bench_R second-pass spot check</h1>
          <button className="rounded bg-slate-800 px-3 py-1 text-sm" onClick={() => setSpotMode(false)}>back to first pass</button>
        </div>
        <div className="text-sm text-slate-400">Item {spotIdx + 1}/100 · first-pass labels are shown; confirm or correct.</div>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          <div className="xl:col-span-2">
            <div className="relative overflow-hidden rounded border border-slate-800 bg-black">
              <img src={`/api/benchR/image/${encodeURIComponent(item.id)}`} className="block max-h-[65vh] w-full object-contain" alt={item.id} />
              {item.width && item.height && (
                <BoxOverlay width={item.width} height={item.height} detections={[]}
                            groundTruth={[...(item.boxes_fire ?? []).map((b: number[]) => ({ label: "fire", box: b as [number, number, number, number] })),
                                          ...(item.boxes_smoke ?? []).map((b: number[]) => ({ label: "smoke", box: b as [number, number, number, number] }))]} />
              )}
            </div>
          </div>
          <div className="space-y-3 rounded border border-slate-800 bg-slate-900/60 p-3">
            <div className="text-sm">first pass: fire {item.has_fire} · smoke {item.has_smoke} · {item.flame_type ?? "n/a"}</div>
            <label className="flex gap-2 text-sm"><input type="checkbox" checked={!!item.has_fire} onChange={(e) => { const items = [...spot.items]; items[spotIdx] = { ...item, has_fire: e.target.checked ? 1 : 0 }; setSpot({ ...spot, items }); }} /> has_fire</label>
            <label className="flex gap-2 text-sm"><input type="checkbox" checked={!!item.has_smoke} onChange={(e) => { const items = [...spot.items]; items[spotIdx] = { ...item, has_smoke: e.target.checked ? 1 : 0 }; setSpot({ ...spot, items }); }} /> has_smoke</label>
            <select className="w-full rounded bg-slate-800 px-2 py-1 text-sm" value={item.flame_type ?? "unknown"}
                    onChange={(e) => { const items = [...spot.items]; items[spotIdx] = { ...item, flame_type: e.target.value }; setSpot({ ...spot, items }); }}>
              {FLAME_TYPES.map((t) => <option key={t}>{t}</option>)}
            </select>
            <button className="w-full rounded bg-ember-600 px-3 py-2 text-sm" onClick={saveSpot}>confirm & next</button>
            {message && <div className="text-xs text-slate-400">{message}</div>}
            {error && <div className="text-xs text-red-400">{error}</div>}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Reference bench — bench_R</h1>
          <div className="text-sm text-slate-400">
            {reviewed}/1000 reviewed · fire-only total {comp.fire_only}/320 · controlled {comp.fire_only_types?.controlled ?? 0}/80 ·
            sparks {((comp.fire_only_types?.spark ?? 0) + (comp.fire_only_types?.screen ?? 0))}/40 ·
            smoke-only {comp.smoke_only}/100 · both {comp.both}/80 · negatives {comp.negative}/500
          </div>
        </div>
        <div className="flex items-center gap-2">
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="reviewer name" className="rounded bg-slate-800 px-2 py-1 text-sm" />
          <button className="rounded bg-slate-800 px-3 py-1 text-sm" onClick={resume}>resume</button>
          {!locked && <button className="rounded bg-slate-800 px-3 py-1 text-sm" onClick={startSpot}>spot check {spotInfo.reviewed ?? 0}/100</button>}
          {locked ? <a className="rounded bg-slate-700 px-3 py-1 text-sm" href="/api/benchR/export">labels.csv</a>
                  : <button className="rounded bg-ember-600 px-3 py-1 text-sm" onClick={lock}>lock bench_R</button>}
        </div>
      </div>
      {locked && <div className="rounded border border-emerald-700 bg-emerald-950/50 px-3 py-2 text-sm">LOCKED · manifest {data.lock.manifest_sha256}</div>}
      {message && <div className="rounded bg-slate-800 px-3 py-2 text-sm">{message}</div>}
      {error && <div className="rounded bg-red-950/60 px-3 py-2 text-sm text-red-200">{error}</div>}
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <span>Keys: Arrows navigate, F fire, S smoke, N neither, A ambiguous, X delete last box, Enter save & next</span>
        {proposed && <span className="rounded bg-amber-900/60 px-2 py-1 text-amber-200">PROPOSED boxes/labels — approve, edit or delete</span>}
        {blind && <span className="rounded bg-blue-900/60 px-2 py-1 text-blue-200">BLIND REVIEW — proposals hidden</span>}
        <label className="flex items-center gap-2">proposal score &gt;=
          <input type="range" min={0} max={0.5} step={0.01} value={minProposalScore} onChange={(e) => setMinProposalScore(Number(e.target.value))} />
          {minProposalScore.toFixed(2)}
        </label>
        <button className="rounded bg-slate-800 px-2 py-1" onClick={() => {
          const p = proposals?.images ?? {};
          const next = rows.findIndex((r, i) => i > idx && !(p[r.id]?.proposals?.length));
          if (next >= 0) setIdx(next);
        }}>next no-proposal</button>
        <span>no-proposal images: {Object.values(proposals?.images ?? {}).filter((p:any) => !(p.proposals ?? []).length).length}</span>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div className="flex justify-center overflow-hidden rounded border border-slate-800 bg-black">
            <div className="relative inline-block max-w-full">
              <img src={`/api/benchR/image/${encodeURIComponent(row.id)}`} className="block max-h-[65vh] max-w-full" alt={row.id}
                   onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
              {dims.w > 0 && dims.h > 0 && (
                <BoxOverlay width={dims.w} height={dims.h} detections={[]}
                            groundTruth={[...boxesFire.map((b) => ({ label: "fire", box: b as [number, number, number, number] })),
                                          ...boxesSmoke.map((b) => ({ label: "smoke", box: b as [number, number, number, number] }))]} />
              )}
              <div ref={canvasRef} className={`absolute inset-0 ${drawMode ? "cursor-crosshair" : ""}`}
                   onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
                {drag && dims.w > 0 && <div className="absolute border-2 border-dashed border-white" style={{
                  left: `${(Math.min(drag.x, drag.x2) / dims.w) * 100}%`, top: `${(Math.min(drag.y, drag.y2) / dims.h) * 100}%`,
                  width: `${(Math.abs(drag.x2 - drag.x) / dims.w) * 100}%`, height: `${(Math.abs(drag.y2 - drag.y) / dims.h) * 100}%` }} />}
              </div>
            </div>
          </div>
          <div className="mt-2 text-xs text-slate-400">id {row.id} · source {row.source} · licence {row.license}</div>
        </div>
        <div className="space-y-2 rounded border border-slate-800 bg-slate-900/60 p-3">
          <div className="text-sm">Image {idx + 1}/1000 · {row.review_status}</div>
          <label className="flex gap-2 text-sm"><input type="checkbox" checked={!!form.has_fire} onChange={(e) => setForm({ ...form, has_fire: e.target.checked ? 1 : 0 })} /> has_fire</label>
          <label className="flex gap-2 text-sm"><input type="checkbox" checked={!!form.has_smoke} onChange={(e) => setForm({ ...form, has_smoke: e.target.checked ? 1 : 0 })} /> has_smoke</label>
          <label className="block text-sm">flame_type
            <select value={form.flame_type ?? "unknown"} onChange={(e) => setForm({ ...form, flame_type: e.target.value })} className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
              {FLAME_TYPES.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>
          <label className="block text-sm">category
            <select value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })} className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
              {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
            </select>
          </label>
          <label className="block text-sm">difficulty
            <select value={form.difficulty} onChange={(e) => setForm({ ...form, difficulty: e.target.value })} className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
              <option>easy</option><option>medium</option><option>hard</option>
            </select>
          </label>
          <label className="flex gap-2 text-sm"><input type="checkbox" checked={!!form.ambiguous} onChange={(e) => setForm({ ...form, ambiguous: e.target.checked ? 1 : 0 })} /> ambiguous</label>
          <textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} placeholder="notes" className="h-16 w-full rounded bg-slate-800 px-2 py-1 text-sm" />
          <div className="rounded border border-slate-700 p-2 text-xs">
            <div className="mb-1">Boxes: drag on image</div>
            <div className="flex flex-wrap gap-1">
              <button className={`rounded px-2 py-0.5 ${drawMode === "fire" ? "bg-red-600" : "bg-slate-800"}`} onClick={() => setDrawMode(drawMode === "fire" ? null : "fire")}>draw fire</button>
              <button className={`rounded px-2 py-0.5 ${drawMode === "smoke" ? "bg-emerald-600" : "bg-slate-800"}`} onClick={() => setDrawMode(drawMode === "smoke" ? null : "smoke")}>draw smoke</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => setBoxesFire(boxesFire.slice(0, -1))}>undo fire</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => setBoxesSmoke(boxesSmoke.slice(0, -1))}>undo smoke</button>
            </div>
            <div className="mt-1 text-slate-400">fire {boxesFire.length} · smoke {boxesSmoke.length}</div>
            {form.has_fire && !boxesFire.length && <div className="text-amber-400">has_fire without a fire box</div>}
            {form.has_smoke && !boxesSmoke.length && <div className="text-amber-400">has_smoke without a smoke box</div>}
          </div>
          <button className="w-full rounded bg-ember-600 px-3 py-2 text-sm disabled:opacity-40" disabled={locked || !reviewer.trim()} onClick={() => save(true)}>save & next</button>
          <button className="w-full rounded bg-slate-800 px-3 py-2 text-sm disabled:opacity-40" disabled={locked || !reviewer.trim()} onClick={() => save(false)}>save only</button>
        </div>
      </div>
    </div>
  );
}
