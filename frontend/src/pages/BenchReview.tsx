import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { BoxOverlay } from "../components/BoxOverlay";

const CATEGORIES = [
  "fire", "smoke", "both",
  "sunset_sunrise", "street_head_lights", "lamps_neon_brake", "chandelier_festival",
  "reflection_glare", "orange_clothing_ppe", "cones_machinery", "flowers_leaves",
  "fog_mist_haze", "clouds", "steam_cooking", "exhaust_chimney", "dust_snow", "overexposed_white",
  "candle", "match_lighter", "gas_stove", "campfire", "welding", "fireworks", "fireplace", "screen_fire",
  "blur_noise_compression", "rain_on_lens", "infrared_night", "other_hard"
];

export default function BenchReview() {
  const [data, setData] = useState<any>(null);
  const [idx, setIdx] = useState(0);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("firesight.reviewer") ?? "");
  const [form, setForm] = useState<any>({});
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [locking, setLocking] = useState(false);
  const [boxesFire, setBoxesFire] = useState<number[][]>([]);
  const [boxesSmoke, setBoxesSmoke] = useState<number[][]>([]);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [drawMode, setDrawMode] = useState<"fire" | "smoke" | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number; x2: number; y2: number } | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  const load = () => api.bench100().then(setData).catch((e) => setError(String(e)));
  useEffect(() => { load(); }, []);
  useEffect(() => { localStorage.setItem("firesight.reviewer", reviewer); }, [reviewer]);

  const rows: any[] = data?.images ?? [];
  const row = rows[idx];
  useEffect(() => {
    if (!row) return;
    setForm({
      has_fire: row.has_fire, has_smoke: row.has_smoke, category: row.category,
      difficulty: row.difficulty, ambiguous: row.ambiguous, notes: row.notes ?? ""
    });
    setBoxesFire(row.boxes_fire ?? []);
    setBoxesSmoke(row.boxes_smoke ?? []);
    setDims({ w: row.width ?? 0, h: row.height ?? 0 });
    setDrawMode(null);
    setDrag(null);
  }, [idx, row?.id]);

  const reviewed = useMemo(() => rows.filter((r) => r.review_status === "reviewed").length, [rows]);
  const missingBoxes = useMemo(() => rows.filter((r) =>
    !r.ambiguous && ((r.has_fire && !(r.boxes_fire ?? []).length) || (r.has_smoke && !(r.boxes_smoke ?? []).length))
  ), [rows]);
  const comp = data?.composition ?? {};

  const save = async (advance = true) => {
    if (!row) return;
    if (!reviewer.trim()) { setError("Enter your name before saving labels."); return; }
    setError("");
    try {
      await api.benchReview({ id: row.id, ...form, reviewer, boxes_fire: boxesFire, boxes_smoke: boxesSmoke,
                              width: dims.w || row.width, height: dims.h || row.height });
      setMessage(`Saved ${idx + 1}/${rows.length}`);
      await load();
      if (advance && idx < rows.length - 1) setIdx(idx + 1);
    } catch (e) {
      setError(String(e));
    }
  };

  const compositionMismatch = comp.fire_only !== 25 || comp.smoke_only !== 15 || comp.both !== 10 || comp.negative !== 50;

  const lock = async () => {
    if (!reviewer.trim()) { setError("Enter your name before locking."); return; }
    if (!confirm("Lock the bench? Labels become immutable and a SHA-256 manifest is written.")) return;
    setLocking(true); setError("");
    try {
      let res;
      try {
        res = await api.benchLock(reviewer, false);
      } catch (e) {
        const msg = String(e);
        if (!msg.includes("frozen composition") && !msg.includes("Lock as reviewed")) throw e;
        const ok = confirm(
          `Your reviewed labels no longer match the frozen 25/15/10/50 composition.\n\n` +
          `Current: fire-only ${comp.fire_only}, smoke-only ${comp.smoke_only}, both ${comp.both}, negative ${comp.negative}.\n\n` +
          `Lock anyway and record an explicit composition override?`
        );
        if (!ok) { setError("Lock cancelled. Edit labels to restore 25/15/10/50, or click lock again to override."); return; }
        res = await api.benchLock(reviewer, true);
      }
      setMessage(`LOCKED. Manifest SHA-256: ${res.manifest_sha256}${res.composition_override ? " (composition override recorded)" : ""}`);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setLocking(false);
    }
  };

  const toImage = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !dims.w || !dims.h) return null;
    return {
      x: Math.max(0, Math.min(dims.w, ((e.clientX - rect.left) / rect.width) * dims.w)),
      y: Math.max(0, Math.min(dims.h, ((e.clientY - rect.top) / rect.height) * dims.h))
    };
  };
  const onDown = (e: React.MouseEvent) => {
    if (!drawMode) return;
    const p = toImage(e);
    if (p) setDrag({ x: p.x, y: p.y, x2: p.x, y2: p.y });
  };
  const onMove = (e: React.MouseEvent) => {
    if (!drag) return;
    const p = toImage(e);
    if (p) setDrag({ ...drag, x2: p.x, y2: p.y });
  };
  const onUp = () => {
    if (!drag || !drawMode) { setDrag(null); return; }
    const box = [Math.min(drag.x, drag.x2), Math.min(drag.y, drag.y2), Math.max(drag.x, drag.x2), Math.max(drag.y, drag.y2)];
    if (box[2] - box[0] > 5 && box[3] - box[1] > 5) {
      if (drawMode === "fire") setBoxesFire([...boxesFire, box]);
      else setBoxesSmoke([...boxesSmoke, box]);
    }
    setDrag(null);
  };

  if (!data) return <div className="text-slate-400">{error || "Loading bench…"}</div>;
  if (rows.length !== 100) return <div className="text-red-400">Expected 100 images, found {rows.length}.</div>;

  const locked = data.locked;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Bench review — testbench100</h1>
          <div className="text-sm text-slate-400">
            {reviewed}/100 reviewed · fire-only {comp.fire_only}/25 · smoke-only {comp.smoke_only}/15 ·
            both {comp.both}/10 · negatives {comp.negative}/50
          </div>
        </div>
        <div className="flex items-center gap-2">
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="reviewer name"
                 className="rounded bg-slate-800 px-2 py-1 text-sm" />
          {locked ? (
            <a className="rounded bg-slate-700 px-3 py-1 text-sm" href="/api/bench100/export">download labels.csv</a>
          ) : (
            <button className="rounded bg-ember-600 px-3 py-1 text-sm disabled:opacity-40" disabled={locking}
                    onClick={lock}>
              {locking ? "locking…" : "lock bench"}
            </button>
          )}
        </div>
      </div>

      {locked && (
        <div className="rounded border border-emerald-700 bg-emerald-950/50 px-3 py-2 text-sm text-emerald-200">
          Locked by {data.lock.reviewer} · manifest {data.lock.manifest_sha256}
        </div>
      )}
      {message && <div className="rounded bg-slate-800 px-3 py-2 text-sm">{message}</div>}
      {!locked && missingBoxes.length > 0 && (
        <div className="rounded border border-red-800 bg-red-950/50 px-3 py-2 text-sm text-red-200">
          <div className="mb-1">{missingBoxes.length} reviewed positive image(s) have no boxes. Draw boxes or mark them ambiguous before locking.</div>
          <div className="flex flex-wrap gap-1">
            {missingBoxes.slice(0, 20).map((r) => (
              <button key={r.id} className="rounded bg-slate-800 px-2 py-0.5 text-[11px]"
                      onClick={() => setIdx(rows.findIndex((x) => x.id === r.id))}>
                {r.category}: {r.id.slice(-34)}
              </button>
            ))}
          </div>
        </div>
      )}
      {!locked && compositionMismatch && (
        <div className="rounded border border-amber-700 bg-amber-950/50 px-3 py-2 text-sm text-amber-200">
          Your review changed the frozen composition: fire-only {comp.fire_only}/25, smoke-only {comp.smoke_only}/15,
          both {comp.both}/10, negative {comp.negative}/50. You can edit labels to restore it, or lock as reviewed and
          the manifest will record an explicit override.
        </div>
      )}
      {error && <div className="rounded bg-red-950/60 px-3 py-2 text-sm text-red-200">{error}</div>}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div className="flex justify-center overflow-hidden rounded border border-slate-800 bg-black">
            <div className="relative inline-block max-w-full">
              <img src={`/api/bench100/image/${encodeURIComponent(row.id)}`} className="block max-h-[70vh] max-w-full" alt={row.id}
                   onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
              {dims.w > 0 && dims.h > 0 && (
                <BoxOverlay width={dims.w} height={dims.h}
                            detections={[]}
                            groundTruth={[
                              ...boxesFire.map((b) => ({ label: "fire", box: b as [number, number, number, number] })),
                              ...boxesSmoke.map((b) => ({ label: "smoke", box: b as [number, number, number, number] }))
                            ]} />
              )}
              <div ref={canvasRef}
                   className={`absolute inset-0 ${drawMode ? "cursor-crosshair" : ""}`}
                   onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
                {drag && dims.w > 0 && (
                  <div className="absolute border-2 border-dashed border-white"
                       style={{
                         left: `${(Math.min(drag.x, drag.x2) / dims.w) * 100}%`,
                         top: `${(Math.min(drag.y, drag.y2) / dims.h) * 100}%`,
                         width: `${(Math.abs(drag.x2 - drag.x) / dims.w) * 100}%`,
                         height: `${(Math.abs(drag.y2 - drag.y) / dims.h) * 100}%`
                       }} />
                )}
              </div>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-1 gap-1 text-xs text-slate-400 md:grid-cols-2">
            <div>id: <span className="text-slate-200">{row.id}</span></div>
            <div>source: <span className="text-slate-200">{row.source}</span></div>
            <div>licence: <span className="text-slate-200">{row.license}</span></div>
            <div>sha256: <span className="text-slate-200">{String(row.sha256).slice(0, 20)}…</span></div>
            <div className="md:col-span-2">origin: {row.origin}</div>
            {row.attribution && <div className="md:col-span-2">attribution: {JSON.stringify(row.attribution)}</div>}
            <div className="md:col-span-2">flags: {JSON.stringify(row.flags)}</div>
          </div>
          <div className="mt-2 flex gap-2">
            <button className="rounded bg-slate-800 px-3 py-1 text-sm" disabled={idx === 0} onClick={() => setIdx(idx - 1)}>previous</button>
            <button className="rounded bg-slate-800 px-3 py-1 text-sm" disabled={idx >= rows.length - 1} onClick={() => setIdx(idx + 1)}>next</button>
            <span className="self-center text-xs text-slate-500">
              {row.kind === "positive" ? "provisional positive (dataset boxes in red/green)" : "provisional hard negative (no boxes)"}
            </span>
          </div>
        </div>

        <div className="space-y-3 rounded border border-slate-800 bg-slate-900/60 p-3">
          <div className="flex items-center justify-between text-sm">
            <span>Image {idx + 1} / 100</span>
            <span className={row.review_status === "reviewed" ? "text-emerald-400" : "text-amber-400"}>{row.review_status}</span>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={!!form.has_fire} onChange={(e) => setForm({ ...form, has_fire: e.target.checked ? 1 : 0 })} />
            has_fire
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={!!form.has_smoke} onChange={(e) => setForm({ ...form, has_smoke: e.target.checked ? 1 : 0 })} />
            has_smoke
          </label>
          <div className="rounded border border-slate-700 p-2 text-xs">
            <div className="mb-1 text-slate-300">Boxes (draw on image: drag a rectangle)</div>
            <div className="flex flex-wrap gap-1">
              <button className={`rounded px-2 py-0.5 ${drawMode === "fire" ? "bg-red-600" : "bg-slate-800"}`}
                      onClick={() => setDrawMode(drawMode === "fire" ? null : "fire")}>draw fire</button>
              <button className={`rounded px-2 py-0.5 ${drawMode === "smoke" ? "bg-emerald-600" : "bg-slate-800"}`}
                      onClick={() => setDrawMode(drawMode === "smoke" ? null : "smoke")}>draw smoke</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => setBoxesFire(boxesFire.slice(0, -1))}>undo fire</button>
              <button className="rounded bg-slate-800 px-2 py-0.5" onClick={() => setBoxesSmoke(boxesSmoke.slice(0, -1))}>undo smoke</button>
            </div>
            <div className="mt-1 text-slate-400">fire boxes: {boxesFire.length} · smoke boxes: {boxesSmoke.length}</div>
            {(form.has_fire && boxesFire.length === 0) && <div className="text-amber-400">has_fire is set but there is no fire box.</div>}
            {(form.has_smoke && boxesSmoke.length === 0) && <div className="text-amber-400">has_smoke is set but there is no smoke box.</div>}
          </div>
          <label className="block text-sm">
            category
            <select value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}
                    className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
              {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
            </select>
          </label>
          <label className="block text-sm">
            difficulty
            <select value={form.difficulty} onChange={(e) => setForm({ ...form, difficulty: e.target.value })}
                    className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
              <option>easy</option><option>medium</option><option>hard</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={!!form.ambiguous} onChange={(e) => setForm({ ...form, ambiguous: e.target.checked ? 1 : 0 })} />
            ambiguous / disputed label
          </label>
          <label className="block text-sm">
            notes
            <textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })}
                      className="mt-1 h-20 w-full rounded bg-slate-800 px-2 py-1 text-sm" />
          </label>
          <div className="text-xs text-slate-500">
            Saving records your name and UTC timestamp. The bench is immutable after locking.
          </div>
          <button className="w-full rounded bg-ember-600 px-3 py-2 text-sm disabled:opacity-40"
                  disabled={locked || !reviewer.trim()} onClick={() => save(true)}>
            {locked ? "locked" : "save & next"}
          </button>
          <button className="w-full rounded bg-slate-800 px-3 py-2 text-sm disabled:opacity-40"
                  disabled={locked || !reviewer.trim()} onClick={() => save(false)}>
            save only
          </button>
        </div>
      </div>
    </div>
  );
}
