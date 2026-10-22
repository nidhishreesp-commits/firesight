import { useState } from "react";
import { useVideoSession } from "../VideoSession";
import { VideoPlayer } from "../components/VideoPlayer";

export default function VideoTest() {
  const {
    file, videoUrl, result, error, busy, elapsed, maxSeconds, sampleFps, lastTime, clipDuration,
    setMaxSeconds, setSampleFps, setLastTime, selectFile, run, clear
  } = useVideoSession();
  const [showAllFrames, setShowAllFrames] = useState(false);

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "firesight_video_result.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const rows = (result?.frames ?? []).filter((f: any) => showAllFrames || f.detections.length || f.alarm?.active);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold">Video test</h1>
        {result && (
          <div className="flex gap-2">
            <button className="rounded bg-slate-800 px-3 py-1 text-xs" onClick={downloadJson}>download JSON</button>
            <button className="rounded bg-slate-800 px-3 py-1 text-xs" onClick={clear}>clear</button>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm text-slate-300">
          clip
          <input type="file" accept="video/*" onChange={(e) => selectFile(e.target.files?.[0] ?? null)} className="mt-1 block text-sm" />
        </label>
        <label className="text-sm text-slate-300">
          analyse first (s)
          <span className="flex items-center gap-1">
            <input type="number" min={5} max={600} value={maxSeconds} onChange={(e) => setMaxSeconds(Number(e.target.value))}
                   className="mt-1 block w-24 rounded bg-slate-800 px-2 py-1" />
            {clipDuration > 0 && (
              <button className="mt-1 rounded bg-slate-800 px-2 py-1 text-[11px]" onClick={() => setMaxSeconds(Math.min(clipDuration, 600))}>
                full {clipDuration}s
              </button>
            )}
          </span>
        </label>
        <label className="text-sm text-slate-300">
          sample fps
          <input type="number" min={1} max={15} value={sampleFps} onChange={(e) => setSampleFps(Number(e.target.value))}
                 className="mt-1 block w-20 rounded bg-slate-800 px-2 py-1" />
        </label>
        <button className="rounded bg-ember-600 px-3 py-1 text-sm disabled:opacity-40" disabled={!file || busy} onClick={run}>
          {busy ? `processing… ${elapsed}s` : "run detection"}
        </button>
      </div>

      <p className="text-xs text-slate-500">
        Boxes are drawn live on the video while it plays or when you seek. Detections are sampled at {sampleFps} fps,
        so boxes update {sampleFps}×/second. If you play beyond the analysed portion, boxes are hidden and a warning appears.
        A full 77 s clip at 5 fps takes ~40 s on this CPU. For maximum recall on fast-growing fireballs, also try sample fps 10.
      </p>

      {error && <div className="rounded bg-red-950/60 px-3 py-2 text-sm text-red-200">{error}</div>}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <VideoPlayer src={videoUrl} result={result} startTime={lastTime} onTime={setLastTime} />
          {result && (
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-400">
              <span className="rounded bg-slate-800 px-2 py-0.5"><span className="mr-1 inline-block h-2 w-2 bg-red-500" />confirmed</span>
              <span className="rounded bg-slate-800 px-2 py-0.5"><span className="mr-1 inline-block h-2 w-2 bg-amber-400" />candidate (not yet N-of-M confirmed)</span>
              <span>last position {lastTime.toFixed(1)}s</span>
            </div>
          )}
        </div>
        <div className="space-y-2">
          {result ? (
            <>
              <Stat label="duration" value={`${result.duration_s.toFixed(1)} s`} />
              <Stat label="analysed" value={`${Number(result.analysed_seconds ?? Math.min(result.duration_s, maxSeconds)).toFixed(1)} s`} />
              <Stat label="frames analysed" value={result.frames_processed} />
              <Stat label="inference FPS" value={result.inference_fps.toFixed(2)} />
              <Stat label="first alarm" value={result.first_alarm_s != null ? `${result.first_alarm_s.toFixed(1)} s` : "none"} />
              <Stat label="detections / confirmed" value={`${result.detections} / ${result.confirmed_detections}`} />
              <Stat label="alarm events" value={result.alarm_events.length} />
              <Stat label="first confirmed" value={JSON.stringify(result.first_confirmed_s)} />
            </>
          ) : (
            <div className="rounded border border-slate-800 bg-slate-900/60 p-3 text-sm text-slate-400">
              No result yet. Choose a clip and run detection. Results survive tab switches; use “clear” to reset.
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-3">
          {result.duration_s > maxSeconds && (
            <div className="rounded bg-amber-950/60 px-3 py-2 text-sm text-amber-200">
              Only the first {maxSeconds}s of this {result.duration_s.toFixed(0)}s clip were analysed.
            </div>
          )}
          {result.detections === 0 && (
            <div className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-300">
              No fire/smoke detections in the processed portion. That is a valid result for a non-fire clip.
            </div>
          )}
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-sm text-slate-300">Timeline (red = alarm active, amber = confirmed detection)</span>
              <label className="text-xs text-slate-400">
                <input type="checkbox" className="mr-1" checked={showAllFrames} onChange={(e) => setShowAllFrames(e.target.checked)} />
                show every sampled frame
              </label>
            </div>
            <div className="flex h-8 w-full overflow-hidden rounded border border-slate-800">
              {(result.frames ?? []).map((f: any, i: number) => (
                <div
                  key={i}
                  title={`${f.t}s`}
                  className={f.alarm?.active ? "flex-1 bg-red-500" : f.confirmed?.length ? "flex-1 bg-amber-500" : "flex-1 bg-slate-800"}
                />
              ))}
            </div>
          </div>
          <div className="max-h-96 overflow-auto rounded border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-slate-800/90">
                <tr>
                  <th className="px-2 py-1">t (s)</th>
                  <th className="px-2 py-1">detections</th>
                  <th className="px-2 py-1">confirmed</th>
                  <th className="px-2 py-1">alarm</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 2000).map((f: any, i: number) => (
                  <tr key={i} className="border-t border-slate-800">
                    <td className="px-2 py-1">{f.t}</td>
                    <td className="px-2 py-1">{f.detections.map((d: any) => `${d.label}:${Number(d.confidence).toFixed(2)}`).join(", ")}</td>
                    <td className="px-2 py-1">{f.confirmed.length}</td>
                    <td className="px-2 py-1">{f.alarm?.active ? `level ${f.alarm.level}` : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900/60 p-2">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="font-medium">{String(value)}</div>
    </div>
  );
}
