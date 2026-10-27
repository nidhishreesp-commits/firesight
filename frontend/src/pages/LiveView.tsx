import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { CameraInfo } from "../types";

export default function LiveView() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [alarm, setAlarm] = useState<string | null>(null);
  const [detections, setDetections] = useState<Record<string, any[]>>({});
  const [policy, setPolicy] = useState<{ include: string[]; exclude: string[]; warning?: string[] }>({ include: [], exclude: [], warning: [] });
  const sockets = useRef<Record<string, WebSocket>>({});
  const FIRE_TYPES = ["uncontrolled", "controlled", "spark_benign", "electrical_arc", "screen", "unknown"];
  const TIERS = ["alarm", "warning", "log"] as const;

  const tierOf = (ftype: string): string => {
    if (policy.exclude.includes(ftype)) return "log";
    if (policy.include.includes(ftype)) return "alarm";
    if ((policy.warning ?? []).includes(ftype)) return "warning";
    return "log";
  };

  const refresh = () => api.cameras().then((r) => setCameras(r.cameras)).catch(() => undefined);

  useEffect(() => {
    refresh();
    api.firePolicy().then(setPolicy).catch(() => undefined);
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, []);

  const setTier = async (ftype: string, tier: string) => {
    const include = policy.include.filter((x) => x !== ftype);
    const exclude = policy.exclude.filter((x) => x !== ftype);
    const warning = (policy.warning ?? []).filter((x) => x !== ftype);
    if (tier === "alarm") include.push(ftype);
    else if (tier === "warning") warning.push(ftype);
    else exclude.push(ftype);
    const next = { include, exclude, warning };
    setPolicy(next);
    await api.setFirePolicy(include, exclude, warning);
  };

  const connect = (id: string) => {
    if (sockets.current[id]) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/live/${id}/ws`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetections((prev) => ({ ...prev, [id]: data.detections }));
      if (data.alarm?.active) {
        setAlarm(`ALARM ${id}: ${(data.alarm.labels ?? []).join(", ")} (level ${data.alarm.level ?? 1})`);
      }
    };
    sockets.current[id] = ws;
  };

  useEffect(() => () => Object.values(sockets.current).forEach((ws) => ws.close()), []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Live view</h1>
        <button className="rounded bg-slate-800 px-3 py-1 text-sm" onClick={refresh}>
          Refresh
        </button>
      </div>
      <div className="rounded border border-slate-800 bg-slate-900/60 p-3 text-sm">
        <div className="mb-1 font-medium">Alarm fire-type policy</div>
        <div className="mb-2 text-xs text-slate-400">
          Each flame type is <code>alarm</code> (page/notify), <code>warning</code> (surface only, never
          alarms) or <code>log</code> (record only). Applied at runtime; no retraining. Until the
          flame-type classifier is trained, detections are <code>unknown</code>.
        </div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
          {FIRE_TYPES.map((ftype) => (
            <label key={ftype} className="flex items-center justify-between gap-2 text-xs">
              <span>{ftype}</span>
              <select
                className="rounded bg-slate-800 px-1 py-0.5 text-xs"
                value={tierOf(ftype)}
                onChange={(e) => setTier(ftype, e.target.value)}
              >
                {TIERS.map((tier) => (
                  <option key={tier} value={tier}>{tier}</option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </div>
      {alarm && (
        <div className="flex items-center justify-between rounded border border-red-500 bg-red-950/60 px-4 py-2 text-red-200">
          <span>{alarm}</span>
          <button className="text-xs underline" onClick={() => setAlarm(null)}>
            dismiss
          </button>
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {cameras.map((cam) => (
          <div key={cam.id} className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <div className="font-medium">{cam.name}</div>
                <div className="text-xs text-slate-400">
                  {cam.id} · {cam.uri} · {cam.status} · {cam.fps.toFixed(1)} fps
                </div>
              </div>
              <div className="space-x-2">
                <button
                  className="rounded bg-ember-600 px-2 py-1 text-xs"
                  onClick={() => api.startCamera(cam.id).then(() => { connect(cam.id); refresh(); })}
                >
                  start
                </button>
                <button className="rounded bg-slate-800 px-2 py-1 text-xs" onClick={() => api.stopCamera(cam.id).then(refresh)}>
                  stop
                </button>
              </div>
            </div>
            <div className="relative aspect-video overflow-hidden rounded bg-black">
              {cam.status === "running" ? (
                <img src={`/api/live/${cam.id}/mjpeg`} className="h-full w-full object-contain" alt={cam.id} />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-slate-500">camera stopped</div>
              )}
              <div className="absolute bottom-1 left-1 max-h-24 overflow-auto rounded bg-black/60 p-1 text-[10px]">
                {(detections[cam.id] ?? []).map((d, i) => (
                  <div key={i} className={d.meta?.confirmed ? "text-red-400" : "text-amber-300"}>
                    {d.label} {Number(d.confidence).toFixed(2)} {d.track_id ? `#${d.track_id}` : ""}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
