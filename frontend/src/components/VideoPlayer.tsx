import { useEffect, useMemo, useRef, useState } from "react";

export function VideoPlayer({
  src,
  result,
  startTime = 0,
  onTime
}: {
  src: string | null;
  result: any;
  startTime?: number;
  onTime?: (t: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const initializedRef = useRef(false);
  const startRef = useRef(startTime);
  const [frameIndex, setFrameIndex] = useState(0);
  const [videoTime, setVideoTime] = useState(0);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const frames: any[] = result?.frames ?? [];
  const lastT = frames.length ? Number(frames[frames.length - 1].t) : 0;
  const beyondAnalysed = frames.length > 0 && videoTime > lastT + 0.5;

  const current = frames[frameIndex];
  const alarm = beyondAnalysed ? null : current?.alarm;

  useEffect(() => {
    if (!src || !videoRef.current) return;
    const v = videoRef.current;
    let raf = 0;
    let lastReport = -1;
    const tick = () => {
      if (frames.length && initializedRef.current) {
        const t = v.currentTime;
        let lo = 0;
        let hi = frames.length - 1;
        while (lo < hi) {
          const mid = Math.ceil((lo + hi) / 2);
          if ((frames[mid].t ?? 0) <= t) lo = mid;
          else hi = mid - 1;
        }
        setFrameIndex(lo);
        if (Math.abs(t - lastReport) > 0.25) {
          lastReport = t;
          setVideoTime(t);
          onTime?.(t);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [src, frames.length, onTime]);

  useEffect(() => {
    initializedRef.current = false;
    startRef.current = startTime;
    setFrameIndex(0);
  }, [src]);

  const boxes = useMemo(() => {
    if (!current || beyondAnalysed) return [];
    const w = dims.w || result?.video_width || 1;
    const h = dims.h || result?.video_height || 1;
    return (current.detections ?? []).map((d: any, i: number) => {
      const [x1, y1, x2, y2] = d.box;
      return {
        key: `${frameIndex}-${i}`,
        x: (x1 / w) * 100,
        y: (y1 / h) * 100,
        w: ((x2 - x1) / w) * 100,
        h: ((y2 - y1) / h) * 100,
        label: d.label,
        confidence: d.confidence,
        confirmed: Boolean(d.meta?.confirmed)
      };
    });
  }, [current, frameIndex, dims, result, beyondAnalysed]);

  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded border border-slate-800 bg-black text-sm text-slate-500">
        {result ? "Re-upload the clip to replay it. Saved stats/timeline are still shown below." : "Choose a clip to preview it here."}
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded border border-slate-800 bg-black"
         style={{ aspectRatio: dims.w && dims.h ? `${dims.w} / ${dims.h}` : "16 / 9" }}>
      <video
        ref={videoRef}
        src={src}
        controls
        playsInline
        className="block h-full w-full bg-black object-contain"
        onLoadedMetadata={(e) => {
          const v = e.currentTarget;
          setDims({ w: v.videoWidth, h: v.videoHeight });
          if (startRef.current > 0 && startRef.current < v.duration - 0.2) {
            v.currentTime = startRef.current;
          } else {
            initializedRef.current = true;
          }
        }}
        onSeeked={() => { initializedRef.current = true; }}
      />
      <div className="pointer-events-none absolute inset-0">
        {boxes.map((b: any) => (
          <div
            key={b.key}
            className={`absolute border-2 ${b.confirmed ? "border-red-500" : "border-amber-400"}`}
            style={{ left: `${b.x}%`, top: `${b.y}%`, width: `${b.w}%`, height: `${b.h}%` }}
          >
            <span className={`absolute -top-5 left-0 px-1 text-[10px] text-white ${b.confirmed ? "bg-red-600" : "bg-amber-600"}`}>
              {b.label} {Number(b.confidence).toFixed(2)}
            </span>
          </div>
        ))}
        {alarm?.active && (
          <div className="absolute left-2 top-2 rounded bg-red-600/90 px-3 py-1 text-sm font-semibold text-white">
            ALARM {alarm.labels?.join(", ")} · level {alarm.level ?? 1}
          </div>
        )}
        {beyondAnalysed ? (
          <div className="absolute inset-x-0 bottom-14 mx-auto w-fit rounded bg-amber-900/90 px-3 py-1 text-xs text-amber-100">
            No analysis beyond {lastT.toFixed(0)}s — increase “analyse first” and run again to see boxes here.
          </div>
        ) : current ? (
          <div className="absolute bottom-2 right-2 rounded bg-black/70 px-2 py-1 text-[11px] text-slate-200">
            video {videoTime.toFixed(1)}s · detections at {Number(current.t).toFixed(1)}s · {boxes.length} box{boxes.length === 1 ? "" : "es"}
          </div>
        ) : null}
      </div>
    </div>
  );
}
