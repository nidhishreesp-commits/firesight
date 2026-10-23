import { Detection, GroundTruth } from "../types";

export function BoxOverlay({
  width,
  height,
  detections,
  groundTruth,
  gtPrefix = "GT ",
  gtBorder = "border-emerald-400",
  gtBadge = "bg-emerald-500"
}: {
  width: number;
  height: number;
  detections: Detection[];
  groundTruth: GroundTruth[];
  gtPrefix?: string;
  gtBorder?: string;
  gtBadge?: string;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {groundTruth.map((gt, i) => {
        const [x1, y1, x2, y2] = gt.box;
        return (
          <div
            key={`gt-${i}`}
            className={`absolute border-2 ${gtBorder}`}
            style={{
              left: `${(x1 / width) * 100}%`,
              top: `${(y1 / height) * 100}%`,
              width: `${((x2 - x1) / width) * 100}%`,
              height: `${((y2 - y1) / height) * 100}%`
            }}
          >
            <span className={`absolute -top-5 left-0 px-1 text-[10px] text-black ${gtBadge}`}>{gtPrefix}{gt.label}</span>
          </div>
        );
      })}
      {detections.map((d, i) => {
        const [x1, y1, x2, y2] = d.box;
        return (
          <div
            key={`d-${i}`}
            className="absolute border-2 border-red-500"
            style={{
              left: `${(x1 / width) * 100}%`,
              top: `${(y1 / height) * 100}%`,
              width: `${((x2 - x1) / width) * 100}%`,
              height: `${((y2 - y1) / height) * 100}%`
            }}
          >
            <span className="absolute -top-5 left-0 bg-red-500 px-1 text-[10px] text-white">
              {d.label} {d.confidence.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
