import { Detection, GroundTruth } from "../types";

export function BoxOverlay({
  width,
  height,
  detections,
  groundTruth
}: {
  width: number;
  height: number;
  detections: Detection[];
  groundTruth: GroundTruth[];
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {groundTruth.map((gt, i) => {
        const [x1, y1, x2, y2] = gt.box;
        return (
          <div
            key={`gt-${i}`}
            className="absolute border-2 border-emerald-400"
            style={{
              left: `${(x1 / width) * 100}%`,
              top: `${(y1 / height) * 100}%`,
              width: `${((x2 - x1) / width) * 100}%`,
              height: `${((y2 - y1) / height) * 100}%`
            }}
          >
            <span className="absolute -top-5 left-0 bg-emerald-500 px-1 text-[10px] text-black">GT {gt.label}</span>
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
