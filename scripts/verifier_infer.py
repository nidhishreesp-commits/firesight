"""P50 detections on the out-of-sample verifier sample (conf 0.10)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="build/verifier_sample.json")
    ap.add_argument("--out", default="build/verifier/detections.jsonl")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    rows = json.loads((PROJECT_ROOT/args.sample).read_text(encoding="utf-8"))
    wh = {}
    for line in (PROJECT_ROOT/"data/splits/train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); wh[r["id"]] = (r.get("width"), r.get("height"))
    out = PROJECT_ROOT/args.out; out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.parent/"paths.txt"
    lst.write_text("\n".join(str(PROJECT_ROOT/r["path"]) for r in rows)+"\n", encoding="utf-8")
    p2i = {str((PROJECT_ROOT/r["path"]).resolve()): r["id"] for r in rows}
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT/"build/kaggle_weights/runs/P50/weights/best.onnx"), task="detect")
    t0 = time.time(); n = 0
    with out.open("w", encoding="utf-8") as fh:
        for res in model.predict(str(lst), imgsz=640, conf=args.conf, batch=args.batch, device="cpu",
                                 verbose=False, stream=True):
            key = p2i.get(str(Path(res.path).resolve()), Path(res.path).stem)
            w, h = wh.get(key, (0, 0))
            if not w or not h:
                row = next((r for r in rows if r["id"] == key), None)
                if row and row.get("w"): w, h = row["w"], row["h"]
            dets = [{"label": "fire" if int(b.cls.item())==0 else "smoke",
                     "conf": round(float(b.conf.item()), 6),
                     "box": [float(v) for v in b.xyxy[0].tolist()]} for b in res.boxes]
            fh.write(json.dumps({"id": key, "w": w, "h": h, "detections": dets})+"\n")
            n += 1
            if n % 500 == 0:
                print(n, "/", len(rows), round(n/max(time.time()-t0,1e-6),2), "img/s", flush=True)
    print("done", n, "in", round(time.time()-t0,1), "s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
