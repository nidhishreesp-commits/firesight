"""Phase 0: collect and describe the actual false positives.

Sources:
  * val.jsonl report slice (model-selection data) - FPs from the saved P50 predictions
  * bench_R DEV (300, allowed for error analysis) - FPs from a fresh P50 inference
Output: build/fp_taxonomy/fps.csv (set,id,path,caption) + counts.
Captions use Florence-2 (cached locally); no network.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from firesight.config import PROJECT_ROOT
from run_bench import calibrate_image_level, decision, load_val
from run_benchR import load_benchR

P50_ONNX = PROJECT_ROOT / "build/kaggle_weights/runs/P50/weights/best.onnx"
VAL_PREDS = PROJECT_ROOT / "build/pilot_eval/preds_P50.jsonl"
OUT = PROJECT_ROOT / "build/fp_taxonomy"


def val_fps() -> list[dict]:
    from pilot_eval import load_list, load_records, read_preds
    dataset = PROJECT_ROOT / "build/kaggle_dataset"
    report = load_records(load_list(dataset / "lists" / "val_report.txt", dataset), dataset)
    preds = read_preds(VAL_PREDS)
    thr = json.loads((PROJECT_ROOT / "build/pilot_eval/metrics_P50.json").read_text())["thresholds"]
    stem2path = {}
    with (dataset / "MANIFEST.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem2path[row["image_id"].replace(":", "_")] = row["path"]
    out = []
    for r in report:
        if (r.meta.get("has_fire") or r.meta.get("has_smoke")):
            continue
        if decision(r, preds.get(r.id, []), thr):
            out.append({"set": "val_report", "id": r.id, "path": str(PROJECT_ROOT / stem2path.get(r.id, r.path))})
    return out


def dev_fps(batch: int = 8) -> list[dict]:
    from ultralytics import YOLO
    from firesight.types import Box, Detection
    records, member = load_benchR()
    dev = [r for r in records if member[r.id] == "dev"]
    cal = load_val()
    OUT.mkdir(parents=True, exist_ok=True)

    def predict(recs, name):
        lst = OUT / f"paths_{name}.txt"
        lst.write_text("\n".join(str(PROJECT_ROOT / r.path) for r in recs) + "\n", encoding="utf-8")
        path_to_id = {str((PROJECT_ROOT / r.path).resolve()): r.id for r in recs}
        model = YOLO(str(P50_ONNX), task="detect")
        preds = {}
        for res in model.predict(str(lst), imgsz=640, conf=0.01, batch=batch, device="cpu",
                                 verbose=False, stream=True):
            preds[path_to_id.get(str(Path(res.path).resolve()), Path(res.path).stem)] = [
                Detection("fire" if int(b.cls.item()) == 0 else "smoke", float(b.conf.item()),
                          Box(*[float(v) for v in b.xyxy[0].tolist()])) for b in res.boxes]
        return preds

    thr = calibrate_image_level(cal, predict(cal, "cal"), target_fpr=0.02)
    preds = predict(dev, "dev")
    out = []
    for r in dev:
        if (r.meta.get("has_fire") or r.meta.get("has_smoke")):
            continue
        if decision(r, preds.get(r.id, []), thr):
            out.append({"set": "bench_R_dev", "id": r.id, "path": str(PROJECT_ROOT / r.path)})
    return out


def caption(paths: list[dict], limit: int = 0) -> None:
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    mid = "microsoft/Florence-2-base"
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    task = "<MORE_DETAILED_CAPTION>"
    for i, row in enumerate(paths):
        if limit and i >= limit:
            row["caption"] = ""
            continue
        try:
            img = Image.open(row["path"]).convert("RGB")
            inputs = proc(text=task, images=img, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                     max_new_tokens=96, num_beams=2)
            text = proc.batch_decode(ids, skip_special_tokens=False)[0]
            row["caption"] = proc.post_process_generation(text, task=task, image_size=img.size)[task].strip()
        except Exception as exc:
            row["caption"] = f"CAPTION_ERROR: {type(exc).__name__}: {exc}"
        if (i + 1) % 10 == 0:
            print("captioned", i + 1, "/", len(paths), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = val_fps() + dev_fps()
    print("FPs:", len(rows), "val:", sum(1 for r in rows if r["set"] == "val_report"),
          "dev:", sum(1 for r in rows if r["set"] == "bench_R_dev"), flush=True)
    if args.caption:
        caption(rows, args.limit)
    with (OUT / "fps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "id", "path", "caption"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "caption": r.get("caption", "")})
    print("saved", OUT / "fps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
