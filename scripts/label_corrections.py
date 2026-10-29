"""Apply reviewer label corrections as a SIDE FILE and report metrics both ways.

Never edits bench_R, testbench100, TEST or val.jsonl. The side file is
label_corrections.csv with columns: id,original_label,corrected_label,reason,reviewer,date.

Labels: negative | fire | smoke | fire+smoke
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from firesight.config import PROJECT_ROOT
from run_bench import wilson
from pilot_eval import load_list, load_records, read_preds


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("id"):
            out[row["id"]] = (row.get("corrected_label") or "").strip().lower()
    return out


def label_of(rec, corrections):
    if rec.id in corrections:
        return corrections[rec.id]
    hf, hs = rec.meta.get("has_fire"), rec.meta.get("has_smoke")
    return "fire+smoke" if hf and hs else "fire" if hf else "smoke" if hs else "negative"


def metrics(records, preds, thr, corrections):
    tp = fp = tn = fn = 0
    for rec in records:
        lab = label_of(rec, corrections)
        positive = lab != "negative"
        dets = preds.get(rec.id, [])
        alarm = any(d.confidence >= thr.get(d.label, 1.0) for d in dets)
        if positive and alarm: tp += 1
        elif positive: fn += 1
        elif alarm: fp += 1
        else: tn += 1
    n = tp + fp + tn + fn
    return {"n": n, "positives": tp + fn, "negatives": tn + fp, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / n, 6) if n else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "recall_95ci": wilson(tp, tp + fn),
            "fpr": round(fp / (tn + fp), 6) if tn + fp else None,
            "fpr_95ci": wilson(fp, tn + fp)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", default="label_corrections.csv")
    ap.add_argument("--dataset", default="build/kaggle_dataset")
    ap.add_argument("--preds", default="build/pilot_eval/preds_P50.jsonl")
    ap.add_argument("--metrics", default="build/pilot_eval/metrics_P50.json")
    args = ap.parse_args()
    corr = load(PROJECT_ROOT / args.corrections)
    ds = PROJECT_ROOT / args.dataset
    records = load_records(load_list(ds / "lists" / "val_report.txt", ds), ds)
    preds = read_preds(PROJECT_ROOT / args.preds)
    thr = json.loads((PROJECT_ROOT / args.metrics).read_text(encoding="utf-8"))["thresholds"]
    orig = metrics(records, preds, thr, {})
    adj = metrics(records, preds, thr, corr)
    print("corrections loaded:", len(corr))
    for name, m in (("original", orig), ("corrected", adj)):
        print(f"{name:10s} n={m['n']} pos={m['positives']} neg={m['negatives']} "
              f"recall={m['recall']} {m['recall_95ci']} fpr={m['fpr']} {m['fpr_95ci']} acc={m['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
