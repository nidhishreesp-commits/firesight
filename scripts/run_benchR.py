from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from firesight.config import PROJECT_ROOT, load_config
from firesight.types import Box, GroundTruth, ImageRecord

from run_bench import (  # noqa: E402  (needs scripts/ on sys.path)
    MODELS, RUNS, by_flame_type, binary_metrics, box_ap, calibrate_image_level,
    contact_sheet, decision, draw_error, get_predictions, group_metrics, load_val,
    mean_iou, policy_metrics, threshold_sweep,
)

BENCHR_DIR = PROJECT_ROOT / "bench_R"
BENCHR_LOCK = BENCHR_DIR / "LOCKED.json"
OUT_DIR = RUNS / "benchR"
SPLITS = ("dev", "sealed", "all")


def load_benchR() -> tuple[list[ImageRecord], dict[str, str]]:
    """Load reviewed bench_R labels from the DEV/SEALED splits.

    index.json still carries the pre-review designed labels; DEV/SEALED carry the
    human-reviewed labels written at lock time, so those are the source of truth.
    """
    if not BENCHR_LOCK.exists():
        raise SystemExit("bench_R is not locked; refusing to benchmark")
    records: list[ImageRecord] = []
    member: dict[str, str] = {}
    for split_name, fname in (("dev", "DEV.jsonl"), ("sealed", "SEALED.jsonl")):
        path = BENCHR_DIR / fname
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            boxes_fire = r.get("boxes_fire") or []
            boxes_smoke = r.get("boxes_smoke") or []
            labels = [GroundTruth("fire", Box(*b)) for b in boxes_fire]
            labels += [GroundTruth("smoke", Box(*b)) for b in boxes_smoke]
            has_fire = int(r.get("has_fire") or 0)
            has_smoke = int(r.get("has_smoke") or 0)
            ftype = r.get("flame_type") or "unknown"
            category = r.get("category") or r.get("family") or "none"
            records.append(ImageRecord(
                id=r["id"], path=str(PROJECT_ROOT / r["file"]),
                width=int(r.get("width") or 0), height=int(r.get("height") or 0),
                source=r.get("source") or "unknown", scene=category, split="benchR",
                labels=labels, category=category,
                meta={
                    "family": r.get("family"),
                    "difficulty": r.get("difficulty") or "medium",
                    "has_fire": has_fire, "has_smoke": has_smoke,
                    "ambiguous": int(r.get("ambiguous") or 0),
                    "license": r.get("license") or "",
                    "flame_types": [ftype] if has_fire else [],
                    "bench_split": split_name,
                },
            ))
            member[r["id"]] = split_name
    return records, member


def pick(records: list[ImageRecord], member: dict[str, str], name: str) -> list[ImageRecord]:
    if name == "all":
        return records
    return [r for r in records if member[r.id] == name]


def _n(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "-"


def evaluate(name: str, records: list[ImageRecord], preds: dict, thresholds: dict) -> dict:
    return {
        "split": name,
        "n": len(records),
        "positives": sum(1 for r in records if r.meta.get("has_fire") or r.meta.get("has_smoke")),
        "fire": binary_metrics(records, preds, thresholds, "fire"),
        "smoke": binary_metrics(records, preds, thresholds, "smoke"),
        "combined": binary_metrics(records, preds, thresholds, None),
        "box_ap": {"fire": box_ap(records, preds, "fire"), "smoke": box_ap(records, preds, "smoke")},
        "mean_iou": {"fire": mean_iou(records, preds, "fire", thresholds.get("fire", 1)),
                     "smoke": mean_iou(records, preds, "smoke", thresholds.get("smoke", 1))},
        "by_group": group_metrics(records, preds, thresholds),
        "by_flame_type": by_flame_type(records, preds, thresholds),
        "policy_uncontrolled_plus_smoke": policy_metrics(records, preds, thresholds),
        "threshold_sweep": threshold_sweep(records, preds),
    }


def write_galleries(model: str, records: list[ImageRecord], preds: dict, thresholds: dict,
                    run_dir: Path) -> list[dict]:
    errors = []
    fp_paths, fn_paths = [], []
    for rec in records:
        dets = [d for d in preds.get(rec.id, []) if d.confidence >= min(thresholds.values())]
        positive = bool(rec.meta.get("has_fire") or rec.meta.get("has_smoke"))
        if positive == decision(rec, dets, thresholds):
            continue
        kind = "false_positive" if not positive else "false_negative"
        path = run_dir / "errors" / kind / f"{rec.id.replace(':', '_')}.jpg"
        draw_error(rec, dets, thresholds, path)
        (fp_paths if kind == "false_positive" else fn_paths).append(path)
        errors.append({
            "kind": kind, "id": rec.id, "bench_split": rec.meta.get("bench_split"),
            "family": rec.meta.get("family"), "category": rec.category,
            "difficulty": rec.meta.get("difficulty"),
            "ground_truth": [g.to_dict() for g in rec.labels],
            "predictions": [d.to_dict() for d in dets],
            "files": [str(path.relative_to(PROJECT_ROOT))],
        })
    contact_sheet(fp_paths, run_dir / "false_positives.jpg", f"{model} false positives")
    contact_sheet(fn_paths, run_dir / "false_negatives.jpg", f"{model} false negatives")
    return errors


def run_one(model_id: str, records, member, val, cfg, args, lock_sha: str) -> dict:
    val_preds = get_predictions(model_id, val, "valall", cfg, overwrite=args.overwrite_cache)
    bench_preds = get_predictions(model_id, records, "benchR", cfg, overwrite=args.overwrite_cache)
    thresholds = calibrate_image_level(val, val_preds, target_fpr=args.target_fpr)
    print(f"  {model_id}: thresholds={thresholds}", flush=True)

    result = {
        "bench": "bench_R",
        "model": model_id,
        "family": MODELS[model_id]["family"],
        "license": MODELS[model_id]["license"],
        "bench_locked_sha256": lock_sha,
        "bench_split_rule": "DEV 300 / SEALED 700, stratified random seed 20261022",
        "calibrated_on": "data/splits/val_mini.jsonl + data/splits/val_flames.jsonl",
        "target_fpr": args.target_fpr,
        "thresholds": thresholds,
        "splits": {},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": {
            "threshold_policy": "thresholds selected on val_mini + val_flames only; bench_R was not used for tuning",
            "sealed_warning": "SEALED metrics are reported for transparency; do not tune against them",
        },
    }
    for split_name in SPLITS:
        sub = pick(records, member, split_name)
        result["splits"][split_name] = evaluate(split_name, sub, bench_preds, thresholds)
        c = result["splits"][split_name]["combined"]
        fpr = c["fp"] / c["n"] if c["n"] else None
        print(f"  {model_id} [{split_name:6s}] n={len(sub):4d} acc={_n(c['accuracy'])} "
              f"recall={_n(c['recall'])} fpr={_n(fpr)}", flush=True)

    if args.galleries:
        run_dir = OUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{model_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
        result["errors"] = write_galleries(model_id, records, bench_preds, thresholds, run_dir)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + model_id
    out = OUT_DIR / f"{run_id}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  saved {out}", flush=True)
    return result


def summarize() -> int:
    import glob
    rows = []
    for p in sorted(glob.glob(str(OUT_DIR / "*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("bench") != "bench_R" or "splits" not in d:
            continue
        rows.append(d)
    if not rows:
        print("no bench_R runs found in", OUT_DIR)
        return 1
    rows.sort(key=lambda d: (d["generated_utc"], d["model"]))
    latest = {}
    for d in rows:
        latest[d["model"]] = d
    rows = sorted(latest.values(), key=lambda d: d["model"])

    def fmt(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, float) else "-"

    for split_name in SPLITS:
        print(f"\n### bench_R [{split_name}]")
        print(f"{'model':28s} {'n':>4s} {'acc':>7s} {'recall':>7s} {'spec':>7s} "
              f"{'fpr':>6s} {'prec':>7s} {'f1':>7s} {'ap50fire':>8s}")
        ordered = sorted(rows, key=lambda d: -(d["splits"][split_name]["combined"]["accuracy"] or 0))
        for d in ordered:
            s = d["splits"][split_name]
            c = s["combined"]
            fpr = c["fp"] / c["n"] if c["n"] else None
            ap = s["box_ap"]["fire"]["ap50"]
            print(f"{d['model']:28s} {s['n']:>4d} {fmt(c, 'accuracy'):>7s} {fmt(c, 'recall'):>7s} "
                  f"{fmt(c, 'specificity'):>7s} {(f'{fpr:.4f}' if fpr is not None else '-'):>6s} "
                  f"{fmt(c, 'precision'):>7s} {fmt(c, 'f1'):>7s} {(f'{ap:.4f}' if ap else '-'):>8s}")

    failed = {d["model"]: d.get("failure") for d in rows if d.get("failure")}
    if failed:
        print("\nfailures:", json.dumps(failed, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run models on the locked bench_R (DEV 300 / SEALED 700 reported separately). "
                    "Thresholds come from val_mini + val_flames only.")
    parser.add_argument("--models", default="all",
                        help="comma-separated model ids, or 'all' (default)")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (smoke test)")
    parser.add_argument("--galleries", action="store_true", help="write FP/FN error galleries")
    parser.add_argument("--summarize", action="store_true", help="only print the comparison table")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        return summarize()

    if not BENCHR_LOCK.exists():
        raise SystemExit("bench_R is not locked; refusing to benchmark")
    wanted = list(MODELS) if args.models.strip().lower() == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model ids: {unknown}")

    cfg = load_config(args.config)
    records, member = load_benchR()
    if args.limit:
        records = records[:args.limit]
        member = {r.id: member[r.id] for r in records}
    val = load_val()
    lock = json.loads(BENCHR_LOCK.read_text(encoding="utf-8"))
    print(f"bench_R={len(records)} dev={sum(1 for v in member.values() if v == 'dev')} "
          f"sealed={sum(1 for v in member.values() if v == 'sealed')} val={len(val)}", flush=True)

    failures = {}
    for model_id in wanted:
        print(f"\n=== {model_id} ===", flush=True)
        try:
            run_one(model_id, records, member, val, cfg, args, lock["manifest_sha256"])
        except Exception as exc:  # keep going so one broken model doesn't kill the batch
            failures[model_id] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED {model_id}: {failures[model_id]}", flush=True)

    if failures:
        (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print("\nfailures:", json.dumps(failures, indent=2))
    return 1 if failures and len(failures) == len(wanted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
