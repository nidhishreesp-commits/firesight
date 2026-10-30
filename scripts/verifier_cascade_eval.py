"""Evaluate the P50@0.10 + verifier v3 cascade on val_cal, val_report, bench_R DEV and the
5,301-image independent negative set.  All thresholds are chosen on val_cal only.

Outputs recall/FPR with Wilson CIs, McNemar vs P50 calibrated and vs P50 at matched FPR,
per-cause FPR on the independent set, and held-out-cause subsets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

P50_THR = {"fire": 0.29, "smoke": 0.33}
P50_CASCADE_CONF = 0.10


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["X"], [json.loads(m) for m in d["meta"]]


def load_samples(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    labels = {r["id"]: int(bool(r.get("labels"))) for r in rows}
    paths = {r["id"]: r["path"] for r in rows}
    return labels, paths


def per_image(meta, scores, p50_cal=True, conf_min=P50_CASCADE_CONF, verifier_v=None):
    """Return image->alarm for P50 calibrated, P50@0.10, or cascade."""
    out: dict[str, bool] = defaultdict(bool)
    for m, s in zip(meta, scores):
        conf = float(m["conf"])
        if p50_cal:
            if conf >= P50_THR.get(m["label"], 1.0):
                out[m["id"]] = True
        elif verifier_v is None:
            if conf >= conf_min:
                out[m["id"]] = True
        else:
            if conf >= conf_min and float(s) >= verifier_v:
                out[m["id"]] = True
    return out


def metrics(alarm: dict[str, bool], pos: set[str], neg: set[str], label: str) -> dict:
    tp = sum(1 for i in pos if alarm.get(i, False))
    fp = sum(1 for i in neg if alarm.get(i, False))
    return {"label": label, "tp": tp, "fp": fp, "n_pos": len(pos), "n_neg": len(neg),
            "recall": round(tp / len(pos), 6) if pos else None,
            "recall_95ci": wilson(tp, len(pos)) if pos else None,
            "fpr": round(fp / len(neg), 6) if neg else None,
            "fpr_95ci": wilson(fp, len(neg)) if neg else None}


def mcnemar(a: dict[str, bool], b: dict[str, bool], ids: set[str], label: str) -> dict:
    b01 = sum(1 for i in ids if a.get(i, False) and not b.get(i, False))
    b10 = sum(1 for i in ids if not a.get(i, False) and b.get(i, False))
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"set": label, "cascade_only": b01, "baseline_only": b10, "p": float(p)}


def calibrate_global_threshold(meta, scores, labels, target_fpr):
    """Single max-confidence threshold on val_cal at FPR<=target; max recall."""
    per = defaultdict(float)
    for m, s in zip(meta, scores):
        per[m["id"]] = max(per[m["id"]], float(m["conf"]))
    pos = {i for i, y in labels.items() if y}
    neg = {i for i, y in labels.items() if not y}
    best = None
    for thr in sorted(set(per.values()), reverse=True):
        tp = sum(1 for i in pos if per.get(i, 0) >= thr)
        fp = sum(1 for i in neg if per.get(i, 0) >= thr)
        fpr = fp / max(len(neg), 1)
        rec = tp / max(len(pos), 1)
        if fpr <= target_fpr and (best is None or rec > best[0]):
            best = (rec, fpr, float(thr))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="build/verifier/model_v3.joblib")
    ap.add_argument("--out", default="build/verifier/cascade_eval_v3.json")
    ap.add_argument("--verifier-v", type=float, default=0.002112214919179678, help="1.5%% point from val_cal")
    args = ap.parse_args()

    bundle = joblib.load(PROJECT_ROOT / args.verifier)
    model = bundle["model"]
    print("model", bundle.get("kind"), "cv", (bundle.get("cv") or {}).get(bundle.get("kind"), {}).get("auc_mean"))

    sets = {
        "val_cal": ("build/verifier/features_val_cal_v3.npz", "build/verifier/sample_val_cal.json"),
        "val_report": ("build/verifier/features_val_report.npz", "build/verifier/sample_val_report.json"),
        "bench_R_dev": ("build/verifier/features_dev.npz", "build/verifier/sample_dev.json"),
        "independent_neg": ("build/verifier/features_eval_neg.npz", "build/verifier/sample_eval_neg.json"),
    }
    loaded = {}
    for name, (feat, sample) in sets.items():
        X, meta = load_npz(PROJECT_ROOT / feat)
        sc = model.predict_proba(X)[:, 1]
        labels, paths = load_samples(PROJECT_ROOT / sample)
        loaded[name] = {"X": X, "meta": meta, "scores": sc, "labels": labels, "paths": paths}

    # The verifier v is selected on val_cal only; cascade val_cal FPR is the matched-FPR target.
    cal = loaded["val_cal"]
    cal_pos = {i for i, y in cal["labels"].items() if y}
    cal_neg = {i for i, y in cal["labels"].items() if not y}
    cascade_alarm = per_image(cal["meta"], cal["scores"], p50_cal=False, verifier_v=args.verifier_v)
    cal_cascade = metrics(cascade_alarm, cal_pos, cal_neg, "cascade")
    p50_match = calibrate_global_threshold(cal["meta"], cal["scores"], cal["labels"],
                                           cal_cascade["fpr"] or 0.02)
    print("val_cal cascade", cal_cascade, "matched P50", p50_match)

    # cause map for the independent set
    pool = [json.loads(l) for l in (PROJECT_ROOT / "data/hard_negatives_v3/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cause_by_id = {r["id"]: r.get("cause", "other") for r in pool if r.get("id")}
    hard_by_cause = defaultdict(int)
    for r in pool:
        if r.get("status") == "hard" and r.get("split") == "holdout":
            hard_by_cause[r.get("cause", "other")] += 1

    results = {"verifier": str(args.verifier), "verifier_v": args.verifier_v,
               "thresholds_source": "val_cal only", "p50_thresholds": P50_THR,
               "p50_cascade_conf": P50_CASCADE_CONF, "sets": {}, "per_cause_independent": {},
               "heldout_causes": {}, "p50_matched_fpr": {"val_cal_recall": p50_match[0] if p50_match else None,
                                                          "val_cal_fpr": p50_match[1] if p50_match else None,
                                                          "threshold": p50_match[2] if p50_match else None}}

    for name, d in loaded.items():
        pos = {i for i, y in d["labels"].items() if y}
        neg = {i for i, y in d["labels"].items() if not y}
        p50cal = per_image(d["meta"], d["scores"], p50_cal=True)
        p50_010 = per_image(d["meta"], d["scores"], p50_cal=False, verifier_v=None)
        casc = per_image(d["meta"], d["scores"], p50_cal=False, verifier_v=args.verifier_v)
        p50match = {}
        if p50_match is not None:
            thr = p50_match[2]
            p50match = defaultdict(bool)
            for m in d["meta"]:
                if float(m["conf"]) >= thr:
                    p50match[m["id"]] = True
        sets[name] = {
            "p50_calibrated": metrics(p50cal, pos, neg, "p50_calibrated"),
            "p50_at_0.10": metrics(p50_010, pos, neg, "p50_at_0.10"),
            "cascade": metrics(casc, pos, neg, "cascade"),
            "p50_matched_fpr": metrics(p50match, pos, neg, "p50_matched_fpr"),
            "mcnemar_vs_p50_cal": {
                "positives": mcnemar(casc, p50cal, pos, "positives"),
                "negatives": mcnemar(casc, p50cal, neg, "negatives"),
            },
            "mcnemar_vs_p50_matched": {
                "positives": mcnemar(casc, p50match, pos, "positives"),
                "negatives": mcnemar(casc, p50match, neg, "negatives"),
            },
        }
        print(name, json.dumps(sets[name]["cascade"]), flush=True)

    indep = loaded["independent_neg"]
    casc = per_image(indep["meta"], indep["scores"], p50_cal=False, verifier_v=args.verifier_v)
    p50cal = per_image(indep["meta"], indep["scores"], p50_cal=True)
    by = defaultdict(lambda: {"n": 0, "cascade_fp": 0, "p50_fp": 0})
    for rid in indep["labels"]:
        cause = cause_by_id.get(rid, "other")
        by[cause]["n"] += 1
        by[cause]["cascade_fp"] += int(casc.get(rid, False))
        by[cause]["p50_fp"] += int(p50cal.get(rid, False))
    for cause, v in sorted(by.items()):
        results["per_cause_independent"][cause] = {
            "n": v["n"], "cascade_fp": v["cascade_fp"], "p50_fp": v["p50_fp"],
            "cascade_fpr": round(v["cascade_fp"] / v["n"], 4),
            "p50_fpr": round(v["p50_fp"] / v["n"], 4),
            "hard_holdout_images": hard_by_cause.get(cause, 0),
            "too_few": v["n"] < 30,
        }
    for cause in ("sun_glare_sky", "skin_person", "traffic_light"):
        if cause in results["per_cause_independent"]:
            results["heldout_causes"][cause] = results["per_cause_independent"][cause]
    results["sets"] = sets
    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["heldout_causes"], indent=2))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
