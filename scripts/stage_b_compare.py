"""Gate 1/2 comparison for E0 vs E1.

Uses image-level max confidence (global threshold), calibrated on val_cal only. Reports:
  * matched-FPR operating points (0.9/1.5/2.0/3.0% target on val_cal) on val_report and the
    independent negatives;
  * matched-recall FPR (E0's val_report recall applied to E1) on val_report and independent;
  * paired McNemar on identical images;
  * per-cause FPR on the independent negatives with denominators and too_few flags.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_preds(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = max((float(d["confidence"]) for d in r.get("detections", [])), default=0.0)
    return out


def load_labels(sample: Path) -> dict[str, int]:
    rows = json.loads(sample.read_text(encoding="utf-8"))
    return {r["id"]: int(bool(r.get("labels"))) for r in rows}


def calibrate(scores: dict[str, float], labels: dict[str, int], target_fpr: float):
    pos = [i for i, y in labels.items() if y]
    neg = [i for i, y in labels.items() if not y]
    best = None
    for thr in sorted(set(scores.values()), reverse=True):
        tp = sum(1 for i in pos if scores.get(i, 0) >= thr)
        fp = sum(1 for i in neg if scores.get(i, 0) >= thr)
        fpr = fp / max(len(neg), 1)
        rec = tp / max(len(pos), 1)
        if fpr <= target_fpr and (best is None or rec > best[0] or (rec == best[0] and fpr < best[1])):
            best = (rec, fpr, float(thr))
    return best


def metrics(scores: dict[str, float], labels: dict[str, int], thr: float):
    pos = [i for i, y in labels.items() if y]
    neg = [i for i, y in labels.items() if not y]
    tp = sum(1 for i in pos if scores.get(i, 0) >= thr)
    fp = sum(1 for i in neg if scores.get(i, 0) >= thr)
    return {"tp": tp, "fp": fp, "n_pos": len(pos), "n_neg": len(neg),
            "recall": tp / len(pos) if pos else None, "recall_95ci": wilson(tp, len(pos)) if pos else None,
            "fpr": fp / len(neg) if neg else None, "fpr_95ci": wilson(fp, len(neg)) if neg else None}


def mcnemar(a: dict[str, float], b: dict[str, float], ids: list[str], ta: float, tb: float):
    b01 = sum(1 for i in ids if a.get(i, 0) >= ta and not b.get(i, 0) >= tb)
    b10 = sum(1 for i in ids if not a.get(i, 0) >= ta and b.get(i, 0) >= tb)
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"e0_only": b01, "e1_only": b10, "p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0-val-cal", required=True)
    ap.add_argument("--e1-val-cal", required=True)
    ap.add_argument("--e0-report", required=True)
    ap.add_argument("--e1-report", required=True)
    ap.add_argument("--e0-indep", required=True)
    ap.add_argument("--e1-indep", required=True)
    ap.add_argument("--sample-val-cal", default="build/verifier/sample_val_cal.json")
    ap.add_argument("--sample-report", default="build/verifier/sample_val_report.json")
    ap.add_argument("--targets", default="0.009,0.015,0.02,0.03")
    ap.add_argument("--out", default="build/stage_b/gate1.json")
    args = ap.parse_args()

    cal_labels = load_labels(PROJECT_ROOT / args.sample_val_cal)
    rep_labels = load_labels(PROJECT_ROOT / args.sample_report)
    e0c = load_preds(PROJECT_ROOT / args.e0_val_cal)
    e1c = load_preds(PROJECT_ROOT / args.e1_val_cal)
    e0r = load_preds(PROJECT_ROOT / args.e0_report)
    e1r = load_preds(PROJECT_ROOT / args.e1_report)
    e0i = load_preds(PROJECT_ROOT / args.e0_indep)
    e1i = load_preds(PROJECT_ROOT / args.e1_indep)
    indep_labels = {i: 0 for i in e0i}
    pool = [json.loads(l) for l in (PROJECT_ROOT / "data/hard_negatives_v3/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cause = {Path(r["path"]).stem: r.get("cause", "other") for r in pool if r.get("path")}

    results = {"targets": [float(t) for t in args.targets.split(",")], "matched_fpr": {}, "matched_recall": {}}
    for target in results["targets"]:
        c0 = calibrate(e0c, cal_labels, target)
        c1 = calibrate(e1c, cal_labels, target)
        if c0 is None or c1 is None:
            results["matched_fpr"][str(target)] = {"error": "no operating point"}
            continue
        t0, t1 = c0[2], c1[2]
        r0, r1 = metrics(e0r, rep_labels, t0), metrics(e1r, rep_labels, t1)
        i0, i1 = metrics(e0i, indep_labels, t0), metrics(e1i, indep_labels, t1)
        neg_ids = [i for i, y in rep_labels.items() if not y]
        indep_ids = list(indep_labels)
        results["matched_fpr"][str(target)] = {
            "E0": {"val_cal_threshold": t0, "val_cal_recall": c0[0], "val_cal_fpr": c0[1],
                   "val_report": r0, "independent": i0},
            "E1": {"val_cal_threshold": t1, "val_cal_recall": c1[0], "val_cal_fpr": c1[1],
                   "val_report": r1, "independent": i1},
            "mcnemar_val_report_negatives": mcnemar(e0r, e1r, neg_ids, t0, t1),
            "mcnemar_independent": mcnemar(e0i, e1i, indep_ids, t0, t1),
            "relative_fpr_change_report": round((r1["fpr"] - r0["fpr"]) / r0["fpr"], 4) if r0["fpr"] else None,
            "relative_fpr_change_independent": round((i1["fpr"] - i0["fpr"]) / i0["fpr"], 4) if i0["fpr"] else None,
        }

    # matched recall: use E0's val_report recall at its 1.5% point, then find each arm's threshold
    base_target = 0.015
    c0 = calibrate(e0c, cal_labels, base_target)
    c1 = calibrate(e1c, cal_labels, base_target)
    if c0 and c1:
        r0 = metrics(e0r, rep_labels, c0[2])
        R = r0["recall"]
        # E0 threshold on val_report at recall >= R with minimum FPR
        def thr_for_recall(scores, labels, rec_target):
            pos = [i for i, y in labels.items() if y]
            neg = [i for i, y in labels.items() if not y]
            best = None
            for thr in sorted(set(scores.values()), reverse=True):
                tp = sum(1 for i in pos if scores.get(i, 0) >= thr)
                fp = sum(1 for i in neg if scores.get(i, 0) >= thr)
                rr = tp / max(len(pos), 1)
                fpr = fp / max(len(neg), 1)
                if rr >= rec_target and (best is None or fpr < best[0]):
                    best = (fpr, float(thr))
            return best
        b0 = thr_for_recall(e0r, rep_labels, R)
        b1 = thr_for_recall(e1r, rep_labels, R)
        if b0 and b1:
            t0, t1 = b0[1], b1[1]
            r0, r1 = metrics(e0r, rep_labels, t0), metrics(e1r, rep_labels, t1)
            i0, i1 = metrics(e0i, indep_labels, t0), metrics(e1i, indep_labels, t1)
            neg_ids = [i for i, y in rep_labels.items() if not y]
            results["matched_recall"] = {
                "recall_level": R, "E0_threshold": t0, "E1_threshold": t1,
                "E0_val_report": r0, "E1_val_report": r1, "E0_independent": i0, "E1_independent": i1,
                "mcnemar_val_report_negatives": mcnemar(e0r, e1r, neg_ids, t0, t1),
                "mcnemar_independent": mcnemar(e0i, e1i, list(indep_labels), t0, t1),
                "relative_fpr_change_report": round((r1["fpr"] - r0["fpr"]) / r0["fpr"], 4) if r0["fpr"] else None,
                "relative_fpr_change_independent": round((i1["fpr"] - i0["fpr"]) / i0["fpr"], 4) if i0["fpr"] else None,
            }
            # per-cause at matched recall (E1 threshold)
            by = defaultdict(lambda: {"n": 0, "e0_fp": 0, "e1_fp": 0})
            for rid in indep_labels:
                c = cause.get(rid, "other")
                by[c]["n"] += 1
                by[c]["e0_fp"] += int(e0i.get(rid, 0) >= t0)
                by[c]["e1_fp"] += int(e1i.get(rid, 0) >= t1)
            results["matched_recall"]["per_cause_independent"] = {
                c: {"n": v["n"], "e0_fpr": round(v["e0_fp"] / v["n"], 4),
                    "e1_fpr": round(v["e1_fp"] / v["n"], 4), "too_few": v["n"] < 30}
                for c, v in sorted(by.items())
            }

    (PROJECT_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
