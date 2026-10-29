"""Cascade evaluation: P50 (0.10/0.15) + verifier, calibrated on val_cal, reported on val_report.

Also: McNemar vs calibrated P50, per-cause FP scorecard, verifier latency.
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from firesight.config import PROJECT_ROOT
from run_bench import wilson
from pilot_eval import load_list, load_records, read_preds
from scipy.stats import binomtest


def load_split(path: Path):
    d = np.load(path, allow_pickle=True)
    meta = [json.loads(m) for m in d["meta"]]
    return d["X"], meta


def score(model, X):
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, (time.perf_counter() - t0) / max(len(X), 1) * 1000


def image_table(meta, scores, t):
    by = defaultdict(list)
    for m, s in zip(meta, scores):
        if m["conf"] >= t:
            by[m["id"]].append((m["conf"], float(s), m["cause"], m["label"]))
    return by


def metrics(table, pos_ids, neg_ids, t, v):
    tp = sum(1 for i in pos_ids if any(s >= v for _, s, _, _ in table.get(i, [])))
    fp = sum(1 for i in neg_ids if any(s >= v for _, s, _, _ in table.get(i, [])))
    return tp, fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build/verifier/model.joblib")
    ap.add_argument("--dir", default="build/verifier")
    ap.add_argument("--targets", default="0.015,0.02")
    ap.add_argument("--out", default="build/verifier/eval.json")
    args = ap.parse_args()
    import joblib
    bundle = joblib.load(PROJECT_ROOT/args.model)
    model = bundle["model"]
    print("model", bundle["kind"], "cv", (bundle.get("cv") or {}).get(bundle["kind"], {}).get("auc_mean"))

    ds = PROJECT_ROOT/"build/kaggle_dataset"
    splits = {}
    for name, sample, feats in (("val_cal", "sample_val_cal.json", "features_val_cal.npz"),
                                ("val_report", "sample_val_report.json", "features_val_report.npz"),
                                ("dev", "sample_dev.json", "features_dev.npz")):
        rows = json.loads((PROJECT_ROOT/args.dir/sample).read_text(encoding="utf-8"))
        pos = {r["id"] for r in rows if r["labels"]}
        neg = {r["id"] for r in rows if not r["labels"]}
        X, meta = load_split(PROJECT_ROOT/args.dir/feats)
        sc, ms = score(model, X)
        splits[name] = {"pos": pos, "neg": neg, "meta": meta, "scores": sc, "latency_ms": ms}
        print(name, "dets", len(X), "pos imgs", len(pos), "neg imgs", len(neg), "latency/det", round(ms, 4), "ms", flush=True)

    targets = [float(x) for x in args.targets.split(",")]
    results = {}
    for target in targets:
        best = None
        for t in (0.10, 0.15):
            table = image_table(splits["val_cal"]["meta"], splits["val_cal"]["scores"], t)
            vs = np.unique(np.round(splits["val_cal"]["scores"], 4))
            for v in np.quantile(vs, np.linspace(0.0, 1.0, 101)):
                tp, fp = metrics(table, splits["val_cal"]["pos"], splits["val_cal"]["neg"], t, v)
                fpr = fp/max(len(splits["val_cal"]["neg"]), 1)
                if fpr > target: continue
                rec = tp/max(len(splits["val_cal"]["pos"]), 1)
                if best is None or rec > best[0]:
                    best = (rec, t, float(v), fpr)
        if best is None:
            results[target] = {"error": "no operating point"}; continue
        rec_cal, t, v, fpr_cal = best
        table = image_table(splits["val_report"]["meta"], splits["val_report"]["scores"], t)
        tp, fp = metrics(table, splits["val_report"]["pos"], splits["val_report"]["neg"], t, v)
        npos, nneg = len(splits["val_report"]["pos"]), len(splits["val_report"]["neg"])
        results[target] = {"p50_threshold": t, "verifier_threshold": v, "val_cal_fpr": fpr_cal,
                           "recall": tp/npos, "recall_95ci": wilson(tp, npos),
                           "fpr": fp/nneg, "fpr_95ci": wilson(fp, nneg),
                           "accuracy": (tp + (nneg-fp))/(npos+nneg), "tp": tp, "fp": fp,
                           "n_pos": npos, "n_neg": nneg}
        print("target", target, json.dumps(results[target]), flush=True)

    # P50 calibrated reference + McNemar on val_report
    preds = read_preds(PROJECT_ROOT/"build/pilot_eval/preds_P50.jsonl")
    thr = json.loads((PROJECT_ROOT/"build/pilot_eval/metrics_P50.json").read_text())["thresholds"]
    def alarm(rid):
        return any(d.confidence >= thr.get(d.label, 1.0) for d in preds.get(rid, []))
    base_tp = sum(1 for i in splits["val_report"]["pos"] if alarm(i))
    base_fp = sum(1 for i in splits["val_report"]["neg"] if alarm(i))
    print("P50 calibrated: recall", base_tp/len(splits["val_report"]["pos"]), "fpr", base_fp/len(splits["val_report"]["neg"]))
    for target, r in results.items():
        if "error" in r: continue
        t, v = r["p50_threshold"], r["verifier_threshold"]
        table = image_table(splits["val_report"]["meta"], splits["val_report"]["scores"], t)
        def casc(rid):
            return any(s >= v for _, s, _, _ in table.get(rid, []))
        for label, ids, base_fn in (("positives", splits["val_report"]["pos"], alarm), ("negatives", splits["val_report"]["neg"], alarm)):
            a = [casc(i) for i in ids]; b = [base_fn(i) for i in ids]
            b01 = sum(1 for x, y in zip(a, b) if x and not y)
            b10 = sum(1 for x, y in zip(a, b) if not x and y)
            p = binomtest(min(b01, b10), b01+b10, 0.5).pvalue if b01+b10 else 1.0
            r.setdefault("mcnemar", {})[label] = {"cascade_only": b01, "p50_only": b10, "p": p}
            print(f"  {target} {label}: cascade-only {b01} vs p50-only {b10}, p={p:.3g}")

    # DEV sanity
    t, v = results[targets[0]]["p50_threshold"], results[targets[0]]["verifier_threshold"]
    table = image_table(splits["dev"]["meta"], splits["dev"]["scores"], t)
    tp, fp = metrics(table, splits["dev"]["pos"], splits["dev"]["neg"], t, v)
    npos, nneg = len(splits["dev"]["pos"]), len(splits["dev"]["neg"])
    results["dev_sanity"] = {"p50_threshold": t, "verifier_threshold": v,
                             "recall": tp/npos, "recall_95ci": wilson(tp, npos),
                             "fpr": fp/nneg, "fpr_95ci": wilson(fp, nneg), "tp": tp, "fp": fp,
                             "n_pos": npos, "n_neg": nneg}
    print("DEV sanity", json.dumps(results["dev_sanity"]))

    # per-cause scorecard on val_report negatives (cause from top detection at t)
    t, v = results[targets[0]]["p50_threshold"], results[targets[0]]["verifier_threshold"]
    table = image_table(splits["val_report"]["meta"], splits["val_report"]["scores"], t)
    causes = defaultdict(lambda: {"neg_images": 0, "p50_fa": 0, "cascade_fa": 0})
    for rid in splits["val_report"]["neg"]:
        dets = table.get(rid, [])
        if not dets: continue
        cause = max(dets, key=lambda d: d[0])[2]
        causes[cause]["neg_images"] += 1
        causes[cause]["p50_fa"] += 1
        causes[cause]["cascade_fa"] += int(any(s >= v for _, s, _, _ in dets))
    results["per_cause_val_report"] = {k: {**vv, "fpr_before": vv["p50_fa"]/vv["neg_images"],
                                           "fpr_after": vv["cascade_fa"]/vv["neg_images"],
                                           "too_few": vv["neg_images"] < 30} for k, vv in sorted(causes.items())}
    (PROJECT_ROOT/args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
