from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    runs = []
    for p in glob.glob(str(PROJECT_ROOT / "bench_results/*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if "combined" in d:
                runs.append(d)
        except Exception:
            continue
    latest = {}
    for d in runs:
        key = (d["model"], d.get("target_fpr"))
        if key not in latest or d["generated_utc"] > latest[key]["generated_utc"]:
            latest[key] = d
    rows = sorted(latest.values(), key=lambda d: (-(d["combined"]["accuracy"] or 0), d["model"]))
    lines = [
        "# testbench100 results history", "",
        "Locked bench: 100 images, composition override 62 positives / 38 negatives. "
        "Thresholds calibrated on `val_mini + val_flames` only; the bench was never tuned on.", "",
        "| Model | Val target FPR | Accuracy (95% CI) | Recall | Specificity | F1 | Fire R | Smoke R | Run |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for d in rows:
        c = d["combined"]
        ci = c.get("accuracy_95ci") or [None, None]
        lines.append(
            "| {model} | {fpr} | {acc} ({lo} to {hi}) | {rec} | {spec} | {f1} | {fr} | {sr} | `{run}` |".format(
                model=d["model"], fpr=d.get("target_fpr"), acc=fmt(c["accuracy"]),
                lo=fmt(ci[0]), hi=fmt(ci[1]), rec=fmt(c["recall"]), spec=fmt(c["specificity"]),
                f1=fmt(c["f1"]), fr=fmt(d["fire"]["recall"]), sr=fmt(d["smoke"]["recall"]),
                run=d["run_id"],
            )
        )
    lines += [
        "", "## Notes", "",
        "- `pyronear_sensitive` is a single-class wildfire-smoke detector (`item` mapped to smoke); it cannot separate fire.",
        "- `pyronear/temporal-model` is a sequence model, not applicable to the single-image bench.",
        "- `select-ai/fire-smoke-detection` had no public weights found; `e1250/safety_detection` has no declared licence and was not downloaded.",
        "- 90% and 95% image-level accuracy were **not reached**; the best is 87% with a 95% CI of 79.0 to 92.2%.",
        "- See `bench_results/<run_id>/false_positives.jpg` and `false_negatives.jpg`.",
    ]
    (PROJECT_ROOT / "bench_results/SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PROJECT_ROOT / "bench_results/summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
