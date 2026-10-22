from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def main() -> int:
    rows = []
    for p in sorted(glob.glob(str(PROJECT_ROOT / "bench_results/*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if "combined" in d:
                rows.append(d)
        except Exception:
            continue
    lines = [
        "# Frozen-bench usage log", "",
        f"`testbench100` is frozen. **Recorded uses: {len(rows)}.**", "",
        "All rows below are evaluations that have already happened, including the Phase 3/4 "
        "model-and-operating-point comparison. That comparison used the bench for model "
        "selection, so the 87% all-flame and 92% policy-filtered numbers are **optimistic**, "
        "not clean generalization estimates. Going forward: evaluate only once per major "
        "retrain, and use `testbench_B` for final claims.", "",
        "| UTC | run_id | model | val target FPR | accuracy | specificity | note |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for d in rows:
        c = d["combined"]
        lines.append(
            f"| {d.get('generated_utc', '')} | `{d.get('run_id','')}` | {d.get('model','')} | "
            f"{d.get('target_fpr','')} | {c.get('accuracy')} | {c.get('specificity')} | "
            "evaluation / model selection (pre-policy or policy diagnostic) |"
        )
    lines += ["", "## Policy diagnostic (not a new evaluation)", "",
              "The 92.0% policy-filtered figure is a **re-scoring of the same 2026-10-22 17:15 run** "
              "with controlled/spark/screen images suppressed and excluded from the denominator. "
              "It is not a separate model evaluation. 12 images were suppressed; 6 of them would "
              "have alarmed if controlled flames were included."]
    (PROJECT_ROOT / "bench_results/BENCH_USAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"uses": len(rows), "out": "bench_results/BENCH_USAGE.md"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
