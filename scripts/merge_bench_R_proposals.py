from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    index = json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))
    raw = []
    for name in ("owlv2", "dino"):
        p = PROJECT_ROOT / f"bench_R/proposals_raw/{name}.jsonl"
        if p.exists():
            raw += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = defaultdict(list)
    for r in raw:
        by_id[r["id"]].append(r)
    # Blind 10%: 60 positives + 40 negatives, stratified-ish and random.
    rng = random.Random(20261022)
    pos = [r for r in index if r.get("proposed_has_fire") or r.get("proposed_has_smoke")]
    neg = [r for r in index if not (r.get("proposed_has_fire") or r.get("proposed_has_smoke"))]
    blind_ids = {r["id"] for r in rng.sample(pos, 60)} | {r["id"] for r in rng.sample(neg, 40)}
    images = {}
    for row in index:
        dets = []
        for rr in by_id.get(row["id"], []):
            for d in rr["detections"]:
                dets.append({**d, "source": rr["model"]})
        merged = []
        for cls in ("fire", "smoke"):
            cls_dets = sorted([d for d in dets if d["label"] == cls], key=lambda d: d["score"], reverse=True)
            clusters = []
            for d in cls_dets:
                placed = False
                for c in clusters:
                    if iou(d["box"], c["box"]) >= 0.5:
                        w = max(c["score"], 1e-6)
                        c["box"] = [round((c["box"][i] * w + d["box"][i] * d["score"]) / (w + d["score"]), 2) for i in range(4)]
                        c["score"] = round(max(c["score"], d["score"]), 4)
                        c["sources"] = sorted(set(c["sources"] + [d["source"]]))
                        placed = True
                        break
                if not placed:
                    clusters.append({"label": cls, "prompt": d.get("prompt"), "box": d["box"],
                                     "score": d["score"], "sources": [d["source"]]})
            merged += clusters
        family = row.get("family", "")
        ftype = {"controlled_fire": "controlled", "spark_screen": "spark",
                 "uncontrolled_fire": "uncontrolled", "both": "uncontrolled"}.get(family, "unknown")
        images[row["id"]] = {
            "blind": row["id"] in blind_ids,
            "proposals": merged,
            "suggested_has_fire": int(any(p["label"] == "fire" and p["score"] >= 0.05 for p in merged)),
            "suggested_has_smoke": int(any(p["label"] == "smoke" and p["score"] >= 0.05 for p in merged)),
            "suggested_flame_type": ftype if any(p["label"] == "fire" for p in merged) else None,
        }
    # coverage per positive family
    coverage = defaultdict(lambda: {"n": 0, "with_proposal": 0})
    for row in index:
        if not (row.get("proposed_has_fire") or row.get("proposed_has_smoke")):
            continue
        c = coverage[row["family"]]
        c["n"] += 1
        c["with_proposal"] += int(bool(images[row["id"]]["proposals"]))
    runtimes = defaultdict(float)
    counts = defaultdict(int)
    for r in raw:
        runtimes[r["model"]] += r.get("elapsed_s", 0)
        counts[r["model"]] += 1
    meta = {
        "proposers": {
            "owlv2": {"model": "google/owlv2-base-patch16-ensemble", "license": "Apache-2.0",
                      "images": counts.get("owlv2", 0), "runtime_s": round(runtimes.get("owlv2", 0.0), 1),
                      "prompts": ["fire", "flame", "candle flame", "smoke", "sparks"]},
            "grounding_dino_tiny": {"model": "IDEA-Research/grounding-dino-tiny", "license": "Apache-2.0",
                                    "images": counts.get("grounding_dino_tiny", 0), "runtime_s": round(runtimes.get("grounding_dino_tiny", 0.0), 1),
                                    "prompts": ["fire", "flame", "candle flame", "smoke", "sparks"]},
        },
        "merge": "per-class weighted box fusion (IoU>=0.5), low threshold 0.01; suggested labels at 0.05",
        "blind_ids": sorted(blind_ids),
        "blind_design": "60 positives + 40 negatives, random seed 20261022; proposals are hidden in the UI",
        "coverage_positive_families": {k: {**v, "rate": round(v["with_proposal"] / v["n"], 4)} for k, v in sorted(coverage.items())},
        "negatives_proposed": 0,
        "note": "Only positives were proposed; negatives need single-key confirmation. No evaluated detector was used.",
    }
    (PROJECT_ROOT / "bench_R/proposals.json").write_text(json.dumps({"meta": meta, "images": images}, indent=2), encoding="utf-8")
    lines = ["# bench_R pre-annotation report", "",
             "**Proposals are not labels.** Human review is the ground truth. No evaluated detector was used.", "",
             "| Proposer | Licence | Images | Runtime |", "|---|---|---:|---:|"]
    for k, v in meta["proposers"].items():
        lines.append(f"| {v['model']} | {v['license']} | {v['images']} | {v['runtime_s']/60:.1f} min |")
    lines += ["", "## Coverage (positives)", "", "| Family | Positives | With proposal | Rate |", "|---|---:|---:|---:|"]
    for k, v in meta["coverage_positive_families"].items():
        lines.append(f"| {k} | {v['n']} | {v['with_proposal']} | {v['rate']*100:.1f}% |")
    lines += ["", f"Blind 10%: {len(blind_ids)} images (60 positives, 40 negatives); proposals hidden in the review UI.",
              "Second-pass spot check is separate (100 images, not the blind sample).",
              "Correction rate per proposer/category is pending human review and will be appended after locking.",
              "", "Licences: OWLv2 and Grounding DINO are Apache-2.0; model cards checked. No restricted model was used."]
    (PROJECT_ROOT / "bench_R/PREANNOTATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(index), "positives_with_proposals": sum(v["with_proposal"] for v in coverage.values()),
                      "blind": len(blind_ids), "runtime_min": {k: round(v / 60, 1) for k, v in runtimes.items()},
                      "coverage": meta["coverage_positive_families"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
