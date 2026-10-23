from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench_R"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    index = {r["id"]: r for r in json.loads((BENCH / "index.json").read_text(encoding="utf-8"))}
    props = json.loads((BENCH / "proposals.json").read_text(encoding="utf-8"))
    reviews = json.loads((BENCH / "reviews.json").read_text(encoding="utf-8")) if (BENCH / "reviews.json").exists() else {}
    blind = set(props["meta"].get("blind_ids", []))
    rows = []
    for rid in blind:
        rev = reviews.get(rid)
        if not rev:
            continue
        prop = props["images"].get(rid, {})
        final_fire = rev.get("boxes_fire", [])
        final_smoke = rev.get("boxes_smoke", [])
        prop_fire = [p["box"] for p in prop.get("proposals", []) if p["label"] == "fire"]
        prop_smoke = [p["box"] for p in prop.get("proposals", []) if p["label"] == "smoke"]
        added_fire = sum(1 for b in final_fire if not any(iou(b, p) >= 0.5 for p in prop_fire))
        added_smoke = sum(1 for b in final_smoke if not any(iou(b, p) >= 0.5 for p in prop_smoke))
        deleted_fire = sum(1 for p in prop_fire if not any(iou(p, b) >= 0.5 for b in final_fire))
        deleted_smoke = sum(1 for p in prop_smoke if not any(iou(p, b) >= 0.5 for b in final_smoke))
        agree = (int(rev["has_fire"]) == int(prop.get("suggested_has_fire", 0))
                 and int(rev["has_smoke"]) == int(prop.get("suggested_has_smoke", 0)))
        rows.append({"id": rid, "family": index.get(rid, {}).get("family"), "agree": agree,
                     "added_boxes": added_fire + added_smoke, "deleted_boxes": deleted_fire + deleted_smoke})
    reviewed_blind = len(rows)
    agreement = sum(1 for r in rows if r["agree"]) / reviewed_blind if reviewed_blind else None
    added = sum(r["added_boxes"] for r in rows)
    deleted = sum(r["deleted_boxes"] for r in rows)
    by_family = defaultdict(lambda: {"n": 0, "agree": 0, "added": 0, "deleted": 0})
    for r in rows:
        b = by_family[r["family"]]
        b["n"] += 1
        b["agree"] += int(r["agree"])
        b["added"] += r["added_boxes"]
        b["deleted"] += r["deleted_boxes"]
    accepted = sum(1 for r in reviews.values() if r.get("accepted_as_proposed"))
    edits = sum(int(r.get("edits_made", 0)) for r in reviews.values())
    lines = ["# bench_R pre-annotation correction report", "",
             f"Reviewed blind sample: {reviewed_blind}/100. Agreement: {agreement*100:.1f}%." if reviewed_blind else "Blind sample not fully reviewed yet.",
             f"Boxes added by reviewer in blind sample: {added}; proposal boxes deleted: {deleted}.",
             f"Non-blind accepted-as-proposed: {accepted}/{len(reviews)}; total edits recorded: {edits}.", "",
             "| Blind family | n | agreement | added | deleted |", "|---|---:|---:|---:|---:|"]
    for fam, v in sorted(by_family.items()):
        lines.append(f"| {fam} | {v['n']} | {v['agree']}/{v['n']} | {v['added']} | {v['deleted']} |")
    (BENCH / "PREANNOTATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"blind_reviewed": reviewed_blind, "agreement": agreement, "added": added, "deleted": deleted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
