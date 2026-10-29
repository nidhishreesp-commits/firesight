"""Gate: prove the collected hard negatives do not overlap bench_R, testbench100 or TEST.

Compares 64-bit pHash (Hamming <= 2 near-duplicate, <= 6 reported as scene-similar) and
checks id/path overlap. Writes <hard-neg>/leakage_report.json with a PASS/FAIL verdict.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def phash(path: Path) -> str | None:
    try:
        import imagehash
        from PIL import Image
        return str(imagehash.phash(Image.open(path).convert("RGB")))
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-neg", default="data/hard_negatives_v2")
    args = ap.parse_args()
    hn = PROJECT_ROOT / args.hard_neg
    rows = [json.loads(l) for l in (hn / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    selected = [r for r in rows if r.get("status") == "selected"]

    bench = json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))
    bench_ph = {r["id"]: r["phash"] for r in bench if r.get("phash")}
    bench_paths = {str(r.get("file", "")).replace("\\", "/").lower() for r in bench}

    tb = json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8"))
    tb_ph = {}
    for r in tb:
        p = PROJECT_ROOT / "testbench100/images" / r["file"]
        if p.exists():
            v = phash(p)
            if v:
                tb_ph[r["id"]] = v
    tb_ids = {r["id"] for r in tb}

    test_rows = [json.loads(l) for l in (PROJECT_ROOT / "data/splits/test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_ph = {r["id"]: r["phash"] for r in test_rows if r.get("phash")}
    test_paths = {str(r.get("path", "")).replace("\\", "/").lower() for r in test_rows}

    ref = {**{f"bench_R:{k}": v for k, v in bench_ph.items()},
           **{f"testbench100:{k}": v for k, v in tb_ph.items()},
           **{f"TEST:{k}": v for k, v in test_ph.items()}}

    near2, near6, missing = [], [], 0
    for r in selected:
        p = PROJECT_ROOT / r["file"]
        h = phash(p)
        if h is None:
            missing += 1
            continue
        for rid, rh in ref.items():
            d = hamming(h, rh)
            if d <= 2:
                near2.append({"hardneg": r["id"], "ref": rid, "distance": d})
            elif d <= 6:
                near6.append({"hardneg": r["id"], "ref": rid, "distance": d})

    hn_paths = {str(r["file"]).replace("\\", "/").lower() for r in selected}
    path_hits = sorted(hn_paths & (bench_paths | test_paths))
    id_hits = sorted({r["id"] for r in selected} & (tb_ids | set(test_ph) | set(bench_ph)))

    report = {
        "selected": len(selected), "unreadable": missing,
        "reference_sets": {"bench_R": len(bench_ph), "testbench100": len(tb_ph), "TEST": len(test_ph)},
        "phash_near_duplicates_le_2": len(near2), "phash_near_duplicate_examples": near2[:10],
        "phash_scene_similar_3_6": len(near6), "phash_scene_similar_examples": near6[:10],
        "path_overlap": len(path_hits), "path_overlap_examples": path_hits[:5],
        "id_overlap": len(id_hits), "id_overlap_examples": id_hits[:5],
        "verdict": "PASS" if not near2 and not path_hits and not id_hits else "FAIL",
    }
    (hn / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected", "phash_near_duplicates_le_2",
                                             "phash_scene_similar_3_6", "path_overlap",
                                             "id_overlap", "verdict")}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
