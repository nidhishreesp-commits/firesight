"""Package the Stage B Kaggle inputs locally (no upload, no GPU launch).

Builds:
  build/kaggle_stage_b_code/            code + yolo11s.pt + stage_b_train.py
  build/kaggle_stage_b_eval_negatives/  5,301 independent negatives + empty labels
  kaggle/stage_b/pilot_stage_b.ipynb    E0/E1 notebook
  kaggle/stage_b/kernel-metadata.json

Hardlinks are used where possible, so the packages do not duplicate disk.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
CODE_SRC = ROOT / "build/kaggle_code"
CODE_OUT = ROOT / "build/kaggle_stage_b_code"
NEG_OUT = ROOT / "build/kaggle_stage_b_eval_negatives"
POOL = ROOT / "data/hard_negatives_v3"
EVAL_LIST = POOL / "lists/eval_neg.txt"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    # ---------------------------------------------------------------- code package
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    for src in CODE_SRC.rglob("*"):
        if src.is_file() and "__pycache__" not in src.parts:
            link_or_copy(src, CODE_OUT / src.relative_to(CODE_SRC))
    for name in ("stage_b_train.py", "stage_b_predict.py"):
        shutil.copy2(ROOT / "kaggle" / name, CODE_OUT / name)
    (CODE_OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "firesight stage b code (private)",
        "id": "nidhishreesp/firesight-stage-b-code",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": "Private E0/E1 training code. Contains stage_b_train.py, "
                       "stage_b_predict.py, pilot_eval.py, run_bench.py, firesight package and "
                       "yolo11s.pt (AGPL-3.0, Ultralytics).",
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- independent negatives
    rows = [json.loads(l) for l in (POOL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_path = {str((ROOT / r["path"]).resolve()).lower(): r for r in rows if r.get("path")}
    paths = [Path(l.strip()) for l in EVAL_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    NEG_OUT.mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels", "lists"):
        (NEG_OUT / d).mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    for i, src in enumerate(paths):
        if not src.exists():
            raise SystemExit(f"missing eval negative {src}")
        key = str(src.resolve()).lower()
        r = by_path.get(key, {})
        sha = r.get("sha256") or __import__("hashlib").sha256(src.read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        suffix = src.suffix.lower() if src.suffix.lower() in (".jpg", ".jpeg", ".png") else ".jpg"
        name = f"eval_{sha[:16]}{suffix}"
        link_or_copy(src, NEG_OUT / "images" / name)
        (NEG_OUT / "labels" / (Path(name).stem + ".txt")).write_text("", encoding="utf-8")
        manifest.append({"id": r.get("id", f"eval:{sha[:16]}"), "package_file": f"images/{name}",
                         "orig_path": str(src.relative_to(ROOT)), "cause": r.get("cause", ""),
                         "source": r.get("source", ""), "licence": r.get("licence", ""),
                         "sha256": sha, "status": r.get("status", ""), "split": "holdout"})
    (NEG_OUT / "lists/eval_neg.txt").write_text(
        "\n".join(m["package_file"] for m in manifest) + "\n", encoding="utf-8")
    (NEG_OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
    (NEG_OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "firesight stage b eval negatives (private)",
        "id": "nidhishreesp/firesight-stage-b-eval-negatives",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": "Independent negative evaluation set for Gate 1/2. Open Images V7 "
                       "validation (image licence CC BY 2.0) + held-out causes + low-confidence "
                       "quarantine. Never trained on.",
    }, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in (NEG_OUT / "images").glob("*"))
    print(json.dumps({"code": str(CODE_OUT), "eval_neg_images": len(manifest),
                      "eval_neg_GB": round(total / 1e9, 3)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
