"""Emit sample JSONs (id,path,w,h,labels) for val_cal, val_report and bench_R DEV."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from firesight.config import PROJECT_ROOT
from pilot_eval import load_list, load_records
from run_benchR import load_benchR

def dump(records, out):
    rows = []
    for r in records:
        rows.append({"id": r.id, "path": r.path, "w": r.width, "h": r.height,
                     "labels": [{"label": g.label, "box": g.box.to_list()} for g in r.labels]})
    (PROJECT_ROOT/out).write_text(json.dumps(rows), encoding="utf-8")
    print(out, len(rows))

ds = PROJECT_ROOT/"build/kaggle_dataset"
dump(load_records(load_list(ds/"lists"/"val_cal.txt", ds), ds), "build/verifier/sample_val_cal.json")
dump(load_records(load_list(ds/"lists"/"val_report.txt", ds), ds), "build/verifier/sample_val_report.json")
records, member = load_benchR()
dump([r for r in records if member[r.id] == "dev"], "build/verifier/sample_dev.json")
