from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT


def kaggle_creds() -> tuple[str, str]:
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if user and key:
        return user, key
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        raw = p.read_text(encoding="utf-8").replace('\\"', '"')
        data = json.loads(raw)
        return data["username"], data["key"]
    raise SystemExit("Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json")


def download_dfire() -> Path:
    import requests
    user, key = kaggle_creds()
    out = PROJECT_ROOT / "data/raw/dfire.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo"
    print("D-Fire: Kaggle mirror, CC0 (upstream per-source terms apply)")
    with requests.get(url, auth=(user, key), stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(out, out.stat().st_size)
    return out


def download_fasdd() -> Path:
    from huggingface_hub import hf_hub_download
    out = PROJECT_ROOT / "data/raw/fasdd"
    out.mkdir(parents=True, exist_ok=True)
    print("FASDD_CV: CC BY 4.0 (attribution required, Wang et al. 2025)")
    path = hf_hub_download(repo_id="weslleyskah/fasdd_cv", filename="FASDD_CV.zip",
                           repo_type="dataset", local_dir=str(out))
    print(path)
    return Path(path)


def download_msfsdb() -> None:
    import subprocess
    out = PROJECT_ROOT / "data/raw/msfsdb.zip"
    print("MS-FSDB: MIT code; Google Drive payload. Attempting gdown...")
    result = subprocess.run([sys.executable, "-m", "gdown", "14ylxaNBVmXjAFXt2h4lnyBe7xELhOVHc", "-O", str(out)])
    if result.returncode != 0 or not out.exists():
        print("MS-FSDB download failed (Drive permission/quota). Not used; see REPORT.md.")


def download_fireviewer(accept: bool) -> None:
    if not accept:
        raise SystemExit("FireViewer weights are non-redistributable. Re-run with --accept-restricted.")
    from huggingface_hub import snapshot_download
    print("FireViewer RT-DETRv2-R50: commercial use with attribution; NO redistribution/shipping of weights.")
    p = snapshot_download("fireviewer/rtdetr-v2-r50-fire-smoke", allow_patterns=["*.json", "*.safetensors", "*.md"])
    print(p)


def download_cctv_nc(accept: bool) -> None:
    if not accept:
        raise SystemExit("Simuletic CCTV is CC BY-NC 4.0 (non-commercial). Re-run with --accept-restricted.")
    from huggingface_hub import snapshot_download
    p = snapshot_download("Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset",
                          repo_type="dataset", local_dir=str(PROJECT_ROOT / "data/raw/cctv_nc"))
    print(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets/models with explicit license acknowledgement.")
    parser.add_argument("--dfire", action="store_true")
    parser.add_argument("--fasdd", action="store_true")
    parser.add_argument("--msfsdb", action="store_true")
    parser.add_argument("--fireviewer", action="store_true")
    parser.add_argument("--cctv-nc", action="store_true")
    parser.add_argument("--accept-restricted", action="store_true")
    parser.add_argument("--all-permissive", action="store_true")
    args = parser.parse_args()
    if args.all_permissive or args.dfire:
        download_dfire()
    if args.all_permissive or args.fasdd:
        download_fasdd()
    if args.msfsdb:
        download_msfsdb()
    if args.fireviewer:
        download_fireviewer(args.accept_restricted)
    if args.cctv_nc:
        download_cctv_nc(args.accept_restricted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
