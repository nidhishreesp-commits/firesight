# FireSight

Production-grade **fire + smoke detection** with a benchmarking harness, a
false-alarm reduction pipeline, a FastAPI backend and a React dashboard.

> **Safety:** FireSight supplements — never replaces — certified fire alarm
> systems. Run it in shadow mode, keep humans in the loop, and do not actuate
> sirens/doors automatically.

Measured results and honest limitations are in [`REPORT.md`](REPORT.md).
Dataset counts are in [`data/README.md`](data/README.md). Licenses are in
[`LICENSES.md`](LICENSES.md).

## What is in here

```
configs/            YAML config (env-overridable via FIRESIGHT__*)
src/firesight/      library: detectors, temporal, verifier, eval, video, API
scripts/            prepare / freeze / benchmark / mine / train / eval / serve / shadow / retrain
frontend/           React + TypeScript + Tailwind dashboard (6 pages)
tests/              unit, API and regression-gate tests
docker/             API + frontend Dockerfiles, nginx, compose
artifacts/          benchmark outputs (metrics.json, PR curves, comparison tables)
```

## Chosen detector

`models/best_fire_v3.onnx` — YOLOv12n, 10.1 MB, fire+smoke, ~14 FPS on 8 CPU
cores. It has the best recall at very low false-alarm rates among the candidates
measured (see `REPORT.md`). RT-DETRv2-R50 is the accuracy ceiling if you have a
GPU and accept its non-redistributable license.

## Quick start

```powershell
python -m pip install -r requirements.txt
# model weights are git-ignored: provision models/best_fire_v3.onnx (see models/README.md)

# 1. data (downloads D-Fire via Kaggle creds + FASDD via Hugging Face)
python scripts/download_data.py --all-permissive
python scripts/prepare_data.py --sources dfire,fasdd,local
python scripts/freeze_test.py
python scripts/make_yolo_dataset.py

# 2. benchmark the frozen test (calibrates thresholds on val only)
python scripts/benchmark.py --models ssprod_best_fire_v3,ssprod_best_fire_v4 `
  --manifest data/splits/test_bench.jsonl --val-manifest data/splits/val_bench.jsonl `
  --target-fpr 0.01 --out artifacts/benchmark_stage1

# 3. backend
python scripts/serve.py            # http://localhost:8000  (docs at /docs)

# 4. frontend
cd frontend; npm install; npm run dev   # http://localhost:5173
```

## Benchmarking

`scripts/benchmark.py` runs any registry model on identical splits, caches raw
detections, calibrates per-class thresholds on **val** at a target image-level
false-alarm rate, then reports on the **frozen test**: precision, recall, mAP@50,
mAP@50-95 per class, FPS, model size, license, recall-at-FAR and bootstrap CIs.

Add a model to `models/registry.yaml` (`backend: yolo|onnx|rtdetr_v2|grounding_dino`).

## Hand-labeled bench (Phase 1-2)

`testbench100/` is the frozen-bench candidate: **50 positives** (25 fire-only,
15 smoke-only, 10 fire+smoke) and **50 hard negatives** across light/colour,
smoke-like, fire-adjacent and camera-artifact families.

**Status: PENDING HUMAN REVIEW. Do not benchmark it and do not report accuracy
until it is locked.** Open the dashboard page **Bench review**, enter your name,
verify/edit every image, then click **lock bench**. Locking writes `labels.csv`,
YOLO labels and a SHA-256 manifest. The API refuses to lock unless all 100 images
are reviewed and the exact composition is preserved.

```powershell
python scripts/build_bench100.py          # rebuild candidates (refuses to overwrite)
python scripts/check_bench_leakage.py     # pHash + sampled CLIP embedding check
python scripts/bench_contact_sheet.py --kind positive
python scripts/bench_contact_sheet.py --kind negative
```

Hard-negative taxonomy and licences: `hard_negatives/CATALOG.md`. Per-image
Wikimedia attribution: `hard_negatives/sources/wikimedia/attribution.jsonl`.
The leakage check is exhaustive for pHash over train/val and sampled for CLIP
embeddings; results are in `testbench100/LEAKAGE_CHECK.json`.

## False-alarm reduction

1. **Hard-negative mining** — `scripts/mine_hard_negatives.py` (TRAIN non-fire
   images + non-fire video only).
2. **Temporal N-of-M + ByteTrack** — fire 3/5, smoke 4/6 with growth cue
   (configurable per camera).
3. **Second-stage verifier** — MobileNetV3 crop classifier (fire / lookalike /
   smoke), `scripts/train_verifier.py`, `scripts/eval_verifier.py`.
4. **Flame-evidence class correction** — a smoke box whose crop is ≥8% flame-
   coloured is relabelled `fire` (evidence kept in `meta.flame_fraction`); no new
   boxes are created on negatives.
4. **Per-camera config** — ROI, confidence, N/M and alarm cooldown in
   `configs/cameras.yaml`.

Ablation numbers are in `REPORT.md` §5 and `artifacts/verifier_ablation/`,
`artifacts/video_ablation/`.

## API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model/camera status |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/config`, `/api/models` | active config, model registry |
| POST | `/api/detect/image` | upload image → boxes + annotated JPEG |
| POST | `/api/detect/video` | upload clip → per-frame timeline + alarm events |
| GET | `/api/testsets`, `/api/testsets/{name}/images` | browse test sets with GT vs prediction, filters (TP/FP/FN/negative) |
| GET | `/api/image/{record_id}` | serve a test image |
| POST | `/api/feedback`, GET `/api/review/queue` | operator review queue (retraining input) |
| GET | `/api/benchmark/latest`, `/api/artifacts/{path}` | benchmark tables, PR curves, metrics |
| GET/POST | `/api/live/cameras`, `/api/live/{id}/start|stop` | camera control |
| GET | `/api/live/{id}/mjpeg` | MJPEG stream with overlays |
| WS | `/api/live/{id}/ws` | JSON detections + alarm state per frame |

Live sources: webcam index, RTSP/HTTP URL, or video file.

## Frontend

React + TypeScript + Tailwind (Vite). Pages: **Live view** (multi-camera tiles,
overlays, alarm banner), **Test bench** (browse/filter/FP-FN feedback),
**Run benchmark** (tables + PR curve), **Video test** (upload + timeline),
**Review queue**, **Model comparison**.

## Production readiness

- **Docker**: `docker compose up --build` → API on :8000, dashboard on :8080.
  Uncomment the `deploy.resources` block for GPU. `.dockerignore` keeps data and
  weights out of the image; mount them as volumes.
- **Model versioning / rollback**: `scripts/retrain_pipeline.py` writes
  `models/versions/<timestamp>/` and updates `models/active.json`.
- **ONNX / TensorRT export**: `scripts/export_onnx.py --format onnx|engine --device cpu|cuda:0`
  (writes a sha256 manifest). TensorRT requires CUDA and the TensorRT runtime.
- **Tests**: `python -m pytest` (unit + API + regression gate).
- **Regression gate**: `tests/test_regression_gate.py` fails the build if TEST
  recall drops > 2 pp or TEST FPR rises > 0.5 pp versus
  `tests/regression_baseline.json`.
- **Shadow mode**: `python scripts/shadow_run.py --source <video|rtsp|0>` logs
  every detection/alarm to JSONL without actuating anything. Run for weeks,
  review, then tune.
- **Monitoring**: structured JSON logs, Prometheus counters/histograms
  (`firesight_detections_total`, `firesight_alarms_total`,
  `firesight_inference_latency_ms`, `firesight_stream_fps`, ...), plus a drift
  monitor skeleton (`src/firesight/monitoring.py`).
- **Retraining**: operator feedback → `data/feedback/queue.jsonl` →
  `scripts/retrain_pipeline.py` → versioned model → regression gate → promote.

## Environment overrides

Any config key can be overridden with a double-underscore env var, e.g.:

```
FIRESIGHT__DETECTOR__MODEL_PATH=models/best_fire_v4.onnx
FIRESIGHT__DETECTOR__CONF__SMOKE=0.4
FIRESIGHT__VERIFIER__ENABLED=true
```

## Honest limitations

See `REPORT.md` §7. In short: smoke recall is low at very low FPR; sunsets,
headlights, lamps, chandeliers, orange PPE, clouds and orange objects are the
main false-positive sources; the TEST split may be contaminated for models
trained on merged D-Fire/FASDD data; non-fire video coverage is only ~3 minutes.
