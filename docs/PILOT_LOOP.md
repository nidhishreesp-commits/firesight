# Pilot loop: shadow mode, review queue and monthly mining

Status: **hooks built, not yet run on site.** No site or camera footage exists yet. This
document defines how pilot evidence enters the same gated E0/E1 loop without contaminating
TEST/SEALED or the final held-out `Sample_videos`.

## 0. Hard rule

`C:\Users\Nidhi\My projects\Sample_videos` is the **final held-out check only**. The monthly
miner refuses any path under that directory (`scripts/pilot_monthly_mining.py` guard).
Pilot footage is stored outside it and is the only footage the loop mines.

## 1. Shadow mode (log every alert, never actuate)

`scripts/shadow_run.py` runs the detector and the temporal layer on a live stream or file and
writes `artifacts/shadow/shadow_<ts>.jsonl`:

- one `summary` row (frames processed, alarm events, first alarm);
- one `frame` row for every frame with a detection or an active alarm, containing frame index,
  timestamp, detections, confirmed detections and alarm state.

The alert-level record therefore has **frame index + camera id + confidence + class + box**. A
reviewer decision is attached through the existing feedback API (section 2), keyed by the alert
record id.

```powershell
python scripts/shadow_run.py --source rtsp://camera-01/stream --model models/best_fire_v3.onnx
```

## 2. Review queue and "false alarm" dismissal

The API already exposes the queue and feedback store:

| endpoint | purpose |
|---|---|
| `GET /api/review/queue?limit=200` | latest alert/feedback rows from `data/feedback/queue.jsonl` |
| `POST /api/feedback` | reviewer verdict: `record_id`, `verdict`, `note`, `model`, `label` |

`data/feedback/queue.jsonl` is the single append-only store. A dismissal row records the alert
id, verdict (`false_alarm`), free-text cause and the reviewer label. The video layer
(`src/firesight/video_filter.py`) consumes the same dismissals:

- `VideoFalseAlarmFilter.update` routes static, non-growing, non-flickering detections to review
  instead of alarm (sun, lamps, sunset glow, clouds);
- `record_dismissal(camera_id, frame, box)` stores frame, box, timestamp and a colour histogram;
- once `min_dismissals` are recorded for the same region/appearance, repeats are suppressed with
  a configurable TTL.

Unit tests: `tests/test_video_filter.py` (static routing, growth, flicker, dismissal suppression).

## 3. Monthly mining job

```powershell
# 1) mine pilot frames/videos (<=20 frames/video, pHash dedupe, P50 at conf >= 0.05)
python scripts/pilot_monthly_mining.py --input data/pilot_frames --out data/pilot_mined

# 2) feed the same Stage A gate (SHA-256 + pHash + id, causes, quarantine, splits)
python scripts/mine_stage_a.py ^
  --manifests data/pilot_mined/manifest.jsonl ^
              data/mined_pools/wikimedia_stage_a/manifest.jsonl ^
              data/mined_pools/openimages/manifest.jsonl ^
              build/stage_a/free_pool_false.jsonl ^
  --out data/hard_negatives_v3
```

The miner writes `pilot_footage` rows with the proprietary-internal licence, applies the same
leakage gate, and holds them out of the independent evaluation set unless explicitly promoted.
Training still only happens through the E0/E1 gate in `docs/GATE_RESULTS.md`; the monthly job
never trains automatically.

## 4. What the pilot must measure (not yet measured)

| metric | definition | status |
|---|---|---|
| alerts per camera-hour | shadow-mode alarm events / camera-hour | not measured |
| false alarms per cause | reviewer dismissals grouped by `label`/`note` | not measured |
| dismissal agree rate | dismissals that survive a second reviewer | not measured |
| recall on real events | confirmed fire/smoke events / real events | not measured |
| latency | detector + temporal + verifier per frame | measured on bench only, not on site |

Until these are measured on the pilot, no claim is made about unseen cameras, sites or
lighting conditions.

## 5. Privacy and retention

- Pilot frames stay on the private pilot storage; only deduped, licence-tagged training crops or
  frames leave it.
- The monthly miner records camera id and timestamp; reviewer notes are stored in the feedback
  queue.
- No pilot frame enters `TEST`, `SEALED`, `bench_R`, `testbench100` or the held-out causes used
  for Stage D.
