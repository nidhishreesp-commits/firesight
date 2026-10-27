import json

import pytest
import yaml

from firesight.api.state import AppState, CameraState, CameraWorker
from firesight.config import load_config

CAM_YAML = """cameras:
  - id: cam-01
    name: Warehouse A
    uri: "0"
    enabled: false
    roi: []
    conf:
      fire: 0.4
    temporal:
      n: 3
      m: 5
    alarm:
      min_duration_s: 1.0
      cooldown_s: 30.0
  - id: cam-02
    name: Loading Bay
    uri: "1"
    enabled: false
    roi: []
    conf: {}
    temporal: {n: 4, m: 6}
"""


def make_state(tmp_path):
    cam = tmp_path / "cameras.yaml"
    cam.write_text(CAM_YAML, encoding="utf-8")
    audit = tmp_path / "audit" / "camera_config.jsonl"
    st = AppState(load_config(), detector=object(), cameras_path=cam, audit_path=audit)
    st.load_cameras()
    return st, cam, audit


def test_per_camera_config_persists_across_restart(tmp_path):
    st, cam, audit = make_state(tmp_path)
    before = st.camera_config("cam-01")
    after = st.update_camera_config(
        "cam-01",
        {"temporal_n": 5, "temporal_m": 8, "alarm_cooldown_s": 45.0,
         "conf": {"fire": 0.55, "smoke": 0.35},
         "roi": [[0.1, 0.1, 0.9, 0.9]]},
        actor="tester",
    )
    assert after["temporal"] == {"n": 5, "m": 8}
    assert after["alarm"]["cooldown_s"] == 45.0
    assert after["conf"] == {"fire": 0.55, "smoke": 0.35}

    saved = yaml.safe_load(cam.read_text(encoding="utf-8"))
    cam01 = [c for c in saved["cameras"] if c["id"] == "cam-01"][0]
    assert cam01["temporal"] == {"n": 5, "m": 8}
    assert cam01["alarm"]["cooldown_s"] == 45.0

    restarted = AppState(load_config(), detector=object(), cameras_path=cam, audit_path=audit)
    restarted.load_cameras()
    reloaded = restarted.camera_config("cam-01")
    assert reloaded["temporal"] == {"n": 5, "m": 8}
    assert reloaded["alarm"]["cooldown_s"] == 45.0
    assert reloaded["roi"] == [[0.1, 0.1, 0.9, 0.9]]
    assert restarted.camera_config("cam-02")["temporal"] == {"n": 4, "m": 6}

    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["actor"] == "tester"
    assert rows[0]["camera_id"] == "cam-01"
    assert rows[0]["before"] == before
    assert rows[0]["after"] == after
    assert set(rows[0]["changed"]) >= {"conf", "roi", "temporal", "alarm"}


def test_audit_log_appends_one_row_per_change(tmp_path):
    st, _, audit = make_state(tmp_path)
    st.update_camera_config("cam-01", {"temporal_n": 4}, actor="alice")
    st.update_camera_config("cam-01", {"temporal_n": 5}, actor="bob")
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["actor"] for r in rows] == ["alice", "bob"]
    assert rows[1]["before"]["temporal"] == {"n": 4, "m": 5}
    assert rows[1]["after"]["temporal"] == {"n": 5, "m": 5}


@pytest.mark.parametrize("updates", [
    {"temporal_n": 9, "temporal_m": 5},
    {"temporal_n": 0},
    {"conf": {"fire": 1.5}},
    {"roi": [[0.1, 0.2, 0.3]]},
    {"alarm_cooldown_s": -1},
])
def test_invalid_camera_config_is_rejected(tmp_path, updates):
    st, cam, audit = make_state(tmp_path)
    original = cam.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        st.update_camera_config("cam-01", updates, actor="tester")
    assert cam.read_text(encoding="utf-8") == original
    assert not audit.exists()


def test_unknown_camera_raises_key_error(tmp_path):
    st, _, _ = make_state(tmp_path)
    with pytest.raises(KeyError):
        st.update_camera_config("cam-99", {"temporal_n": 4}, actor="tester")
    with pytest.raises(KeyError):
        st.camera_config("cam-99")


def test_worker_honours_per_camera_temporal_and_alarm():
    cfg = load_config()
    cam = CameraState(camera_id="c1", name="c1", uri="0", enabled=False,
                      n_of_m=[6, 9], alarm_cfg={"min_duration_s": 2.5, "cooldown_s": 60.0})
    worker = CameraWorker(cam, detector=object(), cfg=cfg)
    assert worker.confirmer.cfg.n_of_m["fire"] == [6, 9]
    assert worker.confirmer.cfg.n_of_m["smoke"] == cfg.temporal.n_of_m["smoke"]
    assert worker.alarm.cfg.min_duration_s == 2.5
    assert worker.alarm.cfg.cooldown_s == 60.0

    cam.n_of_m = [2, 4]
    cam.alarm_cfg = {"cooldown_s": 10.0}
    worker.apply_config()
    assert worker.confirmer.cfg.n_of_m["fire"] == [2, 4]
    assert worker.alarm.cfg.cooldown_s == 10.0
    assert worker.alarm.cfg.min_duration_s == cfg.alarm.min_duration_s
