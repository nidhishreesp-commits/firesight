from firesight.config import AlarmConfig, TemporalConfig
from firesight.temporal import AlarmManager, TemporalConfirmer
from firesight.types import Box, Detection


def det(label="fire", conf=0.8, box=(10, 10, 50, 50)):
    return Detection(label, conf, Box(*box))


def test_n_of_m_confirmation():
    cfg = TemporalConfig(enabled=True, n_of_m={"fire": [3, 5], "smoke": [4, 6]}, max_gap_s=2.0, smoke_growth_min_ratio=0.0)
    confirmer = TemporalConfirmer(cfg)
    confirmed_counts = []
    for i in range(6):
        result = confirmer.update(i, i * 0.2, [det("fire")], size=(100, 100))
        confirmed_counts.append(len(result.confirmed))
    assert confirmed_counts[:2] == [0, 0]
    assert confirmed_counts[2] == 1


def test_smoke_requires_growth_or_confidence():
    cfg = TemporalConfig(enabled=True, n_of_m={"fire": [3, 5], "smoke": [3, 5]}, max_gap_s=2.0, smoke_growth_min_ratio=0.5)
    confirmer = TemporalConfirmer(cfg)
    confirmed = 0
    for i in range(5):
        result = confirmer.update(i, i * 0.2, [det("smoke", conf=0.3)], size=(200, 200))
        confirmed += len(result.confirmed)
    assert confirmed == 0
    confirmer2 = TemporalConfirmer(cfg)
    confirmed2 = 0
    for i in range(5):
        result = confirmer2.update(i, i * 0.2, [det("smoke", conf=0.75)], size=(400, 400))
        confirmed2 += len(result.confirmed)
    assert confirmed2 >= 1


def test_alarm_manager_cooldown_and_escalation():
    cfg = AlarmConfig(min_duration_s=0.0, cooldown_s=100.0, escalation=[{"level": 1, "after_s": 0}, {"level": 2, "after_s": 2}])
    events = []
    manager = AlarmManager(cfg, clear_after_s=1.0, on_event=events.append)
    state = manager.update("cam", 0.0, [det()])
    assert state["active"] is True and state["level"] == 1
    state = manager.update("cam", 3.0, [det()])
    assert state["level"] == 2
    manager.update("cam", 10.0, [])
    assert len(events) == 1
    state = manager.update("cam", 12.0, [det()])
    assert state.get("reason") == "cooldown"

def test_unknown_high_conf_confirms_at_reduced_severity():
    cfg = TemporalConfig(enabled=True, n_of_m={"fire": [3, 5], "smoke": [4, 6]}, max_gap_s=2.0,
                         smoke_growth_min_ratio=0.0, unknown_high_conf=0.6, unknown_auto_alarm_s=60.0)
    confirmer = TemporalConfirmer(cfg)
    results = []
    for i in range(5):
        results.append(confirmer.update(i, i * 0.2, [det("fire", conf=0.9)], size=(100, 100)))
    confirmed = [d for r in results for d in r.confirmed]
    assert confirmed
    assert all(d.meta["unknown_severity"] == "reduced" and d.meta["max_level"] == 1 for d in confirmed)


def test_unknown_low_conf_goes_to_review_queue_then_auto_alarms():
    cfg = TemporalConfig(enabled=True, n_of_m={"fire": [3, 5], "smoke": [4, 6]}, max_gap_s=2.0,
                         smoke_growth_min_ratio=0.0, unknown_high_conf=0.6, unknown_auto_alarm_s=3.0)
    confirmer = TemporalConfirmer(cfg)
    results = []
    for i in range(5):
        results.append(confirmer.update(i, i * 0.5, [det("fire", conf=0.4)], size=(100, 100)))
    assert all(not r.confirmed for r in results)
    assert results[-1].detections[0].meta.get("review_queue") is True
    assert "unknown_severity" not in results[-1].detections[0].meta

    later = confirmer.update(5, 5.0, [det("fire", conf=0.4)], size=(100, 100))
    assert len(later.confirmed) == 1
    assert later.confirmed[0].meta["unknown_severity"] == "auto_alarm"


def test_unknown_escalation_thresholds_are_configurable():
    strict = TemporalConfig(enabled=True, n_of_m={"fire": [1, 2], "smoke": [4, 6]}, max_gap_s=2.0,
                            smoke_growth_min_ratio=0.0, unknown_high_conf=0.95, unknown_auto_alarm_s=600.0)
    confirmer = TemporalConfirmer(strict)
    r = confirmer.update(0, 0.0, [det("fire", conf=0.9)], size=(100, 100))
    assert not r.confirmed
    assert r.detections[0].meta["review_queue"] is True


def test_reduced_severity_caps_alarm_level():
    cfg = AlarmConfig(min_duration_s=0.0, cooldown_s=0.0,
                      escalation=[{"level": 1, "after_s": 0}, {"level": 3, "after_s": 1}])
    manager = AlarmManager(cfg, clear_after_s=1.0)
    capped = det("fire")
    capped.meta["max_level"] = 1
    manager.update("cam", 0.0, [capped])
    state = manager.update("cam", 5.0, [capped])
    assert state["level"] == 1
    manager2 = AlarmManager(cfg, clear_after_s=1.0)
    manager2.update("cam", 0.0, [det("fire")])
    assert manager2.update("cam", 5.0, [det("fire")])["level"] == 3


def test_untracked_low_conf_unknown_is_flagged_for_review():
    cfg = TemporalConfig(enabled=True, n_of_m={"fire": [3, 5], "smoke": [4, 6]}, max_gap_s=2.0,
                         smoke_growth_min_ratio=0.0, unknown_high_conf=0.6, unknown_auto_alarm_s=60.0)
    confirmer = TemporalConfirmer(cfg)
    result = confirmer.update(0, 0.0, [det("fire", conf=0.2)], size=(100, 100))
    assert not result.confirmed
    assert result.detections[0].meta.get("review_queue") is True
    assert result.detections[0].meta.get("review_untracked") is True
