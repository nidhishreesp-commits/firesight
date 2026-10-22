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
