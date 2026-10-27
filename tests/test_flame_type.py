from firesight.flame_type import FireTypePolicy
from firesight.types import Box, Detection


def test_controlled_flame_can_be_suppressed_without_retraining():
    policy = FireTypePolicy(include=["uncontrolled", "unknown"], exclude=["controlled", "spark", "screen"])
    controlled = Detection("fire", 0.9, Box(0, 0, 10, 10), flame_type="controlled")
    uncontrolled = Detection("fire", 0.9, Box(0, 0, 10, 10), flame_type="uncontrolled")
    unknown = Detection("fire", 0.9, Box(0, 0, 10, 10))
    kept, suppressed = policy.partition([controlled, uncontrolled, unknown])
    assert uncontrolled in kept
    assert unknown in kept
    assert controlled in suppressed
    assert controlled.meta["suppressed_by_policy"] is True


def test_policy_can_include_controlled_again():
    policy = FireTypePolicy(include=["uncontrolled", "controlled", "unknown"], exclude=["spark", "screen"])
    det = Detection("fire", 0.9, Box(0, 0, 10, 10), flame_type="controlled")
    kept, suppressed = policy.partition([det])
    assert kept == [det]
    assert suppressed == []


def test_smoke_goes_through_the_same_policy_as_fire():
    policy = FireTypePolicy()
    controlled_smoke = Detection("smoke", 0.9, Box(0, 0, 10, 10), flame_type="controlled")
    uncontrolled_smoke = Detection("smoke", 0.9, Box(0, 0, 10, 10), flame_type="uncontrolled")
    unknown_smoke = Detection("smoke", 0.9, Box(0, 0, 10, 10), flame_type="unknown")
    kept, suppressed = policy.partition([controlled_smoke, uncontrolled_smoke, unknown_smoke])
    assert uncontrolled_smoke in kept
    assert unknown_smoke in kept
    assert controlled_smoke in suppressed
    assert controlled_smoke.meta["suppressed_by_policy"] is True


def test_policy_persists_and_reloads_across_restart(tmp_path):
    path = tmp_path / "fire_type_policy.json"
    policy = FireTypePolicy(include=["uncontrolled"], exclude=["controlled", "spark", "screen", "unknown"])
    policy.save(path)
    reloaded = FireTypePolicy.load(path, default_include=["uncontrolled", "unknown"], default_exclude=[])
    assert reloaded.include == {"uncontrolled"}
    assert reloaded.exclude == {"controlled", "spark", "screen", "unknown"}


def test_empty_include_survives_a_reload(tmp_path):
    path = tmp_path / "fire_type_policy.json"
    FireTypePolicy(include=[], exclude=["unknown"]).save(path)
    reloaded = FireTypePolicy.load(path, default_include=["uncontrolled", "unknown"], default_exclude=[])
    assert reloaded.include == set()


def test_missing_policy_file_falls_back_to_config_defaults(tmp_path):
    policy = FireTypePolicy.load(tmp_path / "absent.json",
                                 default_include=["uncontrolled", "unknown"],
                                 default_exclude=["controlled", "spark", "screen"])
    assert policy.include == {"uncontrolled", "unknown"}
    assert policy.exclude == {"controlled", "spark", "screen"}
