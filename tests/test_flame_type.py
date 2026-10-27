from firesight.flame_type import (
    FLAME_TYPES,
    FireTypePolicy,
    canonical_flame_type,
)
from firesight.types import Box, Detection


def _det(flame_type=None, label="fire"):
    return Detection(label, 0.9, Box(0, 0, 10, 10), flame_type=flame_type)


def test_controlled_flame_can_be_suppressed_without_retraining():
    policy = FireTypePolicy(include=["uncontrolled", "unknown"], exclude=["controlled", "spark", "screen"])
    controlled = _det("controlled")
    uncontrolled = _det("uncontrolled")
    unknown = _det(None)
    kept, suppressed = policy.partition([controlled, uncontrolled, unknown])
    assert uncontrolled in kept
    assert unknown in kept
    assert controlled in suppressed
    assert controlled.meta["suppressed_by_policy"] is True
    assert controlled.meta["policy_tier"] == "log"


def test_policy_can_include_controlled_again():
    policy = FireTypePolicy(include=["uncontrolled", "controlled", "unknown"], exclude=["spark", "screen"])
    det = _det("controlled")
    kept, suppressed = policy.partition([det])
    assert kept == [det]
    assert suppressed == []
    assert det.meta["policy_tier"] == "alarm"


def test_smoke_goes_through_the_same_policy_as_fire():
    policy = FireTypePolicy()
    controlled_smoke = _det("controlled", "smoke")
    uncontrolled_smoke = _det("uncontrolled", "smoke")
    unknown_smoke = _det("unknown", "smoke")
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
    assert reloaded.exclude == {"controlled", "spark_benign", "screen", "unknown"}
    assert reloaded.warning == {"electrical_arc"}


def test_empty_include_survives_a_reload(tmp_path):
    path = tmp_path / "fire_type_policy.json"
    FireTypePolicy(include=[], exclude=["unknown"]).save(path)
    reloaded = FireTypePolicy.load(path, default_include=["uncontrolled", "unknown"], default_exclude=[])
    assert reloaded.include == set()


def test_missing_policy_file_falls_back_to_config_defaults(tmp_path):
    policy = FireTypePolicy.load(tmp_path / "absent.json",
                                 default_include=["uncontrolled", "unknown"],
                                 default_exclude=["controlled", "spark", "screen"],
                                 default_warning=["electrical_arc"])
    assert policy.include == {"uncontrolled", "unknown"}
    assert policy.exclude == {"controlled", "spark_benign", "screen"}
    assert policy.warning == {"electrical_arc"}


def test_taxonomy_is_six_classes_and_legacy_spark_is_an_alias():
    assert FLAME_TYPES == ("uncontrolled", "controlled", "spark_benign", "electrical_arc", "screen", "unknown")
    assert canonical_flame_type("spark") == "spark_benign"
    assert canonical_flame_type("SPARK_BENIGN") == "spark_benign"
    assert canonical_flame_type(None) == "unknown"
    assert canonical_flame_type("") == "unknown"


def test_default_tiers():
    policy = FireTypePolicy()
    assert policy.tier(_det("uncontrolled")) == "alarm"
    assert policy.tier(_det("unknown")) == "alarm"
    assert policy.tier(_det(None)) == "alarm"
    assert policy.tier(_det("controlled")) == "log"
    assert policy.tier(_det("spark_benign")) == "log"
    assert policy.tier(_det("spark")) == "log"
    assert policy.tier(_det("screen")) == "log"
    assert policy.tier(_det("electrical_arc")) == "warning"


def test_warning_tier_is_surfaced_but_never_alarm_worthy():
    policy = FireTypePolicy()
    arc = _det("electrical_arc")
    kept, suppressed = policy.partition([arc])
    assert kept == []
    assert suppressed == [arc]
    assert arc.meta["policy_tier"] == "warning"
    assert arc.meta["warning_by_policy"] is True
    assert "suppressed_by_policy" not in arc.meta


def test_explicit_include_promotes_warning_to_alarm_and_exclude_demotes():
    promoted = FireTypePolicy(include=["uncontrolled", "unknown", "electrical_arc"])
    assert promoted.tier(_det("electrical_arc")) == "alarm"
    demoted = FireTypePolicy(exclude=["controlled", "spark_benign", "screen", "electrical_arc"])
    assert demoted.tier(_det("electrical_arc")) == "log"


def test_warning_tier_persists_and_legacy_file_gets_default_warning(tmp_path):
    path = tmp_path / "fire_type_policy.json"
    FireTypePolicy(include=["uncontrolled"], exclude=["controlled"], warning=["electrical_arc", "screen"]).save(path)
    reloaded = FireTypePolicy.load(path)
    assert reloaded.warning == {"electrical_arc", "screen"}

    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"include": ["uncontrolled"], "exclude": ["spark"]}', encoding="utf-8")
    old = FireTypePolicy.load(legacy)
    assert old.warning == {"electrical_arc"}
    assert old.exclude == {"spark_benign"}
