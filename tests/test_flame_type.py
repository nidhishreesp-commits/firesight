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
