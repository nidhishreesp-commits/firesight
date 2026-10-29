import numpy as np
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter


def frame(value=120):
    return np.full((100, 100, 3), value, dtype=np.uint8)


def det(box=(10, 10, 50, 50), conf=0.8, track=1):
    d = Detection("smoke", conf, Box(*box))
    d.track_id = track
    return d


def test_static_detection_is_routed_to_review():
    f = VideoFalseAlarmFilter(FilterConfig(min_frames=4, window=10))
    img = frame()
    review_seen = False
    for i in range(6):
        kept, review, suppressed = f.update("cam", i * 0.2, img, [det(conf=0.5)])
        if review:
            review_seen = True
            assert review[0].meta["static_scene"] is True
            assert review[0].meta["review_queue"] is True
    assert review_seen


def test_growing_detection_stays():
    f = VideoFalseAlarmFilter(FilterConfig(min_frames=4, growth_min_ratio=0.15))
    img = frame()
    kept_all = []
    for i in range(6):
        size = 20 + i * 10
        kept, review, _ = f.update("cam", i * 0.2, img, [det(box=(10, 10, 10 + size, 10 + size), conf=0.7)])
        kept_all += kept
    assert len(kept_all) == 6


def test_flicker_prevents_static_verdict():
    f = VideoFalseAlarmFilter(FilterConfig(min_frames=4, flicker_min_std=0.04))
    img = frame()
    confs = [0.3, 0.9, 0.3, 0.9, 0.3, 0.9]
    review_seen = False
    for i, c in enumerate(confs):
        _, review, _ = f.update("cam", i * 0.2, img, [det(conf=c)])
        review_seen |= bool(review)
    assert not review_seen


def test_reviewer_dismissal_suppresses_repeats():
    f = VideoFalseAlarmFilter(FilterConfig(min_dismissals=1))
    img = frame()
    f.record_dismissal("cam", img, (10, 10, 50, 50))
    f.record_dismissal("cam", img, (10, 10, 50, 50))
    kept, review, suppressed = f.update("cam", 1.0, img, [det()])
    assert suppressed and suppressed[0].meta["video_suppressed"] == "reviewer_dismissal"
    assert not kept
