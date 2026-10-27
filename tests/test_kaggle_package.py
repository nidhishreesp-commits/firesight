import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kaggle_package as kp  # noqa: E402


def row(rid, source="fasdd", has_fire=1, has_smoke=0, phash="0000000000000000", path=None):
    return {"id": rid, "source": source, "phash": phash,
            "path": path or f"data/processed/images/{source}/train/{rid.split(':')[-1]}.jpg",
            "labels": ([{"label": "fire"}] if has_fire else []) + ([{"label": "smoke"}] if has_smoke else [])}


def exam(ids=(), phash=None, test_ids=(), tb_ids=()):
    return {"bench_ids": set(ids), "bench_phash": dict(phash or {}), "bench_files": set(),
            "tb_ids": set(tb_ids), "tb_phash": {}, "test_ids": set(test_ids),
            "test_phash": {}, "test_paths": set()}


def test_stratified_pick_keeps_class_ratio_at_every_fraction():
    rows = ([row(f"fasdd:train:f{i}", has_fire=1, has_smoke=0) for i in range(60)]
            + [row(f"fasdd:train:s{i}", has_fire=0, has_smoke=1) for i in range(30)]
            + [row(f"fasdd:train:b{i}", has_fire=1, has_smoke=1) for i in range(10)])
    import random
    for frac in (0.1, 0.25, 0.5, 1.0):
        picked = kp.stratified_pick(rows, frac, random.Random(1))
        counts = {}
        for r in picked:
            key = kp.stratum(r)
            counts[key] = counts.get(key, 0) + 1
        assert counts["f1s0"] == max(1, round(60 * frac))
        assert counts["f0s1"] == max(1, round(30 * frac))
        assert counts["f1s1"] == max(1, round(10 * frac))


def test_leakage_report_passes_for_clean_package():
    package = [row("fasdd:train:a", phash="1111111111111111"), row("dfire:train:b", source="dfire")]
    report = kp.leakage_report(package, exam(ids={"benchR:x"}, phash={"benchR:x": "ffffffffffffffff"}))
    assert report["verdict"] == "PASS"
    assert report["id_overlap"] == {"bench_R": 0, "testbench100": 0, "TEST": 0}


def test_leakage_report_fails_on_id_overlap():
    package = [row("benchR:x")]
    report = kp.leakage_report(package, exam(ids={"benchR:x"}))
    assert report["verdict"] == "FAIL"
    assert report["id_overlap"]["bench_R"] == 1


def test_leakage_report_fails_on_test_and_testbench_overlap():
    report = kp.leakage_report([row("dfire:test:1")], exam(test_ids={"dfire:test:1"}))
    assert report["verdict"] == "FAIL"
    report = kp.leakage_report([row("dfire:test:2")], exam(tb_ids={"dfire:test:2"}))
    assert report["verdict"] == "FAIL"


def test_leakage_report_fails_on_phash_near_duplicate():
    package = [row("fasdd:train:a", phash="0000000000000000")]
    # one bit apart
    report = kp.leakage_report(package, exam(phash={"benchR:y": "0000000000000001"}))
    assert report["verdict"] == "FAIL"
    assert report["phash_near_duplicates_hamming_le_2"] == 1


def test_leakage_report_fails_on_proprietary_path():
    package = [row("fasdd:train:a", path="Sample_videos/ppe_2/frame.jpg")]
    report = kp.leakage_report(package, exam())
    assert report["verdict"] == "FAIL"
    assert report["proprietary_paths_found"] == 1


def test_only_public_sources_are_allowed():
    assert set(kp.ALLOWED_SOURCES) == {"dfire", "fasdd"}
    assert set(kp.EXCLUDED_SOURCES) == {"local_seed", "local_negative"}
