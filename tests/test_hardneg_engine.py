import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hardneg_causes import CATEGORIES, EASY_CAUSES, HELD_OUT_CAUSES, QUERIES  # noqa: E402
from collect_stage_a_images import allowed_license  # noqa: E402
from collect_commons_categories import CATEGORY_MAP  # noqa: E402


def test_every_cause_has_description_queries_and_category_mapping():
    for cause in CATEGORIES:
        assert CATEGORIES[cause].strip()
        assert QUERIES.get(cause), f"missing search queries for {cause}"
        assert CATEGORY_MAP.get(cause), f"missing Commons category mapping for {cause}"


def test_coverage_includes_all_named_lookalikes():
    required = {
        "sun_glare_sky", "sunset_sunrise", "warm_artificial_light", "traffic_light",
        "vehicle_lights", "haze_cloud_fog_mist", "dust_plaster", "screen_fire",
        "reflection_lens_flare", "orange_ppe", "skin_person", "cave_warm_rock",
        "smoke_steam_nonfire", "orange_vegetation", "neon_lights", "stage_chandelier",
        "snow_white_overexposed", "exhaust", "camera_artifacts",
        "ordinary_indoor", "ordinary_outdoor",
    }
    assert required <= set(CATEGORIES)
    assert len(CATEGORIES) >= 20


def test_at_least_two_whole_cause_holdouts():
    assert len(HELD_OUT_CAUSES) >= 2
    assert HELD_OUT_CAUSES <= set(CATEGORIES)
    assert EASY_CAUSES <= set(CATEGORIES)


@pytest.mark.parametrize("name", [
    "CC0", "Public domain", "CC BY 4.0", "CC BY-SA 4.0", "CC BY 2.0", "PD-USGov",
])
def test_allowed_licences(name):
    assert allowed_license(name)


@pytest.mark.parametrize("name", [
    "CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-ND 4.0", "All rights reserved", "",
])
def test_rejected_licences(name):
    assert not allowed_license(name)
