"""Complete hard-negative cause taxonomy for the FireSight detector.

This is deliberately broader than the v1 taxonomy.  It covers every lookalike named in the
programme brief plus the families measured in docs/FP_TAXONOMY.md and
docs/HARD_NEGATIVE_COLLECTION.md:

  sun/sky glare, backlit person/vehicle, sunset, warm lights, traffic lights, street lamps,
  vehicle lights, haze/cloud/fog/mist, dust/plaster, screens, reflections/lens flare,
  orange PPE, skin, cave/warm rock, steam, orange vegetation, neon, stage/chandelier,
  snow/overexposed white, exhaust, camera artefacts, ordinary indoor/outdoor.

Two causes are deliberately held out of training so Stage D can measure generalisation:
  traffic_light, sun_glare_sky
"""
from __future__ import annotations

# cause -> one-line description used for CLIP verification and documentation
CATEGORIES: dict[str, str] = {
    "sun_glare_sky": "sun glare, lens flare from the sun, a bright overexposed sky, or a person or vehicle backlit by the sun",
    "sunset_sunrise": "a sunset or sunrise with an orange, red or yellow sky",
    "warm_artificial_light": "a street lamp, lantern, warm light bulb or city lights glowing at night",
    "traffic_light": "a red or orange traffic light or railway signal",
    "vehicle_lights": "bright car headlights, brake lights, tail lights or light trails at night",
    "haze_cloud_fog_mist": "clouds, haze, fog, mist, overcast sky or a hazy horizon",
    "dust_plaster": "construction dust, plaster dust, sawdust, sandblasting or a dust cloud",
    "screen_fire": "a television, monitor or phone screen showing fire, orange graphics or a poster",
    "reflection_lens_flare": "a reflection on water, wet road, glass or metal, or lens flare and glare",
    "orange_ppe": "an orange high-visibility vest, orange workwear, a traffic cone or a safety helmet",
    "skin_person": "a close-up portrait of a person's face or skin",
    "cave_warm_rock": "a cave interior or red-orange rock formation lit warmly",
    "smoke_steam_nonfire": "steam or water vapour from a cooling tower, kettle, geyser or vent, with no fire",
    "orange_vegetation": "orange or red flowers, autumn leaves or orange foliage",
    "neon_lights": "red or orange neon signs and lamps at night",
    "stage_chandelier": "a chandelier or stage, concert and festival lighting",
    "snow_white_overexposed": "snowfall, a blizzard, a white sky or an overexposed white wall",
    "exhaust": "vehicle exhaust fumes or diesel exhaust, with no fire",
    "camera_artifacts": "motion blur, lens dirt, rain on the lens or sensor noise",
    "ordinary_indoor": "a normal office, warehouse, hallway or shop interior with no fire",
    "ordinary_outdoor": "a normal street, building, car park, landscape or park with no fire",
}

# Wikimedia Commons search queries.  Deliberately avoid queries that overwhelmingly return
# real fire (candle, bonfire, wildfire) and avoid combustion smoke for the steam family.
QUERIES: dict[str, list[str]] = {
    "sun_glare_sky": [
        "sun glare", "lens flare sun", "backlit portrait", "contre-jour photography",
        "silhouette person sunset", "sun behind clouds", "overexposed sky", "sun glare road",
        "vehicle backlit sunset", "bright sky silhouette",
    ],
    "sunset_sunrise": [
        "sunset", "sunrise", "orange sky", "red sky", "dusk sky", "sunset sea",
        "sunset clouds", "orange sunset landscape",
    ],
    "warm_artificial_light": [
        "street lamp at night", "lantern", "warm light lamp", "city lights night",
        "light bulb warm", "lamp post night", "street lighting", "illuminated building night",
    ],
    "traffic_light": [
        "traffic light", "traffic signal", "red traffic light", "orange traffic light",
        "railway signal", "pedestrian signal", "road signal light",
    ],
    "vehicle_lights": [
        "car headlights", "tail lights", "brake lights", "light trails road",
        "car light trail night", "night traffic lights",
    ],
    "haze_cloud_fog_mist": [
        "clouds", "cumulus clouds", "fog", "mist", "haze", "overcast sky",
        "cloudscape", "foggy landscape", "hazy horizon",
    ],
    "dust_plaster": [
        "construction dust", "plaster dust", "sawdust", "sandblasting", "dust cloud",
        "demolition dust", "agricultural dust", "dust storm",
    ],
    "screen_fire": [
        "television screen", "computer monitor", "video game screenshot", "movie poster",
        "digital display orange", "screen showing fire", "arcade screen",
    ],
    "reflection_lens_flare": [
        "water reflection", "wet road reflection", "glass reflection", "lens flare",
        "sun glare water", "metal reflection", "mirror reflection",
    ],
    "orange_ppe": [
        "orange safety vest", "traffic cone", "high visibility vest", "construction helmet",
        "orange workwear", "road worker", "hi-vis jacket",
    ],
    "skin_person": [
        "portrait photograph", "selfie", "close-up face", "person portrait",
    ],
    "cave_warm_rock": [
        "cave interior", "red rock canyon", "slot canyon", "sandstone formation",
        "cave", "rock formation sunset",
    ],
    "smoke_steam_nonfire": [
        "steam", "water vapour", "kettle steam", "cooling tower", "geothermal steam",
        "steam vent", "steam locomotive", "hot spring steam",
    ],
    "orange_vegetation": [
        "orange flowers", "autumn leaves", "autumn foliage", "red flowers", "orange foliage",
    ],
    "neon_lights": [
        "neon sign", "red neon sign", "orange neon light", "neon lights night",
    ],
    "stage_chandelier": [
        "chandelier", "stage lighting", "concert stage lights", "festival lights",
    ],
    "snow_white_overexposed": [
        "snowfall", "blizzard", "snow landscape", "blowing snow", "overexposed photo",
        "white sky", "bright white wall",
    ],
    "exhaust": [
        "vehicle exhaust", "diesel exhaust", "exhaust fumes", "car exhaust pipe",
    ],
    "camera_artifacts": [
        "motion blur", "lens dirt", "rain on lens", "sensor noise",
    ],
    "ordinary_indoor": [
        "office interior", "warehouse interior", "indoor hallway", "shopping mall interior",
    ],
    "ordinary_outdoor": [
        "street scene", "warehouse exterior", "office building", "landscape", "park",
        "car park",
    ],
}

# Causes whose images are never trained on.  Stage D measures them separately.
HELD_OUT_CAUSES: set[str] = {"traffic_light", "sun_glare_sky"}

# Causes that are especially prone to containing real smoke/fire; always route highs to review.
SENSITIVE_CAUSES: set[str] = {"screen_fire", "smoke_steam_nonfire", "exhaust", "stage_chandelier"}

# Easy-background families (P50 should not fire, or fires only weakly).
EASY_CAUSES: set[str] = {"ordinary_indoor", "ordinary_outdoor"}

TIER1: set[str] = {
    "haze_cloud_fog_mist", "dust_plaster", "sunset_sunrise", "sun_glare_sky",
    "warm_artificial_light", "reflection_lens_flare",
}
