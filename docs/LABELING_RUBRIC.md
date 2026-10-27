# Labeling rubric (one page)

Six classes: **uncontrolled · controlled · spark_benign · electrical_arc · screen · unknown**
Ask one question: *what would this do with no human intervention?* Label what it **is**, not how alarming it looks.

## Class table

| class | means | alarm tier | examples (≥10 each) |
|---|---|---|---|
| **uncontrolled** | Combustion that is spreading or would spread; you cannot stop it by removing the ignition source. Includes its **smoke**. | **ALARM** | building fire · roof fire · room interior alight · forest/brush/wildfire · grass/field fire · vehicle/car/truck fire · tyre fire · rubbish/bin/dumpster fire · warehouse/plant fire · oil/gas flare-up · burning victim's belongings · power transformer **with open flame** |
| **controlled** | Deliberate, attended, contained. Ends when the fuel/ignition stops. **Smoke inherits this class.** | log only | candle · lighter · match · gas-stove burner · fireplace · campfire in a ring/pit · BBQ/grill · controlled agricultural burn · incense · cigarette/cigar smoke · wood stove · hob flame · gas boiler pilot · tiki torch |
| **spark_benign** | Brief hot particles, **no sustained flame**, no spread. Non-electrical cause. | log only | welding · angle grinder · metal cutting · plasma cutter · sparkler · fireworks burst · flint/ferro rod · grinding wheel · rail grinding · steel chipping · anvil/forging sparks · car exhaust backfire sparks |
| **electrical_arc** | Arcing/short-circuit at electrical equipment. Blue-white flash or sparking wire, **no sustained flame**. | **WARNING** | short circuit · sparking socket/outlet · overloaded power strip · burning/flickering wiring · arcing breaker panel · frayed cable sparks · motor brush arcing · transformer arc · Christmas-light short · junction-box flash · battery short sparks |
| **screen** | Fire **depicted on a display** — pixels, no combustion. | log only | TV news wildfire · monitor fireplace screensaver · cinema fire scene · phone fire video · LED video wall · stage flame projection · laptop campfire clip · fireplace app on tablet · digital signage embers · monitor showing explosion · screen-saver lava |
| **unknown** | Context is insufficient to be sure. **Prefer this over a guess.** | per configured rule | cropped/edge-off flame · no visible source · occluded fire · heavy blur/low res · too small to judge · night scene with only glow · two readings equally plausible |

## Presence labels (decide before the class)

- `has_fire = 1` only for **physical** combustion with visible flame/glow.
- `has_smoke = 1` for real smoke/soot plumes from combustion.
- **Not smoke:** steam, water vapour, dust (construction, flour, cement), fog/haze, vehicle exhaust, breath, dry-ice fog, powder → these are **negatives** (`has_fire=0, has_smoke=0`).
- Depicted fire on a screen: `has_fire=0, has_smoke=0`, class = `screen`.
- If smoke is present without visible flame → `has_smoke=1, has_fire=0`, class from the source.

## Smoke: controlled vs uncontrolled

Smoke always inherits its source.

| controlled smoke | uncontrolled smoke |
|---|---|
| cooking fumes from a lit hob · chimney from a working fireplace · incense · cigarette plume · BBQ smoke · greenhouse/ag burn smoke · steam from a pot | smoke from a building/roof · vehicle fire · vegetation/wildfire · burning bin · electrical panel with flame · industrial/chemical fire · black plume from tyres or bitumen |

If you **cannot see the source** and cannot tell whether it is a controlled process, use **unknown**.

## Decision order (5 steps, stop at the first that fits)

1. Combustion on a **display**? → `screen` (and presence 0/0).
2. Real flame/glow or real smoke present? → set `has_fire` / `has_smoke` (dust/steam/exhaust = neither).
3. Source **visible, intentional, attended, contained**? → `controlled`.
4. Only **brief sparks, no flame**? → electrical equipment → `electrical_arc`; otherwise → `spark_benign`.
5. Spreading / unbounded / unstoppable by removing the ignition? → `uncontrolled`. Still unsure → **`unknown`**, and set `ambiguous=1`.

## Boxes

- Draw the tightest box that contains the visible flame **or** the smoke region you mean; do not include unrelated frame edges.
- Multiple disjoint regions → one box each.
- Smoke that fades off-frame → box the dense part only, not the whole sky.
- Store the box **plus** a 2–3× context box when the class depends on surroundings (stove, TV, welding rig) — the context classifier needs it.

## Quality control

- **Audit:** random 50–100 images per source/class get a human check. ≥90% correct → accept the batch; below → fix proposer settings and re-sample. Report the error rate.
- **Consistency:** re-label a random 100 after a ≥1-week delay; report agreement (Cohen's κ). Target κ ≥ 0.80; resolve every disagreement and add a line here if a rule was unclear.
- **Cluster review:** accept by default, click only the wrong ones (~40 per grid); one-by-one review only for low-confidence or proposer-disagreeing items.
- **Time:** expected hours are stated *before* any labeling task; sessions stop at 45 minutes and ask first.
- **Bench:** `bench_R` and `testbench100` are locked — never edit them. The bench keeps its legacy single `spark` value (37 images); the spark_benign / electrical_arc split is recorded in a **side file** only.
