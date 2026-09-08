# Part B Implementation Plan — Reflectance, Albedo, Sensor Model & Pair Generation
### Renderer for ISRO PS 26166 · agent-ready spec for Antigravity

**Scope:** `context.md` §6.3, §6.5–6.7, plus §6.10's radiometric half and the §7 `P1` pairing node.
**Depends on:** Part A (§6.1–6.2, §6.4) — complete, validated, delivered.
**Reads with:** `context.md` (bible), `critique.md` (F1–F24 / Q1–Q27), `resolutions.md` (patch log — **overrides `context.md` where they disagree**).

---

## 0. How to drive this with Antigravity

This document is written to be handed to an agent as a task spec, not read as prose. Structure it in the workspace as:

```
/renderer
  AGENTS.md                  ← §1 + §2 + §9 of this doc, verbatim
  docs/part_b_plan.md        ← this file
  configs/render.yaml        ← §7
  src/renderer/…             ← §5 modules
  tests/…                    ← §6
  fixtures/                  ← §4.3 synthetic Part-A stand-ins
```

Kickoff prompt to paste:

> Read `docs/part_b_plan.md` and `AGENTS.md`. Implement tasks T0–T4 only. Do not touch anything under `src/renderer/geometry/` — that is Part A and is frozen. Every task has acceptance criteria under §8; write the test **before** the implementation and show me the failing test first. Stop after T4 and report the throughput number from T4.3 before starting T5.

Three rules that matter more than the rest, because they are the ones an agent will silently break:

1. **The interface in §4 is frozen.** If a task seems to need a change to it, stop and ask. A drifting interface is the exact failure mode the A/B split exists to prevent (§12).
2. **Every physical constant, range and magic number lives in `configs/render.yaml`**, never inline. §6.7 requires the randomisation ranges to be auditable, and F19/F21 exist because untested numbers got embedded in reasoning.
3. **Assumption-tier comments are mandatory.** Any constant not traceable to a source gets `# A-TIER ASSUMPTION: <what, why, what breaks if wrong>`. These get harvested into the slide deck and the report. Do not let the agent quietly pick a plausible number.

---

## 1. What Part B is, in one paragraph

Part A turns a DEM into geometry: a normal field `n(x,y)` and a horizon map that answers "is this pixel in shadow at sun azimuth A, elevation e?" in O(1). Part B turns that geometry into **images that a matcher can train on** — applying a reflectance law, draping albedo, running the result through a camera, and emitting labelled pairs. The deliverable is not "a pretty render." The deliverable is a directory of image pairs with exact pixel correspondences and a manifest, plus the render-vs-real evidence panel (V7) that proves the pairs are worth training on.

**Success is measured by §6.10 and Milestone 1b, not by how photoreal the output looks.** F6 is explicit: a render can look obviously synthetic to a human and still teach correct invariance, and a photoreal render can be useless if the shadow geometry is wrong. Do not spend time on visual polish.

---

## 2. Decisions — the two open review items, closed

The draft plan flagged two items for user review. Both are decidable now from `resolutions.md`; neither needs to block.

### 2.1 View vector — **nadir, `v = [0,0,1]`, and do not parse roll/pitch**

Correct for v1, for four reasons:

- The entire 20K-pair training set is **orthorectified** (§6.9). Part A's output already lives in a map-projected DEM grid; there is no perspective camera in the pipeline at all. §6.8 restricts Blender/Cycles to the perspective set, which is explicitly out of scope.
- Georgakis & Ansar used ortho for their 25K-pair set and reserved a perspective camera for a separate, smaller dataset. `context.md` §6.4(c) tells you to follow that split.
- **Nadir does not degenerate the model.** µ = **n·v** = `n_z` under nadir, and `n_z = 1/√(1+p²+q²) < 1` on any slope. Emission angle still varies across the tile, so the Lommel-Seeliger term does real work. Someone will raise this — have the answer ready.
- It buys a large simplification: with `v = [0,0,1]`, `cos α = s·v = sin(e_sun)`, so **phase angle is fully determined by sun elevation**: `α = 90° − e_sun`. `L(α)` collapses to `L(e_sun)`. Implement the general form anyway (§5.1) but know the identity holds, and **assert it in a test** — it is a free correctness check on your angle conventions.

**But:** keep `v` as a first-class function argument, shape-broadcastable to `(H,W,3)`, defaulting to nadir. Off-nadir then costs one config line later instead of an API break. And there is **one case where nadir is wrong**: the §6.10 validation render must match the real image's actual geometry, so parse `emission_angle` from that product's PDS4 label and construct `v` from it for that run only. Everything else stays nadir.

### 2.2 Albedo — **real drape is the default, procedural is the fallback**

This is already decided in `resolutions.md` **F16**, and it inverts the draft plan's framing. Do not treat the drape as a stretch goal.

Three sources, in priority order:

| Source | Use | Notes |
|---|---|---|
| **Real LROC NAC / OHRC ortho, photometrically flattened** | **Default** at any site with a co-located DTM (Vikram, §8.3) | Real boulders, real small-crater texture. Best defence against sim-to-real gap. |
| **Procedural fractal albedo** | Fallback for tiles with no co-located image — i.e. the bulk SLDEM2015 build | Not "worse," just lower-information |
| LROC WAC 100 m mosaic | **Low-frequency albedo level only** (mare/highland offset) | **Not a texture source.** At 100 m it is coarser than a 20 m/px render — it can set the DC level and nothing else. Do not let the agent use it as a drape. |

Two things must be built into the drape from the start:

- **Holes.** Pixels shadowed in the source image have no recoverable albedo. Inpaint, and **carry an explicit `albedo_valid` mask forward** into the pairing stage so those pixels can be excluded from the training GT. §6.5 names the holes; the mask is what makes them harmless instead of silently poisoning the labels.
- **The F17 cancellation asymmetry.** Â = I_real / R_LL(i₀,e₀,α₀), then re-render as Â · R_LL(i₁,e₁,α₁). Reflectance-model error **partially cancels at the source geometry and does not cancel at the target geometry** — so drape fidelity degrades as Δ(sun angle) from the source grows. The drape is most accurate exactly where the problem is easiest. This is a known, stated limitation, and T6.2 measures it rather than assuming it away.

---

## 3. Corrections to the draft plan

Six issues in `implementation_plan.md` as written. The first two would produce a wrong renderer that still runs.

**C1 — The reflectance formula is missing a factor of 2.** The draft has

> `I = A × [ L(α)·µ₀/(µ₀+µ) + (1−L(α))·µ₀ ]`

`context.md` §6.3 has `R_LL = A · [ 2·L(α)·µ₀/(µ₀+µ) + (1−L(α))·µ₀ ]`. The 2 is not cosmetic — it is what makes both limits normalise consistently. Check at `i = e = 0` (µ₀ = µ = 1): correct form gives `L·1 + (1−L)·1 = 1` for **any** L. The draft's form gives `0.5L + (1−L)`, which is discontinuous in L at the same geometry — pure Lommel-Seeliger would come out at half the brightness of pure Lambert on flat ground at zenith sun. Ablation C (Lambert vs Lunar-Lambert) would then measure a brightness scaling, not a shading law. **Add the L-independence-at-normal-incidence check as a unit test** (§6, UT-3).

**C2 — "Maximum radiance at phase angle 0 (opposition surge)" is the wrong unit test.** Lunar-Lambertian has **no opposition surge term** — the surge lives in Hapke's B(α), which §6.3 puts explicitly out of scope. Writing this test forces one of two bad outcomes: it passes vacuously, or the agent "fixes" it by adding an opposition term you decided not to model. Replace with the limit and monotonicity tests in §6 (UT-1…UT-6).

**C3 — The sensor chain omits pushbroom line-jitter and does not specify linear space.** §6.6 lists jitter as step 5, and the ASP docs report OHRC jitter on the order of the GSD (~0.25 m) — a real, documented effect at exactly the scale you claim sub-pixel accuracy at. Also: **noise must be injected in linear electron space, PSF must be applied before shot noise.** Blur is optical and happens before detection; blurring after adding noise spatially correlates the noise and makes the SNR test in §6 pass while the physics is wrong. Ordering is specified in §5.3.

**C4 — Domain-randomisation ranges contradict §6.7.** Draft says sun elevation [5°, 85°] and PSF σ [0.5, 2.0]. §6.7 says elevation 20–70° (equatorial) / 0–6° (polar) and PSF σ 0.4–1.0. The elevation range matters: 5–85° is neither regime, and per §6.2 these are *different problems* — the plan must state which one it targets (**equatorial for the core build**, per §6.2's recommendation, with polar as stretch). Use §6.7's table. It goes in config, not code.

**C5 — The DR module randomises appearance but not pair geometry.** The draft's `domain_randomization.py` covers lighting, albedo and sensor. §6.7's table also requires **albedo scale ×0.7–1.4, additive fractal albedo noise σ 5–15%, terrain roughness amplitude, scale ratio 1–20×, and rotation ±180°.** The last two are the ones that actually create a *scale-and-rotation-invariant* training set — without them you train illumination invariance only, and F3 is the whole reason §5.3 exists.

**C6 — Nothing emits training pairs.** §7's flowchart runs `Sensor model → P1[Anchor/observation pairing + exact GT correspondences] → fine-tune`. The draft ends at domain randomisation. **P1 is unowned in the current A/B split** — it is not in §6.3 or §6.5–6.7, so on a literal reading neither renderer role has it, and the matcher person is presumably assuming it arrives with the data. Claim it in writing today (§5.4 specs it; it is a few hours' work once the sensor output exists) or get it assigned. Do not discover this on day 10.

---

## 4. Frozen interface — Part A ⇄ Part B

**Build this first, before any physics.** This is the highest-value structural item in the plan: with a frozen contract and synthetic fixtures, Part B develops at full speed without ever blocking on Part A, which is the stated purpose of the split.

### 4.1 What Part A hands over

```python
@dataclass(frozen=True)
class TerrainTile:
    normals:      np.ndarray  # (H, W, 3) float32, unit-length, +Z up, map-grid frame
    elevation:    np.ndarray  # (H, W)    float32, METRES (not km — Part A's fix)
    horizon:      np.ndarray  # (H, W, K) float32, max horizon elevation angle in RADIANS,
                              #                    per azimuth bin, bin k centred at 2πk/K
    valid:        np.ndarray  # (H, W)    bool, False = nodata / void-fill / halo
    lat0, lon0:   float                   # tile origin, degrees
    dlat, dlon:   float                   # degrees per pixel
    gsd_m:        float                   # ground sample distance, metres (at tile centre lat)
    K:            int                     # number of azimuth bins (16 or 32)
    halo_px:      int                     # halo width; strip before emitting pairs
```

```python
def cast_shadow(tile: TerrainTile, sun_az_rad: float, sun_el_rad: float) -> np.ndarray:
    """(H, W) bool. True = in shadow. O(1) per pixel via horizon-map lookup."""
```

### 4.2 Conventions — pin these in `AGENTS.md`, they are the classic silent-failure source

- Azimuth: **clockwise from north**, radians. Sun vector `s = (sin A·cos e, cos A·cos e, sin e)` (§6.2).
- Normals: `n = (−∂h/∂x, −∂h/∂y, 1)` normalised, Horn 3×3, **latitude-corrected spacing** `Δx = Δlon·(π/180)·R·cos(lat)`, `R = 1_737_400 m`.
- Array indexing: `array[row, col]` = `[y, x]`; `+y` is **south** (increasing row = decreasing latitude) in the equirectangular grid. Every geometry bug in this class comes from a sign flip here.
- All angles internally in **radians**; degrees only at the config and label-parsing boundary.

### 4.3 Fixtures — the unblocking move

Write `tests/fixtures/synthetic_terrain.py` producing `TerrainTile` instances with **analytically known** answers, so every Part B test runs with Part A absent:

| Fixture | Terrain | Known answer |
|---|---|---|
| `flat_plane` | h = 0 | n = [0,0,1] everywhere; no shadow at any e > 0; R = A·µ₀ exactly |
| `constant_slope(θ, φ)` | tilted plane | closed-form n; shadow ⟺ e < θ when sun azimuth opposes dip |
| `single_cone(h, r)` | axisymmetric cone | shadow length = h/tan(e), analytically checkable |
| `gaussian_bump` | smooth C² bump | smooth normals; tests numerical stability, no discontinuities |
| `checkerboard_nodata` | flat + `valid=False` blocks | tests mask propagation end to end |

Every fixture returns real `horizon` arrays computed by brute force in the fixture itself (slow but correct) — that also gives you an independent oracle for Part A's fast horizon map if you ever need to cross-check it.

---

## 5. Module specification

```
src/renderer/
  geometry/          ← PART A. FROZEN. Do not edit.
  reflectance.py     ← §5.1
  albedo.py          ← §5.2   (split out — the drape is primary now, it earns its own module)
  sensor.py          ← §5.3
  pairing.py         ← §5.4   (the P1 node; see C6)
  randomize.py       ← §5.5
  pipeline.py        ← §5.6   orchestrator
  config.py          ← §7     schema + hashing
```

### 5.1 `reflectance.py`

```python
def sun_vector(az_rad: float, el_rad: float) -> np.ndarray:               # (3,)

def cos_incidence(normals, s) -> np.ndarray:                              # (H,W) µ₀ = n·s, clipped ≥0
def cos_emission(normals, v) -> np.ndarray:                               # (H,W) µ  = n·v, clipped ≥0
def phase_angle(s, v) -> np.ndarray | float:                              # α = arccos(s·v)

def lunar_lambert_L(alpha_rad, table: LTable) -> np.ndarray | float:
    """Phase-dependent blend weight L(α) ∈ [0,1]. Table-driven, monotonic interp."""

def reflectance(normals, s, v, albedo, L_table, model="lunar_lambert"):
    """
    model ∈ {"lambert", "lommel_seeliger", "lunar_lambert"}  ← Ablation C switch

    lunar_lambert:  R = A · [ 2·L(α)·µ₀/(µ₀+µ)  +  (1−L(α))·µ₀ ]      ← note the 2 (C1)
    lommel_seeliger: R = A · 2·µ₀/(µ₀+µ)
    lambert:         R = A · µ₀

    Returns (H,W) float32 ≥ 0. Guard µ₀+µ → 0 with an epsilon; that denominator
    goes to zero at grazing incidence on steep slopes and will produce inf.
    """
```

**On `L(α)`:** `context.md` gives a starting shape rising from ~0.6 at α=10° toward ~1.0 by α=100° and labels it A-tier — correctly, because it is not fitted. Implement it as an interpolated **table**, not a formula, so an ISIS `pho_emp_local` table can be dropped in if anyone gets ISIS installed. Note honestly that even the *direction* of the trend is unverified. This matters less than it looks: **Ablation C tests whether the shading law affects matching at all**, and F19 requires the cost argument (Hapke needs parameters you cannot fit in a hackathon) to be the load-bearing justification, with the "shadows dominate" prediction kept downstream and explicitly labelled speculative. If C comes back null, the L(α) shape question is moot and you say so.

### 5.2 `albedo.py`

```python
def photometric_flatten(image_real, normals, s0, v0, L_table, clamp=(0.02, 0.6)):
    """
    Â = I_real / R_LL(i₀,e₀,α₀), clamped. Returns (albedo, valid_mask).
    valid_mask is False where the source was shadowed (unrecoverable, F17) or clamped.
    """

def inpaint_albedo(albedo, valid_mask, method="telea") -> np.ndarray

def procedural_albedo(shape, rng, *, mean, scale_range, fbm_octaves, fbm_sigma):
    """Fractal (fBm) albedo field. Fallback path for tiles with no co-located image."""

def drape(albedo, valid_mask, wac_dc_level=None) -> AlbedoField
```

Clamp range `(0.02, 0.6)` is A-tier — lunar normal albedo runs roughly 0.07 (mare) to 0.15 (highlands), so the bounds are deliberately loose to catch division blowups rather than to encode physics. Label it.

### 5.3 `sensor.py` — order is load-bearing

```python
def apply_sensor(radiance, cfg, rng) -> np.ndarray:
    """
    radiance: (H,W) float32, linear, unitless reflectance × albedo.
    Returns uint8 or uint16 DN.

    ORDER — do not reorder (C3):
      1. line_jitter     per-row along-track sub-pixel shift, low-order random walk.
                         Geometric: resamples the scene, so it precedes the optics.
      2. secondary_illum radiance += eps * ambient_estimate   ← see note below
      3. to_electrons    e⁻ = radiance * signal_scale
      4. psf_blur        Gaussian σ px. OPTICAL — must precede noise.
      5. shot_noise      rng.poisson(e⁻).  λ=0 in shadow → 0 electrons. Correct.
      6. read_noise      e⁻ += rng.normal(0, sigma_read_e)
      7. gain_offset     DN = e⁻ / gain + offset
      8. quantize        round, clip to [0, 2^bits − 1], cast
      9. to_8bit         optional, matches the ISRO baseline's normalisation
    """
```

Two notes worth more than they look:

**Shadowed pixels must not be exactly zero.** Real shadowed regions carry a read-noise floor and faint secondary illumination — they are dark, not black. Steps 5–7 give you that automatically: shot noise at λ=0 contributes nothing, read noise and offset give a realistic floor. This directly fixes the hard-zero problem visible in the current Part A output (§10), and it removes a trivial synthetic/real discriminator that a network would otherwise latch onto.

**`secondary_illum` (step 2) is a 3-line change that closes part of F2.** F2's list of unmodelled degrees of freedom opens with "secondary illumination off crater walls." A constant `eps ∈ [0.005, 0.03]` fraction of the local unshadowed mean, added inside shadows, is not radiosity — but it makes the network see *varying* shadow floors rather than a constant one, and it turns "we don't model secondary illumination" into "we randomise a crude proxy for it." Cheap, honest, and it is one of the few places you can shrink F2 rather than just admit it.

### 5.4 `pairing.py` — the P1 node (see C6)

```python
def make_pair(tile, render_a, render_b, shadow_a, shadow_b, warp: Similarity, cfg):
    """
    render_a/b: two renders of the SAME tile at DIFFERENT illuminations.
    GT correspondence in the tile grid is the IDENTITY — that is the free label.
    Apply a known similarity warp W to member B to inject scale (1–20×) and
    rotation (±180°); GT becomes W.

    Returns PairSample:
      image_a, image_b   uint8/16
      transform          3×3, exact
      match_mask         (H,W) bool — pixels with VALID correspondence
      meta               sun geometry for both, Δaz, Δel, config hash, tile id
    """
```

`match_mask` is the part to get right. Exclude any pixel that is:

- shadowed in **either** member (no signal → a correspondence there is a label on noise),
- outside `tile.valid` (nodata/void-fill),
- inside `halo_px` of the edge,
- outside the warp's valid support in B,
- `albedo_valid == False` (drape holes, §2.2).

Emitting correspondences in shadowed regions is the training-time twin of F13 — it teaches the matcher to produce matches where no information exists, which is the exact failure mode the PS's uniformity requirement already rewards by accident. Do not do it at training time too.

**Δaz sampling policy:** bias toward large Δazimuth (that is the invariance you are selling) but keep small-Δ pairs in the mix or the network never sees the easy case. Q25 (real Δaz distribution in the OHRC archive) is pure label-parsing with no GPU needed — if someone has PRADAN access, get the real histogram and sample against it instead of guessing.

**Output format is an interface with the matcher person.** EfficientLoFTR and RoMa dataloaders want different things. Agree it in writing before writing the writer. Default: HDF5 shards + a `manifest.jsonl`, one line per pair.

### 5.5 `randomize.py`

Per-sample draw over §6.7's full table — all nine rows, not the four in the draft (C5):

| Parameter | Range | Owner |
|---|---|---|
| Sun azimuth | 0–360° | §5.1 |
| Sun elevation | **20–70° equatorial** / 0–6° polar | §5.1 |
| Albedo scale | ×0.7–1.4 | §5.2 |
| Additive albedo noise | fractal, σ 5–15% | §5.2 |
| Terrain roughness amplitude | tuned to reproduce Vikram slope P95 = 9.25° | §5.2 / Part A |
| PSF σ | 0.4–1.0 px | §5.3 |
| Read noise σ | 0.5–3 DN | §5.3 |
| Secondary illum ε | 0.005–0.03 | §5.3 |
| Scale ratio (pair) | 1×–20× | §5.4 |
| Rotation (pair) | ±180° | §5.4 |

Every draw is seeded and logged into the sample's `meta`. A pair must be exactly reproducible from `(config_hash, tile_id, sample_index)`.

State in the report what §6.7 already says: **the randomisation is wider than the true distribution because the true distribution is unknown.** That is the honest framing and it is stronger than pretending otherwise.

### 5.6 `pipeline.py`

`render_tile(tile, cfg, rng) → List[PairSample]`. Caches per-tile invariants (normals, horizon, albedo) once; loops illuminations; the per-render cost is elementwise + one O(1) shadow lookup, which is what makes §6.9's throughput claim plausible. Streams shards to disk — never accumulates 20K pairs in RAM.

---

## 6. Test plan

**Unit — `reflectance.py`**

- **UT-1** `model="lunar_lambert", L=0` ≡ `model="lambert"` to 1e-6, on every fixture.
- **UT-2** `L=1` ≡ `2·µ₀/(µ₀+µ)` to 1e-6.
- **UT-3** *(the C1 catch)* On `flat_plane`, sun at zenith, nadir view: `R == A` for **all** L ∈ [0,1]. Fails loudly if the factor of 2 is dropped.
- **UT-4** R monotonically decreases as incidence increases at fixed emission; R = 0 for i ≥ 90°.
- **UT-5** R is exactly linear in A (doubling albedo doubles radiance) — guards against a stray clamp.
- **UT-6** *(convention check)* With nadir v, assert `phase_angle(s, v) == π/2 − el` to 1e-6 across the elevation sweep. A cheap, total check on the azimuth/elevation/vector conventions.
- **UT-7** `constant_slope`: reflectance matches the closed-form value.

**Unit — `sensor.py`**

- **ST-1** Poisson variance ≈ mean over a flat-radiance patch, within Monte-Carlo tolerance.
- **ST-2** Read-noise variance matches `sigma_read_e²` on a zero-signal patch.
- **ST-3** PSF-then-noise vs noise-then-PSF produce **measurably different** noise autocorrelation — asserts the ordering is actually being honoured, not just documented.
- **ST-4** Shadowed pixels have non-zero variance and a mean above 0 DN (the read-noise floor).
- **ST-5** Quantisation round-trips within ½ LSB; no clipping at nominal exposure.
- **ST-6** Line jitter produces row-correlated, column-uncorrelated displacement (the signature of a pushbroom, not of white noise).

**Unit — `albedo.py` / `pairing.py`**

- **AT-1** Round-trip: flatten a synthetic `I = A·R_LL` at geometry 0, recover Â ≈ A within tolerance where unshadowed.
- **AT-2** `albedo_valid` is False exactly on the source shadow mask.
- **PT-1** With `warp = identity` and both members unshadowed, `match_mask` is all-True inside the halo-stripped region.
- **PT-2** Every `match_mask == True` pixel maps under `transform` to an in-bounds, unshadowed, valid pixel in B. Assert on a random 10k sample per pair. **This is the single most important test in the suite** — a silent bug here poisons the whole training set and shows up only as "the model doesn't converge" on day 12.
- **PT-3** Determinism: same `(config_hash, tile_id, sample_index)` → byte-identical output.

**Integration**

- **IT-1** Full pipeline on `gaussian_bump`, 8 illuminations → 28 pairs, all tests pass, no NaN/inf anywhere.
- **IT-2** Config hash changes ⟹ output changes; config unchanged ⟹ output byte-identical.

---

## 7. Config

Single `configs/render.yaml`, SHA-256 hashed, hash written into every sidecar and into `manifest.jsonl`. Sections: `reflectance` (model, L-table, epsilon), `albedo` (source, clamp, fbm params), `sensor` (all of §5.3), `randomize` (§5.5 table), `pairing` (Δaz policy, warp ranges, output format), `runtime` (device, shard size, tile list).

Schema-validated on load (pydantic). An out-of-range value is a hard failure, not a warning — F19 is what happens when unchecked numbers propagate into reasoning.

---

## 8. Task graph and gates

| # | Task | Acceptance |
|---|---|---|
| **T0** | Freeze interface (§4), write fixtures (§4.3), stub `cast_shadow` | Fixtures import with Part A absent; `pytest` collects and runs |
| **T1** | `reflectance.py` | UT-1…UT-7 pass |
| **T2** | `sensor.py` | ST-1…ST-6 pass |
| **T3** | `albedo.py` — procedural first, drape second | AT-1, AT-2 pass |
| **T4** | `pipeline.py` + `randomize.py`, wire to **real** Part A | IT-1, IT-2 pass · **T4.3 ⇒ GATE A** |
| **T4.3** | **F21 throughput measurement.** Render one real tile end to end, record wall-clock, extrapolate to the 20K target | **Report the number before starting T5.** If the projection exceeds ~half the remaining budget: **cut tiles, not illuminations** (§6.9 — illumination diversity is the signal). Re-run Milestone 1b's arithmetic at the corrected pair count. |
| **T5** | `pairing.py` | PT-1, PT-2, PT-3 pass · format agreed with matcher owner in writing |
| **T6** | **§6.10 render-vs-real at Vikram** — joint with Part A | **GATE B.** Part A owns shadow-mask IoU; Part B owns NCC / SSIM / gradient-orientation histogram distance on the **non-shadowed** region. Report all as a fraction of the empirical ceiling (same metrics between two *real* images of the same terrain at similar sun angles). No published target exists — §6.10 is explicit — so the ceiling **is** the target. Produces panel **V7**. |
| **T6.2** | **F17 drape-degradation curve** | Re-render the drape at Δaz = 0/45/90/180° from the source image's own geometry; plot shadow-mask IoU and NCC against Δaz. One extra column from work T6 already does. Expect degradation — the point is to bound it. |
| **T7** | Bulk build to the T4.3-corrected pair count | Manifest complete, spot-check 20 random pairs by eye |
| **T8** | **Milestone 1b transfer probe** — joint with matcher owner | **GATE C.** 500–1,000 pairs from one tile near the validation site → short fine-tune probe (not the final model) → test on P1 against the off-the-shelf baseline from §13.1 step 3. Probe beats off-the-shelf ⟹ proceed. It doesn't ⟹ **invoke Plan B (§5.8) now**, before more rendering time is sunk. |

Gates A/B/C are the F15/F21/F23 fixes made operational. `resolutions.md` F23 is explicit: **the order is load-bearing; the calendar is not.** Do not put day numbers on this until Q9/Q10/Q11 come back from the team.

---

## 9. Non-goals — do not let the agent wander here

- **No Hapke.** §6.3 puts it at stretch-only, and F19 fixes the justification order: the citable reason is McEwen's demonstrated result that lunar-Lambert fits Hapke over the relevant range; the second reason is that Hapke needs parameters you cannot fit in a hackathon; the "shadows dominate" prediction is downstream and explicitly speculative. If an agent proposes Hapke, the answer is Ablation C first.
- **No opposition surge** (follows from the above — see C2).
- **No Blender / Cycles / CORTO.** §6.8 recommends the NumPy/PyTorch kernel precisely because it is controllable and dependency-free.
- **No perspective camera.** Ortho only.
- **No IIRS spectral or thermal modelling.** IIRS routes through MatchAnything at §5.5 (Apache 2.0, confirmed — R10 struck), not through the renderer.
- **No edits under `geometry/`.** Bugs found in Part A get filed, not fixed in place.
- **No visual-polish work.** F6.

---

## 10. Diagnostics on the current Part A output

Three observations from the shadow-map image, worth resolving *before* T6, because once Part B is attached these will be misattributed to the reflectance model.

1. **Directional anisotropy consistent with striping.** Mean absolute column-to-column change is ~2.2× the row-to-row change (1.99 vs 0.91 DN). JPEG blocking would be roughly isotropic, so this is unlikely to be compression alone. Most probable causes, in order: azimuth-bin quantisation in the horizon map (K=16 aliases much harder than K=32 — worth a K sweep), ray-march step size coarse relative to the pixel grid, or the DEM's own along-track striping surviving into the normals. **Test:** re-render the same tile at K=16 vs K=32 and at two march step sizes; if the anisotropy moves, it is the horizon map, not the data.

2. **Edge shadow inflation.** Shadow fraction is 0.59–0.60 in the outer 10 columns against 0.456 in the interior. That is the signature of rays marching off the tile edge and defaulting to "occluded." The 128 px halo (§6.1) exists for exactly this — confirm it is being **stripped before use** and not just carried. If the halo is already stripped and the inflation persists, the march is terminating early at the array bound.

3. **35% of pixels are exactly zero.** Hard-zeroing shadowed pixels. Not a bug in Part A, but §5.3's sensor chain must replace it with a read-noise floor before any pair generation — a network handed exact-zero shadows will learn "exactly 0" as a synthetic-image feature, which is F2's failure mode in its most literal form.

Also worth ruling out: the hard-edged rectangular black regions at the top-right and left of the frame do not look like terrain shadow. Confirm they are real occlusion and not nodata or crop boundary leaking through as shadow.

---

## 11. Open items to route

**To the matcher owner (blocking T5):** exact pair format and dataloader contract. HDF5 shards + `manifest.jsonl` unless told otherwise.

**To the team lead (blocking nothing, but decide today):** who owns the P1 pairing node (C6)? Part B is claiming it by default in this plan.

**To the team (Q9/Q10/Q11, `resolutions.md` §D):** GPU/RAM/storage and for how long; days until internal submission; who owns renderer vs matcher vs UI vs slides. T4.3's cut decision and the whole of §8's sequencing need Q9 and Q10.

**To whoever gets PRADAN access first (Q25):** real sun-azimuth-Δ distribution across the OHRC archive. Pure label parsing, no GPU, no rendering — and it turns §5.4's Δaz sampling from a guess into a measurement.

**Unresolvable from here (Q1, `resolutions.md` §B.1):** sub-pixel accuracy relative to which image's pixel grid — a ~20× swing on the headline number. Organiser-only. Does not block Part B, but every RMSE Part B feeds into the report must carry the F20 dual-report convention: held-out protocol **and** same-set-as-fit protocol, both, always.
