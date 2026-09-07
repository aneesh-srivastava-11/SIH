# Part A Renderer — Geometry & Shadows

Documentation for `dem_loader.py`, `horizon_maps.py`, and `render_test.py` —
the DEM-loading, surface-normal, and horizon-map shadow-casting pipeline
from PS 26166's renderer (context.md §6.1–6.2, §6.4).

This is Part A of the renderer. It produces geometry (normals) and a
shadow mask from a DEM and a sun direction. It does **not** implement
reflectance/albedo/sensor noise — that's Part B (§6.3, §6.5–6.7).
`render_test.py` uses a flat constant-albedo shading model only to
validate that Part A's geometry is correct in isolation.

---

## 1. What the pipeline does, end to end

```
  DEM file (SLDEM2015 / NAC DTM / TMC-2 DEM)
        │
        ▼
  dem_loader.py  ── crop to a footprint, tile, extract lat-aware pixel spacing
        │
        ▼
  horizon_maps.py ── surface normals (Horn 3x3) + sun vector
        │                        │
        │                        ▼
        │              horizon-map precompute (K azimuth bins)
        │                        │
        └───────────┬────────────┘
                     ▼
              shadow_mask() ── O(1) lookup: shadowed iff sun_elev < horizon_angle
                     │
                     ▼
  render_test.py ── composite: albedo * max(0, n·s), zeroed where shadowed
                     │
                     ▼
              8-bit test image ── diff against a real cropped OHRC/NAC image
```

The reason this is split into three files: `dem_loader.py` only knows
about rasters and coordinates, `horizon_maps.py` only knows about geometry
and light, and `render_test.py` is a thin script that wires the two
together for validation. Part B's reflectance/sensor code will replace
`render_test.py`'s single `albedo * shading` line with the real
Lunar-Lambertian model and the sensor chain — it should not need to
touch the other two files.

---

## 2. `dem_loader.py`

### Purpose
Open a DEM raster, crop it to a target lon/lat footprint, and tile it
for the renderer.

### Key assumption (read this before using a real file)
The loader assumes the DEM's affine transform is in **degrees per
pixel** (a geographic-style raster), not a projected constant-metres
raster. This matters because true ground distance per degree of
longitude shrinks with `cos(latitude)` — a projected raster's fixed
metres-per-pixel value silently bakes in one reference latitude and
is wrong everywhere else. If `gdalinfo <your_file>` shows a projected
CRS with a fixed metre pixel size, reproject to a plain lat/lon grid
first:
```
gdalwarp -t_srs EPSG:4326 input_dem.tif reprojected_dem.tif
```
(EPSG:4326 is a stand-in here purely to force a degrees-per-pixel
grid — the Moon isn't Earth-WGS84, but rasterio/GDAL only need the
transform's unit convention, not true Earth geodesy, for this step.)

### Functions

**`load_and_crop(dem_path, min_lon, max_lon, min_lat, max_lat, margin_deg=0.05)`**
Opens `dem_path`, windows it to the bounding box (padded by `margin_deg`
so tile halos have real data near the edges), and returns:
- `dem` — the cropped height array (metres, float64)
- `transform` — the affine transform of the cropped window
- `dlon_deg`, `dlat_deg` — pixel spacing in degrees

Get `min_lon/max_lon/min_lat/max_lat` from your PDS4 parser's
`geometry.json` corners — take the min/max lat and lon across all four
corners (`upper_left`, `upper_right`, `lower_left`, `lower_right`).

**`row_center_lats(transform, n_rows)`**
Returns the latitude of each row's center. Feed this straight into
`horizon_maps.ground_spacing()`.

**`tile_dem(dem, transform, tile_size=1024, halo=128)`**
A generator yielding `(tile, row0, col0, lat_deg_for_tile_rows)` for
each tile. `row0`/`col0` are the tile's offset *excluding* the halo, so
you can stitch renders back into the full mosaic later. Tiles at the
DEM's edge are naturally smaller — no synthetic padding is invented.

### Running it on a different input image
Nothing in this file is specific to P1. To point it at a different
product:
1. Get that product's corners from its own `geometry.json` (your
   `pds4_parser.py` output).
2. Pick the right DEM for that footprint's latitude:
   - `|lat| < 60°` → SLDEM2015 tile covering that lon/lat box
   - `|lat| ≥ 60°` or you need higher resolution at a specific site →
     LOLA polar products or a NAC stereo DTM (only exists at select
     sites — e.g. `NAC_DTM_VIKRAMSITE1` at 69.37°S, 32.32°E)
3. Call `load_and_crop(dem_path, min_lon, max_lon, min_lat, max_lat)`
   with that product's bounds.

---

## 3. `horizon_maps.py`

### Purpose
Surface normals, the sun vector, and the horizon-map shadow precompute —
the actual geometry engine.

### Functions

**`ground_spacing(lat_deg, dlon_deg, dlat_deg)`**
Converts angular pixel spacing to ground metres, correcting for
latitude: `dx = dlon_deg · (π/180) · R_MOON · cos(lat)`,
`dy = dlat_deg · (π/180) · R_MOON`. `R_MOON = 1,737,400 m` (a sphere,
not an ellipsoid). Returns `(dx, dy)`, each shaped `(ny,)` — one value
per DEM row, since `dx` depends on that row's latitude.

**`surface_normals(h, dx, dy)`**
Horn 3×3 gradient kernel (what `gdaldem slope` uses — less noisy than a
plain central difference) over height array `h`, using the per-row
`dx`/`dy` from `ground_spacing()`. Returns `(ny, nx, 3)` unit normals,
`n = (-∂h/∂x, -∂h/∂y, 1) / |·|`.

**`sun_vector(azimuth_deg, elevation_deg)`**
`s = (sin(A)cos(e), cos(A)cos(e), sin(e))`. Azimuth is clockwise from
north; pull `azimuth_deg`/`elevation_deg` straight from a product's
`geometry.json` (`sun_azimuth_deg`, `sun_elevation_deg`).

**`horizon_map(h, dx, dy, k_azimuths=16, max_dist_m=5000.0, n_steps=40)`**
For each pixel and each of `k_azimuths` evenly-spaced directions,
marches outward in geometrically-spaced steps (dense near the pixel,
sparse far away) and records the steepest horizon elevation angle seen.
This is the expensive one-time cost — see §5 (Performance) before
scaling it up. Returns `(horizon_deg, azimuths_deg)`.

**`shadow_mask(sun_azimuth_deg, sun_elevation_deg, horizon_deg, azimuths_deg)`**
O(1)-per-pixel lookup: finds the nearest precomputed azimuth bin and
returns `True` wherever `sun_elevation_deg < horizon_deg[..., bin]`.
This is the payoff of the precompute — call this as many times as you
want (different sun angles for domain randomization) without
re-marching.

### Running it on a different input image
Same DEM-in, geometry-out interface regardless of the source product:
1. Get `h` (height array), `dx`, `dy` from `dem_loader.py` for that
   product's footprint (via `ground_spacing()` using its `row_center_lats`).
2. Call `surface_normals(h, dx, dy)` once per DEM tile — it doesn't
   change with sun angle, so cache it.
3. Call `horizon_map(h, dx, dy, ...)` once per DEM tile — also
   sun-angle-independent, also cacheable.
4. Call `shadow_mask(sun_az, sun_el, horizon_deg, azimuths_deg)` once
   per **render** — this is the only step that changes per sun angle,
   which is the entire point of the precompute.

If the new product's footprint spans a very different latitude, sanity
check that `ground_spacing()`'s `cos(lat)` correction is doing
something non-trivial — near the equator it's close to 1 and easy to
not notice if it's silently wrong; near the poles it matters a lot.

---

## 4. `render_test.py`

### Purpose
A validation-only compositor: `albedo · max(0, n·s)`, zeroed wherever
`shadow_mask()` says shadowed. Constant albedo is deliberate — it
isolates whether the *shape* (normals + shadows) is right, independent
of anything Part B is doing with reflectance or texture.

### Functions

**`render_shading_and_shadow(h, dx, dy, sun_azimuth_deg, sun_elevation_deg, k_azimuths=16, max_dist_m=5000.0, n_steps=40, albedo=0.12)`**
Runs the full chain (normals → horizon map → shadow mask → composite)
and returns `(image, n, shadow)` — the rendered image plus the
intermediate normals and shadow mask, since you'll want to inspect
those separately when something looks wrong.

**`to_8bit(image)`**
Normalizes the float render to 8-bit so it's directly comparable to a
real OHRC/NAC crop (which is already `uint8`).

### Running it on a different input image
```python
from dem_loader import load_and_crop, row_center_lats
from horizon_maps import ground_spacing
from render_test import render_shading_and_shadow, to_8bit
import numpy as np, json

# 1. Load that product's parsed geometry (from your pds4_parser.py output)
geom = json.load(open("data/<product>_geometry.json"))
lons = [float(v[1]) for v in geom["corners"].values()]
lats = [float(v[0]) for v in geom["corners"].values()]

# 2. Get the matching DEM crop
dem, transform, dlon_deg, dlat_deg = load_and_crop(
    "path/to/your_dem_tile.tif",
    min(lons), max(lons), min(lats), max(lats),
)

# 3. Ground spacing, lat-corrected
lat_deg = row_center_lats(transform, dem.shape[0])
dx, dy = ground_spacing(lat_deg, dlon_deg, dlat_deg)

# 4. Render at this product's real sun geometry
image, normals, shadow = render_shading_and_shadow(
    dem, dx, dy,
    sun_azimuth_deg=float(geom["sun_azimuth_deg"]),
    sun_elevation_deg=float(geom["sun_elevation_deg"]),
)

img8 = to_8bit(image)
np.save(f"render_{geom['product_id']}.npy", img8)
```

Then diff `img8` against that product's real cropped array (resampled
to the DEM's GSD first, since the DEM and the real image are almost
certainly at different resolutions) — that's your per-product §6.10
validation check.

---

## 5. Performance notes (read before scaling up)

- `horizon_map()` is the expensive step: cost scales with
  `k_azimuths × n_steps × (ny × nx)`. The synthetic self-test (200×200,
  K=16, 30 steps) ran in **0.85s** — that does **not** linearly predict
  a 1024×1024 tile's cost; time it directly on your real tile size
  before committing to how many tiles/illuminations you plan to render
  (this is R12 / F21 in the project's risk log — measure, don't
  extrapolate).
- If a real tile is too slow, cheapest knobs in order:
  1. Lower `n_steps` first — shadow accuracy degrades gracefully.
  2. Lower `k_azimuths` from 16→8 only as a last resort — this is a
     real accuracy cut (coarser azimuth resolution), not just speed.
  3. `max_dist_m` — only lower this if your terrain's relief is small
     enough that distant horizons can't matter; check against the
     DEM's actual elevation range first.
- `surface_normals()` and `horizon_map()` don't depend on sun angle —
  compute them once per DEM tile and reuse across every sun-angle
  render of that tile (that's the entire point of domain randomization
  being cheap).

---

## 6. Known placeholders — fix before trusting real output

| Placeholder | Where | What to do |
|---|---|---|
| `albedo=0.12` constant | `render_test.py` | Fine for geometry validation; Part B replaces this with real Lunar-Lambertian + albedo drape (§6.3, §6.5) |
| `dlon_deg = dlat_deg = 0.26 / 111320` | `render_test.py`'s `__main__` self-test only | Never used outside the synthetic test — real runs get `dlon_deg`/`dlat_deg` from `dem_loader.load_and_crop()` |
| Emission/view vector not modeled | not in this pipeline at all | Shadows only need the sun vector, so this is fine for Part A; Part B needs a view vector for `µ = n·v` in the reflectance model — see the note on using P1's `roll`/`pitch` fields as an approximation |
| Degrees-per-pixel raster assumption | `dem_loader.py` | Check `gdalinfo` on your actual DEM file before assuming this holds |

---

## 7. Sanity checks built into each script

Each file's `__main__` block runs a synthetic self-test with an assert,
so you can confirm the code itself is correct before pointing it at
real data:
- `horizon_maps.py` — a single hill, checks the shadowed fraction is
  neither 0% nor ~100% at P1's real (very low) sun elevation.
- `dem_loader.py` — an in-memory synthetic raster matching P1's
  footprint, checks the cropped latitude range matches the requested
  bounds.
- `render_test.py` — chains both, checks the render has non-zero
  signal and a non-degenerate shadow fraction.

Run any of them directly (`python3 horizon_maps.py`, etc.) to re-verify
after you make changes.
