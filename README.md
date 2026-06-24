# Sphere Imposter Baker

Blender 5.1 addon that bakes sprite-sheet impostors of any selection of objects by rendering them from a sphere of cameras arranged in latitude slices. Outputs a packed atlas PNG with embedded metadata, a lookup texture for runtime cell-picking, and an interactive verification preview that follows your viewport.

Built for retro / PSP-style game pipelines but works for any impostor use case.

## Install

1. Open Blender 5.1 → Edit → Preferences → Get Extensions → ⌄ menu → "Install from Disk"
2. Pick `sphere_imposter_baker.zip`
3. 3D Viewport → press `N` → **Imposter** tab

Older Blender (4.2+) also works via the same flow; below 4.2 you'd need to extract into your addons folder.

## Quick Start

1. Select your object(s). Multi-selection is fully supported — bake operates on the combined bounding box, including children of selected objects.
2. Set **Output Sheet** location, prefix, and per-sprite resolution.
3. Set **Number of Slices** (default 5). First and last slices are auto-locked as single-camera poles (+90°/-90°). Middle slices' camera counts are editable.
4. (Optional) Click **Show Cameras** to preview camera positions in the viewport before committing.
5. Click **Bake from Selection**.
6. Switch viewport shading to **Material Preview** to see the verification sprite. Orbit around it; it should follow your view and switch sprite cells at slice/azimuth boundaries.

## Settings

### Output Sheet
- **Sheet Location** — folder for atlas, LUT, and (optional) individual PNGs
- **Prefix** — filename prefix
- **Sheet Resolution** — per-sprite cell size in pixels. Atlas size = `cells × sheet_resolution`
- **Transparency** — `Alpha` (smooth RGBA) or `Dither` (8×8 Bayer → binary alpha, PSP-style stippled transparency)

### Sphere Slices
- **Number of Slices** (3–33) — total latitude rings including the two poles
- Per-slice **camera count** editable inline in the list (poles are locked at 1)
- Per-slice **azimuth offset** for staggering rings against each other
- Presets: **Equator-Weighted** (1, 3, 8, 3, 1 = 16 sprites) and **Uniform** (8 per middle slice)

### Camera
- **Type**: Orthographic (recommended for impostors) or Perspective
- **Auto-fit Camera** — computes distance and ortho scale from your selection's bounding sphere. For perspective cameras, derives the distance needed to fit the bounding sphere in the FOV
- **Padding** — extra space around the object as a fraction of bounding radius

### Render Options
- **Render Engine** — Use Scene Engine (default), force EEVEE, or force Cycles. If your scene engine is Workbench, it auto-switches to EEVEE since Workbench ignores `film_transparent`
- **Cycles Samples** — only used when Cycles is selected (default 16, fine for impostors)
- **Lighting** — Auto / Scene Only / Ambient / Shadeless:
  - *Auto* (default): use scene lights if any; if none, fall back to **Shadeless** (replace each material's surface with an Emission of its base color)
  - *Ambient*: temporarily override world background with white emission
  - *Shadeless*: always replace materials with Emission (true fullbright, matches what a game engine reads from the texture without lighting)
- **Hide Non-Selected Meshes** — hide other meshes during render so they don't pollute alpha. Lights stay visible
- **Delete Per-Sprite PNGs** — auto-delete the individual sprite files after the atlas is packed (default on)

### Verification Sprite
- **Create Verification Sprite** — drop a textured plane at the selection center with the impostor shader applied
- **Billboard to Camera** — plane permanently follows the **viewport camera** via a 30 fps timer (not the scene camera — works with any viewport navigation)
- **Preview Offset** — distance from bake center, so the preview doesn't sit inside the original (set to 0 to overlap)

## Output

```
sprite_atlas.png      ← packed sprite sheet (with embedded metadata, see below)
sprite_lut.png        ← lookup texture for runtime cell-picking
sprite_manifest.json  ← human-readable metadata: slices, cameras, sprites
sprite_NNNN_*.png     ← individual sprites (auto-deleted by default)
```

### Atlas packing
Dense square-ish grid: `cols = ceil(sqrt(N))`, `rows = ceil(N/cols)`. 16 sprites → 4×4, 9 → 3×3, 25 → 5×5. No wasted space.

### Embedded PNG metadata
The atlas PNG includes an `iTXt` chunk with keyword `SIB_AtlasMeta` containing JSON:
```json
{
  "version": 1, "sprite_size": 256, "cols": 4, "rows": 4, "count": 16,
  "transparency": "alpha",
  "sprites": [
    {"i": 0, "el": 90, "az": 0, "col": 0, "row": 0, "x": 0, "y": 0},
    ...
  ]
}
```
Every sprite carries its elevation, azimuth, and atlas position. The PNG is fully self-describing — no sidecar needed in your asset pipeline.

### LUT
A 128×64 PNG (Non-Color) mapping `(azimuth, elevation)` → `(col_offset, row_offset)` of the matching atlas cell. Sample it in your engine's shader with the world-space view direction to pick the right sprite at runtime. The Blender verification preview uses this same LUT.

### Reading metadata in your pipeline
`read_atlas_metadata.py` is bundled in the zip — pure stdlib, drop it in your asset toolchain:
```bash
python read_atlas_metadata.py sprite_atlas.png
# Atlas: 4×4 cells, 256px each, 16 sprites, transparency: alpha

python read_atlas_metadata.py sprite_atlas.png 0 45
# Nearest cell to (el=0, az=45): col=2 row=1 (actual el=0 az=45)
```
Provides `read_atlas_metadata(path)` and `find_nearest_cell(meta, el, az)`.

## Tips

- **Black sprites?** Check the active view layer (top right of the viewport) for a Material Override — clear it or switch to the default `ViewLayer`. The bake auto-clears overrides during render but custom view layers can have other settings interfering. Try `Render Engine = Cycles` if EEVEE behaves weirdly with your scene.
- **Preview plane shows the LUT instead of the atlas:** you're in Solid viewport shading. Switch to **Material Preview** (the sphere icon in the top-right of the viewport).
- **Preview doesn't follow camera:** make sure "Billboard to Camera" is checked. The polling timer needs at least one open 3D viewport. If you closed and reopened the file, the timer re-registers automatically when you bake again.
- **Cameras inside the selection:** make sure Auto-fit is on. For perspective cameras it now derives distance from FOV + bounding sphere radius.
- **Verification:** the embedded LUT image is `(R = col_offset, G = row_offset, B = 0)` — useful for porting the impostor shader to your engine. Sample with `(az_normalized, el_normalized)` UVs using nearest-neighbor.

## How the runtime shader works

The verification plane's shader is the runtime model you'd port to your engine:

1. View direction from object center to camera (constant per object → no per-fragment tearing)
2. `az = atan2(y, x)`, `el = asin(z)`, normalize to [0,1]
3. Sample LUT (Closest filter, REPEAT extension) → returns `(col_offset, row_offset)` of cell's bottom-left UV in the atlas
4. Compute final atlas UV: `lut.rg + planar_uv * (1/cols, 1/rows)`
5. Sample atlas (Closest, CLIP) → output color + alpha

Both texture samples use Closest interpolation, so transitions between cells are instant — no blending.

## License

GPL-3.0-or-later
