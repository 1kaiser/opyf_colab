# opyf_colab

**[Open In Colab](https://colab.research.google.com/github/1kaiser/opyf_colab/blob/main/jax_3d_reconstruction_colab.ipynb)**

End-to-end hydraulic analysis pipeline: event-day video → metric flow depth → discharge → IS-code canal design → 3D model. Built on JAX, Depth Pro, LightGlue, and FreeCAD.

---

## Pipeline at a Glance

<table>
<tr>
<td width="42%" valign="top">

**Repo layout**
```
opyf_colab/
├── pipeline.py              ← single entry-point
│
├── modules/
│   ├── depth_to_elevation.py
│   ├── align_pointclouds.py
│   ├── canal_optimizer.py
│   ├── canal_cad.py
│   ├── infer_depth.py
│   ├── infer_features.py
│   ├── infer_stereo.py
│   ├── infer_video.py
│   ├── segment_water.py
│   ├── reconstruction.py
│   └── annotated_pipeline_viz.py
│
├── assets/                  ← generated figures
├── data/brague/             ← auto-downloaded
├── models/jax/              ← JAX model weights
│   ├── jax_depth_pro/
│   ├── jax_lightglue/
│   ├── jax_mast3r/
│   └── jax_vggt/
├── tests/Test_Brague_flood/
└── web/jax-js-fem/
```

**Stage flow**
```
Stage 0  ─ Download release assets
           ↓
Stage 1  ─ Extract frames from video
           ↓
Stage 2  ─ Load Depth Pro weights
           ↓
Stage 3  ─ Load ortho + dry mask
           ↓
Stage 4  ─ Rasterise MNT bed model
           ↓
Stage 5  ─ Per-frame depth inference
           Z_surface(x,y) = s·d + t
           ↓
Stage 6a ─ Aggregate → flow_depth.tif
           h(x,y) = Z_surface − Z_bed
           ↓
Stage 6b ─ LightGlue bank matching
           → homography H
           → GCP scale solve (s, t)
           → water volume (m³)
           ↓
Stage 7  ─ Manning + IS codes
           → B, D, side slope
           → FreeCAD 3D model
           ↓
Stage 8  ─ Annotated visualisation
```

</td>
<td width="58%" valign="top">

![Annotated pipeline](assets/annotated_pipeline.png)

</td>
</tr>
</table>

---

## Quick Start

```bash
# Full pipeline — auto-downloads all assets, then runs all stages
JAX_PLATFORMS=cpu python3 pipeline.py

# Fast re-run (skip download + depth inference if outputs exist)
JAX_PLATFORMS=cpu python3 pipeline.py --skip-download --skip-depth --skip-canal
```

All outputs land in `output/brague/` and figures in `assets/`.

---

## Input Data

<table>
<tr>
<td width="33%" align="center">

**Event Video**<br>
`IMG_1139.MOV` · 1920×1080 · 22 MB<br>
Brague flood, downstream bridge<br>
23 November 2019

</td>
<td width="33%" align="center">

**Extracted Frame**<br>
<img src="assets/frame_sample.png" width="320"/><br>
`output/brague/frames/frame_NNNNN.png`

</td>
<td width="33%" align="center">

**Pre-event Ortho**<br>
`Ortho.tif` · 2.4 mm/px<br>
15228 × 13222 px · EPSG:2154<br>
37 m × 32 m reach

</td>
</tr>
</table>

---

## Analysis Stages

### Stages 1–6a — Flow Depth Raster

```
Event video  ──► Depth Pro JAX ──► inverse depth d(x,y) [per frame]
                                         │
              MNT.xyz ──► Z_bed(x,y) ────┤  scale solve:  Z = s·d + t
                                         │  (dry-pixel GCPs from LightGlue)
                                         ▼
                              h(x,y) = Z_surface − Z_bed   [metres]
                                         │
                                         ▼
                              Q = ∫ α·V(x,y)·h(x,y) dA    [m³/s]
                                  α = 0.9 ± 0.1  (Welber 2016)
                                  V from LSPIV (opyflow)
```

<img src="assets/flow_depth_result.png" width="100%"/>

*Flow depth h(x,y) colourmap over the ortho grid. Blue = shallow, red = deep.*

---

### Stage 6b — Point Cloud Alignment + Water Volume

LightGlue matches bank features (rocks, concrete edges) between the event frame and the dry ortho. The matched keypoints are known-dry, so their MNT elevation is ground truth.

```
Event frame PNG          Ortho.tif (pre-event, dry)
      │                         │
      └──── SuperPoint ──────────┘
            LightGlue
            ↓
      Bank keypoint matches  (guaranteed-dry bank features)
            │
      ┌─────┴──────────────────────────────────────┐
      │  H : frame pixels → ortho pixels           │
      │  GCP Lambert-93 coords from ortho CRS      │
      │  Z_bed at each GCP from MNT (interpolated) │
      │  s, t solve:  Z_bed = s·d_DepthPro + t     │
      └─────────────────────────────────────────────┘
            │
      Warp inv_depth → ortho grid via H⁻¹
      Z_surface = s·d_warped + t
      h(X,Y) = Z_surface − Z_bed
      Volume = Σh · pixel_area  [m³]
```

<img src="assets/multiview_comparison.png" width="100%"/>

*Top / front / side / isometric views of the bed point cloud (terrain), flood surface, water column (cyan), and bank GCPs (yellow stars).*

---

### Stage 7 — Discharge → Design Requirements → IS Code Canal Section

#### From measurement to design discharge

| Measured | Value |
|---|---|
| h_mean (flow depth) | **1.152 m** |
| h_max | **2.124 m** |
| Wet area | ~39,700 m² |
| Scale factor s | 0.030–0.035 |
| Z offset t | ~11.9 m |
| Water volume | computed per run |

#### Analytical minimum-cost section (Chow 1959 / IS 10430:2000)

For the hydraulically efficient trapezoidal section the classical result gives **R = D/2**, which yields a closed-form solution:

```
coef = 2√(1 + m²) − m                          [m = side slope H:V]

D    = [ Q · n · 2^(2/3) / (√S · coef) ]^(3/8) [water depth, metres]

B    = 2D · (√(1 + m²) − m)                    [bed width, metres]
```

IS code constraints applied after:

| Standard | Clause | Parameter |
|---|---|---|
| IS 5968:1987 | Table 1, Cl 8.1 | Min curve radius by Q |
| IS 10430:2000 | Table 1 | Freeboard by Q |
| IS 10430:2000 | Table 2 | Side slope — concrete: 1.5H:1V |
| IS 10430:2000 | Cl 4.1 | Manning's n = 0.018 (concrete) |
| IS 10430:2000 | Cl 4.2 | Velocity 0.6–2.5 m/s |

#### Design output (Q = 50 m³/s, S = 1:5000, concrete lining)

```
  ┌──────────────────────────────────────────────────┐
  │   CANAL DESIGN REPORT  (IS 5968 / IS 10430)      │
  │──────────────────────────────────────────────────│
  │   Bed width   B  :   2.59 m                      │
  │   Water depth D  :   4.27 m                      │
  │   Side slope     :   1.5 : 1  (H:V)              │
  │   Top width      :  17.20 m                      │
  │   Velocity    V  :   1.30 m/s  ✓ [0.6–2.5]       │
  │   Freeboard      :   0.60 m   (IS 10430 Table 1)  │
  │   Min radius     : 1000 m     (IS 5968 Table 1)   │
  │   Slope          :   1 : 5000                     │
  │   Manning n      :   0.018                        │
  │   IS Compliance  :   ✓ PASS                       │
  └──────────────────────────────────────────────────┘
```

**Measurement → discharge → IS design chain:**

![Design chain](assets/design_chain.png)

*Left: flow depth distribution from Stage 6a with h_mean and h_max marked.  
Centre: Manning's equation components (A, P, R, V, Q) for the optimal section.  
Right: B–D parameter space — yellow dashed contour = Q target, green = V_min limit, red = V_max limit, red dot = IS-compliant solution.*

**Labeled cross-section:**

![Canal cross-section](assets/canal_section.png)

*Trapezoidal section: blue = water body, grey = concrete lining (IS 10430), hatched = freeboard zone. All IS-code dimensions annotated.*

---

### Stage 7b — Reach Geometry: Slope + Curvature → Informed Design

The flow-depth raster is used to extract the **hydraulic thalweg centreline** and measure the terrain geometry that feeds directly back into the design stage.

```
flow_depth.tif  ─► wet mask  ─► fill holes + distance transform
                                      │
                              Row-wise centroid weighted by dist²
                              (hydraulic thalweg = D8 flow path ridge)
                                      │
                              Gaussian smoothing (σ = 1 m)
                                      │
                  ┌───────────────────┴──────────────────────────┐
                  │  Reach metrics                                │
                  │    L    = 44.8 m  (arc length)               │
                  │    B̄    = 21.5 m  (mean wet width)           │
                  │    S    = 1:185   (best-fit terrain slope)    │
                  │    R_min= 5.9 m   (tightest bend, measured)  │
                  │    R̄    = 280 m   (mean curvature radius)    │
                  └──────────┬───────────────────────────────────┘
                             │
                  ┌──────────▼───────────────────┐
                  │  IS code feedback             │
                  │  IS 5968 R_min = 1000 m       │
                  │  → 5.9 m << 1000 m            │
                  │  Terrain too curved for a     │
                  │  straight concrete canal:     │
                  │  add bends / realign           │
                  │  IS 10430 S_design ≤ 1:5000   │
                  │  → drop structures required   │
                  └──────────────────────────────┘
```

<img src="assets/canal_3d_overlay.png" width="100%"/>

*Five-panel geo-referenced overlay: **A** — 3D terrain surface (Lambert-93) with flood and designed canal cross-section at mid-reach; **B** — plan view with centreline colour-coded by radius of curvature (green = gentle, red = tight); **C** — curvature radius profile vs IS 5968 minimum (red dashed); **D** — longitudinal profile showing terrain, flood, and canal invert; **E** — mid-reach cross-section comparing existing flood width with the IS-code designed trapezoidal section.*

### Stage 7c — 3D CAD Model via FreeCAD

The IS-compliant section is swept along a 1 km alignment (IS 5968 curve) and exported as STEP + OBJ. The section is then placed back into the ortho photo coordinate system for visual validation.

```bash
# Generate 3D model + 2D section drawings
freecadcmd modules/canal_cad.py \
    --params output/canal_params.json \
    --output output/canal

# Optional: compound (multi-stage berm) section
freecadcmd modules/canal_cad.py --params ... --compound
```

Outputs: `canal_model.obj`, `canal_model.step`, `canal_section.step`

---

## Release Downloads (v1.0.0)

| Asset | Size | Purpose |
|---|---|---|
| [`depth_pro.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/depth_pro.msgpack) | 1.8 GB | Depth Pro JAX weights |
| [`MNT.xyz`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/MNT.xyz) | 1.1 GB | Pre-event terrain — Brague, Lambert-93, 4.9 mm |
| [`Ortho.tif`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/Ortho.tif) | 564 MB | Orthorectified GeoTIFF — 15228×13222 px, 2.4 mm/px |
| [`superpoint_lightglue.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint_lightglue.msgpack) | 46 MB | LightGlue JAX weights |
| [`superpoint.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint.msgpack) | 5 MB | SuperPoint JAX weights |
| [`IMG_1139.MOV`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/IMG_1139.MOV) | 23 MB | Flood video — downstream bridge |
| [`IMG_1142.MOV`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/IMG_1142.MOV) | 13 MB | Flood video — upstream bridge |

`pipeline.py` downloads all missing assets automatically on first run.

---

## Brague Flood Case Study

The Brague river flood at Biot (French Riviera, 23 November 2019).  
CRS: **EPSG:2154** (Lambert-93) — ortho and MNT share the same projection.

| Input | File | Spec |
|---|---|---|
| Pre-event terrain | `MNT.xyz` | 4.9 mm point spacing, Lambert-93 |
| Ortho image | `Ortho.tif` | 2.4 mm/px, 37 m × 32 m reach |
| Event video (downstream) | `IMG_1139.MOV` | 1920×1080 · 22 MB |
| Event video (upstream) | `IMG_1142.MOV` | 1920×1080 · 13 MB |

---

## Model Architecture

```
ReconstructionPipeline (modules/reconstruction.py)
├── DepthPro   → metric inverse depth + FOV per frame
├── SuperPoint → sparse keypoints + descriptors
├── LightGlue  → mutual-best keypoint matches
├── Concentric zones → 3 radial rings (reduce radial drift)
├── Kabsch  (rigid)      → R, t alignment
├── Umeyama (similarity) → s·R, t alignment  [recommended]
└── Open3D Poisson → .ply point cloud + .glb mesh
```

Weights: [github.com/1kaiser/d_jax/releases](https://github.com/1kaiser/d_jax/releases)

---

## Future Roadmap

![Future roadmap](assets/future_roadmap.png)

*How the existing pipeline outputs (blue) feed five planned extensions: GRAINnet grain-size inference (teal), JAX fluid solver for superstructure flow stability (green), JAX FEM for superstructure lining design (orange), sub-structure foundation design (red-orange), and surface-material → subsurface soil composition (purple).*

---

### GRAINnet — Bed Grain Size → Manning's n

Manning's roughness coefficient n is not a fixed lookup value — it is a direct function of the bed material gradation. The Strickler formula (for gravel/cobble beds) gives:

```
n = d90^(1/6) / K_s       K_s ≈ 21–26 m^(1/3)/s  (gravel bed, Bathurst 1985)
```

[`1kaiser/GRAINnet`](https://github.com/1kaiser/GRAINnet) infers the grain-size distribution (d50, d84, d90) directly from a river-bed image using deep learning, without field sieving.

```
Ortho.tif  ──► GRAINnet inference
                    │
                    ▼
             d50(X,Y), d84(X,Y), d90(X,Y)   [mm, spatially distributed]
                    │
         Strickler / Hey (1979) ─────────────┤
         n(X,Y) = d90(X,Y)^(1/6) / K_s      │
                                             ▼
                              Spatially variable Manning's n
                                             │
                    ┌────────────────────────┴──────────────────┐
                    │  Re-solve Manning Q with n(X,Y)           │
                    │  Q = (1/n)·A·R^(2/3)·√S                  │
                    │  → more accurate discharge estimate       │
                    │  → IS 10430 n=0.018 assumption validated  │
                    └───────────────────────────────────────────┘
```

**Why this matters:** The current pipeline uses a fixed n = 0.018 (IS 10430, concrete lining). The Brague at flood stage is a gravel-cobble bed (GW/GP class); d90 ≈ 50–150 mm gives n ≈ 0.025–0.040. A 40–100% change in n shifts Q by the same factor. GRAINnet removes the need for manual particle sampling at every cross-section and enables spatially varying roughness maps.

---

### JAX Fluid Solver — Superstructure Flow Stability

The D8 thalweg analysis already shows the Brague reach has a terrain slope of ~1:55 — well into the **supercritical flow** regime for any reasonable canal geometry. The JAX-based shallow-water solver closes the loop between the measured terrain and the designed section's hydraulic stability.

```
flow_depth.tif  ──► D8 thalweg  ──► S_long = 1:55 (measured)
canal_section   ──► Froude number Fr = V/√(g·D)
                         │
                    Fr > 1?  ──► supercritical: hydraulic jump, standing waves
                         │
                    JAX-CFD / Saint-Venant 2D
                    (google/jax-cfd or web/jax-js-fem)
                         │
                         ▼
              Q_sim(t) vs Q_target          [capacity validation]
              hydraulic jump location X_j   [where to place stilling basin]
              flow depth envelope h(x,t)    [overtopping check]
              energy grade line EGL(x)      [drop structure spacing]
```

For the Brague (S=1:55, Q=50 m³/s): drop structures must dissipate ~0.9 m of head per metre of channel — the fluid solver determines how many drops, their spacing, and whether the downstream section re-enters subcritical flow before the next structure.

---

### JAX FEM — Superstructure Lining Design (IS 456 / IS 3370)

Once the flow regime is known (hydrostatic + dynamic pressure, uplift during emptying), a JAX FEM solver computes the stress state in the concrete lining:

```
canal_section.step  ──► JAX FEM mesh (shell elements)
hydrostatic load        ──► p = ρ·g·h at each node
earth pressure (Ka)     ──► lateral soil thrust on walls
temperature gradient    ──► thermal shrinkage / expansion
IS 456:2000             ──► limit-state design: M_u, V_u, N_u
IS 3370:2009            ──► liquid-retaining: crack width ≤ 0.2 mm
                                │
                                ▼
                     Steel schedule (mm²/m)
                     Slab / wall thickness
                     Joint spacing (thermal)
```

Integration point: [`tianjuxue/jax-fem`](https://github.com/tianjuxue/jax-fem).

---

### JAX FEM — Sub-structure Foundation Design (IS 6403 / IS 8009)

The **sub-structure** (cut-off wall, apron, raft foundation, side-slope toe) sits below the canal invert and must resist seepage, scour, and bearing failure. The 1 m depth soil composition from GRAINnet / surface-material classification provides the input parameters without borehole data.

```
GRAINnet  d50(X,Y) ──► soil_class(X,Y): GW · SW · CL …
Surface → USCS class ──► φ (friction angle), c (cohesion), γ (unit weight)
D8 thalweg depth     ──► scour depth estimate (Lacey / Neill formula)
                                │
                    ┌───────────┴────────────────────────────────┐
                    │  JAX FEM sub-structure mesh                │
                    │  Bearing capacity:  q_ult = c·Nc + γ·D·Nq  │
                    │  Settlement:        δ from Boussinesq       │
                    │  Seepage uplift:    u = γ_w · h_w           │
                    │  Scour protection:  apron length (IS 8237)  │
                    │  IS 6403:1981       safe bearing capacity   │
                    │  IS 8009:1976       settlement calculation  │
                    └────────────────────────────────────────────┘
                                │
                                ▼
                     Foundation depth D_f (m)
                     Raft / strip footing dimensions
                     Cut-off wall depth (anti-seepage)
                     Toe protection apron length
```

---

### Surface Material → Subsurface Soil Composition (0–1 m)

Event-day imagery captures the material exposed at the bed. Object classification on the ortho frames maps surface texture to USCS geotechnical class, giving a coarse 1 m depth profile without borehole data. This feeds both the GRAINnet grain-size path (for Manning's n) and the sub-structure design path (for bearing capacity).

```
Ortho.tif  ──► SegNet / DINO features  ──► surface class map
                                               │
                      USCS / IS 1498 ──────────┤  texture → soil class
                      GRAINnet d50   ──────────┤  grain size → USCS GW/SW/CL
                                               ▼
                              soil_class(X,Y)  ──► φ, c, γ, k (permeability)
                              composition(0–1 m)  ──► Boussinesq settlement
                                               │
                               ┌───────────────┴──────────────┐
                               │  Sub-structure  (JAX FEM)    │
                               │  q_ult, D_f, δ_settle        │
                               └──────────────────────────────┘
```

**Connection chain:** bed surface texture (GW/GP cobble gravel) → Strickler n → hydraulic design → hydrostatic load → superstructure lining FEM → sub-structure foundation FEM → IS 6403 safe bearing capacity. One image, full design dossier.

---

## Credits

- **opyflow** LSPIV: [groussea/opyflow](https://github.com/groussea/opyflow) — Rousseau (2019)
- **Brague dataset**: Vigoureux et al., SimHydro 2021
- **blender-colab skeleton**: [ynshung/blender-colab](https://github.com/ynshung/blender-colab)

Created and maintained by [1kaiser](https://github.com/1kaiser)
