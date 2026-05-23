# opyf_colab

A high-performance research ecosystem for fluid velocimetry (PIV), 3D metric reconstruction, and automated civil engineering design, optimized for JAX-based computation and Google Colab.

**[Open In Colab](https://colab.research.google.com/github/1kaiser/opyf_colab/blob/main/jax_3d_reconstruction_colab.ipynb)**

---

## Pipeline Overview

The full hydraulic analysis pipeline converts raw field data (video + aerial imagery) into an optimised canal design through a differentiable closed loop:

```
Event-day video          Pre-event ortho + MNT terrain
       │                          │
       ▼                          ▼
 Depth Pro JAX           MNT.xyz rasterised
 (metric depth)          (bed elevation Z_bed)
       │                          │
       └────────────┬─────────────┘
                    ▼
         Z_surface − Z_bed = h(x,y)   [flow depth]
                    │
                    ▼
         Q = ∫ α · V(x,y) · h(x,y) dA   [discharge]
                    │
                    ▼
         JAX Canal Optimizer  (IS 5968 / IS 10430)
                    │
                    ▼
         FreeCAD 3D + 2D engineering drawings
```

![Annotated pipeline](assets/annotated_pipeline.png)

---

## Quick Start

```bash
# Full pipeline — auto-downloads all assets from the release, then runs
JAX_PLATFORMS=cpu python3 pipeline.py

# Skip download if assets already present, skip re-running depth inference
JAX_PLATFORMS=cpu python3 pipeline.py --skip-download --skip-depth --skip-canal
```

`pipeline.py` auto-downloads all release assets (video, weights, MNT, ortho) via wget on first run.  
All outputs land in `output/brague/` and the annotated figure at `assets/annotated_pipeline.png`.

---

## Repository Structure

```
opyf_colab/
├── pipeline.py                      Single entry-point — all stages, auto-download
├── modules/                         All pipeline components
│   ├── depth_to_elevation.py        Frames → Z_surface → flow depth raster
│   ├── canal_optimizer.py           JAX optimizer + IS 5968/10430 compliance
│   ├── canal_cad.py                 FreeCAD 3D sweep + 2D section + reach model
│   ├── infer_depth.py               Depth Pro CLI wrapper
│   ├── infer_features.py            SuperPoint + LightGlue + homography
│   ├── infer_stereo.py              MASt3R dense stereo
│   ├── infer_video.py               VGGT video geometry
│   ├── segment_water.py             Water mask: color / diff / SegNet (Flax)
│   ├── reconstruction.py            Kabsch/Umeyama alignment, point cloud utils
│   ├── visualise_comparison.py      Ortho vs depth multi-view comparison
│   └── annotated_pipeline_viz.py    5-stage annotated figure → assets/
├── assets/
│   └── annotated_pipeline.png       Generated pipeline figure
├── data/brague/
│   ├── MNT.xyz                      Pre-event terrain (auto-downloaded)
│   ├── Ortho.tif                    Orthorectified aerial image (auto-downloaded)
│   └── dxf_data.json                Cross-section survey data
├── canal_design/
│   └── canal_params.json            IS-compliant canal dimensions (optimizer output)
├── models/jax/                      JAX model implementations
│   ├── jax_depth_pro/               Depth Pro (ViT + decoder)
│   ├── jax_lightglue/               SuperPoint + LightGlue
│   ├── jax_mast3r/                  MASt3R dense stereo
│   ├── jax_reconstruction/          Geometry utils
│   └── jax_vggt/                    VGGT video geometry
├── tests/Test_Brague_flood/
│   ├── IMG_1139.MOV                 Flood video downstream (auto-downloaded)
│   ├── IMG_1142.MOV                 Flood video upstream (auto-downloaded)
│   └── Brague_Flood_LSPIV.ipynb
├── pipelines/                       Legacy stand-alone scripts (kept for reference)
└── web/jax-js-fem/                  TypeScript JAX-in-browser canal optimizer
```

---

## Release Downloads (v1.0.0)

| Asset | Size | Purpose |
|---|---|---|
| [`depth_pro.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/depth_pro.msgpack) | 1.8 GB | Depth Pro JAX weights |
| [`MNT.xyz`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/MNT.xyz) | 1.1 GB | Pre-event terrain model — Brague river |
| [`Ortho.tif`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/Ortho.tif) | 564 MB | Orthorectified GeoTIFF — 15228×13222 px |
| [`superpoint_lightglue.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint_lightglue.msgpack) | 46 MB | LightGlue JAX weights |
| [`superpoint.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint.msgpack) | 5 MB | SuperPoint JAX weights |
| [`IMG_1139.MOV`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/IMG_1139.MOV) | 23 MB | Flood video — downstream bridge |
| [`IMG_1142.MOV`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/IMG_1142.MOV) | 13 MB | Flood video — upstream bridge |

`pipeline.py` downloads all missing assets automatically on first run via wget.

---

## Brague Flood Case Study

The Brague river flood at Biot (French Riviera, 23 November 2019).

| Input | File | Resolution |
|---|---|---|
| Pre-event terrain | `MNT.xyz` | 4.9 mm point spacing |
| Ortho image | `Ortho.tif` | 2.4 mm/pixel, 37 m × 32 m reach |
| Event video (downstream) | `tests/.../IMG_1139.MOV` | 1920×1080, 22 MB |
| Event video (upstream) | `tests/.../IMG_1142.MOV` | 1920×1080, 12 MB |

CRS: **EPSG:2154** (Lambert-93) — ortho and MNT share the same projection.

**Results:**

| Metric | Value |
|---|---|
| Flow depth h_mean | **1.152 m** |
| Flow depth h_max | **2.124 m** |
| Wet area | ~6.9 M pixels (~39,700 m²) |
| Depth Pro FOV | 61–63° |
| Scale factor s | 0.030–0.035 |
| Z offset t | ~11.9 m (≈ mean bed elevation) |
| Canal bed width (IS design) | 49.6 m |
| Canal water depth | 1.55 m |

---

## Flow Depth Computation

```
Z_surface(x,y)  — Depth Pro inference on event frames, registered to Lambert-93
−  Z_bed(x,y)   — MNT.xyz rasterised to ortho grid (4.9 mm → 2.4 mm bilinear)
= h(x,y)        — flow depth in metres at each ortho pixel

Discharge:  Q = ∫ α · V_surface(x,y) · h(x,y) dA
                α = 0.9 ± 0.1  (Welber et al. 2016)
                V_surface from LSPIV (opyflow)
```

Scale solve (dry-pixel constraint): `Z_abs = s · d_Depth_Pro + t`

---

## Canal Design

The JAX optimizer finds a minimum-cost trapezoidal section satisfying Manning's equation:

```bash
# Standalone optimizer
JAX_PLATFORMS=cpu python3 canal_design/jax_canal_optimizer.py

# 3D model + 2D drawings (requires FreeCAD)
freecadcmd canal_design/generate_canal_assets.py \
    --params canal_design/canal_params.json --output output/canal_assets
```

IS code compliance: IS 5968:1987 (curve radius), IS 10430:2000 (freeboard, side slopes, velocity ≤ 2.5 m/s).

---

## Model Architecture

```
ReconstructionPipeline (modules/reconstruction.py)
├── DepthPro   → metric inverse depth + FOV per frame
├── SuperPoint → sparse keypoints + descriptors
├── LightGlue  → mutual-best keypoint matches
├── Concentric zones → 3 radial rings, reduce radial drift
├── Kabsch  (rigid)      → R, t alignment
├── Umeyama (similarity) → s·R, t alignment  [recommended]
└── Open3D Poisson → .ply point cloud + .glb mesh
```

Weights: [github.com/1kaiser/d_jax/releases](https://github.com/1kaiser/d_jax/releases)

---

## Credits

- **opyflow** LSPIV: [groussea/opyflow](https://github.com/groussea/opyflow) — Rousseau (2019)
- **Brague dataset**: Vigoureux et al., SimHydro 2021
- **blender-colab skeleton**: [ynshung/blender-colab](https://github.com/ynshung/blender-colab)

Created and maintained by [1kaiser](https://github.com/1kaiser)
