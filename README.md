# opyf_colab 🚀

A high-performance research ecosystem for fluid velocimetry (PIV), 3D metric reconstruction, and automated civil engineering design, optimized for JAX-based computation and Google Colab.

## ✅ [Open In Colab](https://colab.research.google.com/github/1kaiser/opyf_colab/blob/main/jax_3d_reconstruction_colab.ipynb)

---

## 🌊 Pipeline Overview

The full hydraulic analysis pipeline converts raw field data (video + aerial imagery) into an optimised canal design through a **differentiable closed loop**:

```
📷 Multi-view images          🎥 Event-day video
        │                             │
        ▼                             ▼
  Depth Pro JAX               LSPIV (opyflow)
  (metric depth)              (surface velocity)
        │                             │
        ▼                             │
  Z_surface(x,y)                      │
  − Z_bed  (MNT)                      │
  = h(x,y) flow depth                 │
        │                             │
        └──────────┬──────────────────┘
                   ▼
        Q = ∫ α · V(x,y) · h(x,y) dA
        (discharge across transect)
                   │
                   ▼
        JAX Canal Optimizer (IS 5968 / IS 10430)
        (Manning's equation, gradient descent)
                   │
                   ▼
        jax-cfd simulation → validate Q_sim ≈ Q_target
                   │
                   ▼
        MCTS iteration → optimal (B, D, S_side, slope)
                   │
                   ▼
        FreeCAD 3D + 2D engineering drawings
```

---

## 📁 Repository Structure

```
opyf_colab/
├── data/
│   ├── pinecone_subset/        📸 10 frames for 3D reconstruction testing
│   └── brague/                 🌊 Brague flood event data
│       ├── MNT.xyz             Pre-event terrain model (Lambert-93, 4.9mm spacing)
│       └── Ortho.tif           Orthorectified aerial image (2.4mm/px, EPSG:2154)
├── models/jax/
│   ├── jax_depth_pro/          Depth Pro — metric depth from single image
│   ├── jax_lightglue/          SuperPoint + LightGlue — feature matching
│   ├── jax_mast3r/             MASt3R — dense stereo
│   ├── jax_reconstruction/     Geometry utils (Kabsch, Umeyama, lift_points)
│   └── jax_vggt/               VGGT — video geometry estimation
├── pipelines/
│   ├── pipeline_jax.py         3D reconstruction (DepthPro + LightGlue + Open3D)
│   └── depth_to_elevation.py   🆕 Flow depth raster from event frames + MNT
├── canal_design/
│   ├── jax_canal_optimizer.py  JAX gradient descent → IS-code optimal dimensions
│   ├── design_canal_is_v2.py   IS 5968/10430 calculator + FreeCAD 3D
│   ├── generate_canal_assets.py Full assets: 3D body + 2D section + elevation
│   ├── generate_reach_cad.py   DXF survey data → FreeCAD 3D reconstruction
│   └── design_combined_canal.py Compound cross-section (multi-stage canals)
├── inference/
│   ├── infer_depth_pro.py      Single-image metric depth
│   ├── infer_lightglue.py      Two-image keypoint matching + visualisation
│   ├── infer_mast3r.py         Dense stereo inference
│   └── infer_vggt.py           Video geometry inference
├── tests/Test_Brague_flood/
│   ├── IMG_1139.MOV            📹 Flood video — downstream bridge (22 MB)
│   ├── IMG_1142.MOV            📹 Flood video — upstream bridge (12 MB)
│   ├── 1139/birdEyeTransf1139.png  Orthorectified bird-eye frame
│   ├── 1139/mask_1139.png      Stabilisation mask (1920×1080)
│   └── Brague_Flood_LSPIV.ipynb   LSPIV analysis notebook
├── jax_3d_reconstruction_colab.ipynb   Core 3D vision notebook
├── jax_3d_canal_reconstruction.ipynb   Canal design notebook
└── opyf_Eumetsat_velocimetry.ipynb     Satellite velocimetry notebook
```

---

## 📦 Release Downloads (v1.0.0)

| Asset | Size | Purpose |
|---|---|---|
| [`depth_pro.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/depth_pro.msgpack) | 1.8 GB | Depth Pro JAX weights — metric depth |
| [`MNT.xyz`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/MNT.xyz) | 1.1 GB | Pre-event terrain model — Brague river, Lambert-93, 4.9 mm spacing |
| [`Ortho.tif`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/Ortho.tif) | 564 MB | Orthorectified aerial GeoTIFF — 15228×13222 px, **2.4 mm/pixel** |
| [`superpoint_lightglue.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint_lightglue.msgpack) | 46 MB | LightGlue JAX weights |
| [`superpoint.msgpack`](https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0/superpoint.msgpack) | 5 MB | SuperPoint JAX weights |

Download to `weights/` and `data/brague/` before running the pipeline.

---

## 🌊 Brague Flood Case Study — Input Data

The Brague river flood at Biot (French Riviera, 23 November 2019) is the primary test case.

| Input | File | Resolution |
|---|---|---|
| Pre-event terrain | `MNT.xyz` | 4.9 mm point spacing |
| Ortho image | `Ortho.tif` | **2.4 mm/pixel**, covers 37 m × 32 m reach |
| Event video (downstream) | `tests/.../IMG_1139.MOV` | 1920×1080, 22 MB |
| Event video (upstream) | `tests/.../IMG_1142.MOV` | 1920×1080, 12 MB |
| Cross-section survey | `canal_design/dxf_data.json` | Ground level, water level, stone pitching layers |

The ortho and MNT share the same CRS (**EPSG:2154** Lambert-93), allowing direct pixel↔elevation alignment.

---

## 🔬 Flow Depth Computation

The `pipelines/depth_to_elevation.py` module computes **pixel-accurate flow depth** from event-day imagery:

```
Z_surface(x,y)  — Depth Pro inference on event frames, registered to Lambert-93
−  Z_bed(x,y)   — MNT.xyz rasterised to ortho grid (4.9 mm → 2.4 mm bilinear)
= h(x,y)        — flow depth in metres at each ortho pixel

Discharge:  Q = ∫ α · V_surface(x,y) · h(x,y) dA
                ↑ LSPIV surface velocity (opyflow)
                α = 0.9 ± 0.1  (Welber et al. 2016)
```

Run:
```bash
JAX_PLATFORMS=cpu python3 pipelines/depth_to_elevation.py \
    --video   tests/Test_Brague_flood/IMG_1139.MOV \
    --mnt     data/brague/MNT.xyz \
    --ortho   data/brague/Ortho.tif \
    --weights weights/depth_pro.msgpack \
    --out-dir output/brague \
    --n-frames 5
```

Outputs: `output/brague/flow_depth.tif`, `z_surface.tif`, `pipeline_meta.json`

---

## 🏗️ Canal Design

The JAX optimizer finds the minimum-cost trapezoidal section satisfying Manning's equation and IS codes:

```bash
# Step 1 — optimise hydraulic dimensions
JAX_PLATFORMS=cpu python3 canal_design/jax_canal_optimizer.py
# → canal_params.json

# Step 2 — generate 3D model + 2D drawings (requires FreeCAD)
freecadcmd canal_design/generate_canal_assets.py \
    --params canal_params.json --output output/canal_assets
```

**IS code compliance:**
- IS 5968:1987 — minimum curve radius by discharge class
- IS 10430:2000 — freeboard, side slopes, velocity limits (max 2.5 m/s concrete)

---

## 📊 Model Architecture

```
jax_3d_reconstruction_colab.ipynb
└── ReconstructionPipeline (pipelines/pipeline_jax.py)
    ├── DepthPro   → metric inverse depth + FOV per frame
    ├── SuperPoint → sparse keypoints + descriptors
    ├── LightGlue  → mutual-best keypoint matches
    ├── Concentric zones → 3 radial rings, reduce radial drift
    ├── Kabsch  (rigid)      → R, t alignment
    ├── Umeyama (similarity) → s·R, t alignment  [recommended]
    └── Open3D Poisson → .ply point cloud + .glb mesh
```

**Weights:** Available at [github.com/1kaiser/d_jax/releases](https://github.com/1kaiser/d_jax/releases)

---

## ⚖️ IS Code Reference

| Standard | Clause | Parameter |
|---|---|---|
| IS 5968:1987 | Table 1, Cl 8.1 | Minimum curve radius by Q |
| IS 10430:2000 | Table 1 | Freeboard by Q |
| IS 10430:2000 | Table 2 | Side slope (concrete: 1.5:1) |
| IS 10430:2000 | Cl 4.1 | Manning's n = 0.018 (concrete) |
| IS 10430:2000 | Cl 4.2 | Max velocity 2.5 m/s |

---

## Credits

- **opyflow** LSPIV algorithm: [groussea/opyflow](https://github.com/groussea/opyflow) — Rousseau (2019)
- **Brague dataset**: Vigoureux et al., SimHydro 2021
- **blender-colab skeleton**: [ynshung/blender-colab](https://github.com/ynshung/blender-colab)

---

## Disclaimer

Created and maintained by [1kaiser](https://github.com/1kaiser)
