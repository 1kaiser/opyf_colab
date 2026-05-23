# %% [markdown]
# # opyf_colab — Hydraulic Analysis Pipeline
#
# **Brague flood event reconstruction** — monocular depth · JAX LSPIV ·
# D8 thalweg · IS 10430 canal design.
#
# Run all cells top-to-bottom, or via Papermill:
# ```bash
# JAX_PLATFORMS=cpu conda run -n num_gpu papermill pipeline_nb.ipynb pipeline_nb.ipynb \
#   -p SKIP_DOWNLOAD True -p SKIP_DEPTH True -p SKIP_ALIGN True
# ```

# %% tags=["parameters"]
# Papermill parameters — override any value from the CLI with -p KEY VALUE

VIDEO_DOWN    = "data/brague/IMG_1139.MOV"
VIDEO_UP      = "data/brague/IMG_1142.MOV"
MNT_XYZ       = "data/brague/MNT.xyz"
ORTHO_TIF     = "data/brague/Ortho.tif"
DP_WEIGHTS    = "weights/depth_pro.msgpack"
SP_WEIGHTS    = "weights/superpoint.msgpack"
LG_WEIGHTS    = "weights/superpoint_lightglue.msgpack"
OUT_DIR       = "output/brague"
ASSETS_DIR    = "assets"
N_FRAMES      = 5

SKIP_DOWNLOAD  = True
SKIP_DEPTH     = True
SKIP_ALIGN     = True
SKIP_D8        = False
SKIP_PC_CHECK  = False
SKIP_CANAL     = False
SKIP_CANAL_VIZ = False
SKIP_LSPIV     = False
SKIP_VIZ       = False

# %%
import os, sys, json, types, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.chdir(Path(__file__).parent if "__file__" in dir() else Path("."))

print("Working dir :", Path(".").resolve())
print("Python      :", sys.executable)

# %% [markdown]
# ## Stage 0 — Download release assets
# Downloads videos, weights, MNT, Ortho from the GitHub release if absent.

# %%
if not SKIP_DOWNLOAD:
    from pipeline import download_assets
    download_assets()
else:
    print("Stage 0 skipped — assets assumed present")

# %% [markdown]
# ## Stages 1–5 — Depth inference pipeline
#
# Extracts frames → Depth Pro inference → GCP-aligned metric depth →
# `flow_depth.tif`  (aggregated median over N frames).

# %%
meta_path = Path(OUT_DIR) / "pipeline_meta.json"
flow_tif  = Path(OUT_DIR) / "flow_depth.tif"

if SKIP_DEPTH and flow_tif.exists() and meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    print("Depth pipeline skipped — loaded", meta_path)
    print(f"  h_mean={meta['h_final_mean']:.3f} m   h_max={meta['h_final_max']:.3f} m")
else:
    args = types.SimpleNamespace(
        video=VIDEO_DOWN, mnt=MNT_XYZ, ortho=ORTHO_TIF,
        weights=DP_WEIGHTS, out_dir=OUT_DIR, assets=ASSETS_DIR,
        n_frames=N_FRAMES, sp_weights=SP_WEIGHTS, lg_weights=LG_WEIGHTS,
    )
    from pipeline import run_depth_pipeline
    meta = run_depth_pipeline(args)

# %% [markdown]
# ## Stage 6b — Point cloud alignment + water volume
# LightGlue bank-feature matching → homography → inundated volume (m³).

# %%
if not SKIP_ALIGN:
    import numpy as np, rasterio as _rio
    from modules.depth_to_elevation import load_mnt, rasterise_mnt
    from pipeline import run_alignment_stage

    with _rio.open(ORTHO_TIF) as src:
        _tf    = src.transform
        _shape = (src.height, src.width)
    X_m, Y_m, Z_m = load_mnt(MNT_XYZ, subsample=5)
    z_bed = rasterise_mnt(X_m, Y_m, Z_m, _tf, _shape)

    ar = run_alignment_stage(
        Path(OUT_DIR), z_bed, _tf, X_m, Y_m, Z_m,
        Path(ASSETS_DIR), SP_WEIGHTS, LG_WEIGHTS,
    )
    if ar:
        print(f"Water volume  : {ar['volume_m3']:,.0f} m³")
        print(f"Inundated area: {ar['area_m2']:,.0f} m²")
else:
    print("Stage 6b skipped")

# %% [markdown]
# ## Stage 6c — JAX D8 thalweg
# Fill sinks → D8 flow directions → accumulation → thalweg trace →
# slope / curvature / bearing profile.
# Output: `assets/d8_thalweg.png`

# %%
d8_result = None
if not SKIP_D8:
    from pipeline import run_d8_thalweg
    d8_result = run_d8_thalweg(Path(OUT_DIR), Path(ASSETS_DIR))
    if d8_result:
        g = d8_result["geometry"]
        print(f"Thalweg  L={g['reach_length_m']:.1f} m   S={g['S_long']:.5f}   "
              f"R_min={g['min_radius_m']:.1f} m")
else:
    print("Stage 6c skipped")

# %%
from IPython.display import Image, display
for f in ["d8_thalweg.png"]:
    p = Path(ASSETS_DIR) / f
    if p.exists(): display(Image(str(p), width=700))

# %% [markdown]
# ## Stage 6d — Pointcloud × Ortho alignment check
# Drapes Ortho.tif RGB onto MNT.xyz elevation; 4-panel diagnostic.
# Output: `assets/pointcloud_ortho_check.png`

# %%
if not SKIP_PC_CHECK:
    from pipeline import run_pc_ortho_check
    run_pc_ortho_check(MNT_XYZ, ORTHO_TIF, Path(ASSETS_DIR))
else:
    print("Stage 6d skipped")

# %%
p = Path(ASSETS_DIR) / "pointcloud_ortho_check.png"
if p.exists(): display(Image(str(p), width=900))

# %% [markdown]
# ## Stage 7 — JAX canal optimiser (IS 10430)
# Manning-Strickler minimum wetted-perimeter section, IS velocity compliance.
# Output: `canal_design/canal_params.json`,
# `assets/canal_section.png`, `assets/design_chain.png`

# %%
canal_dir = Path("canal_design")
cp_path   = canal_dir / "canal_params.json"

if SKIP_CANAL and cp_path.exists():
    with open(cp_path) as f:
        canal_params = json.load(f)
    print("Canal optimizer skipped — loaded", cp_path)
else:
    from pipeline import run_canal_optimizer
    canal_params = run_canal_optimizer(meta, canal_dir, Path(ASSETS_DIR))

print(f"B={canal_params['bed_width_m']:.3f} m  D={canal_params['water_depth_m']:.3f} m  "
      f"Q={canal_params['Q_calculated_m3s']:.2f} m³/s  V={canal_params['velocity_ms']:.3f} m/s")

# %%
for f in ["canal_section.png", "design_chain.png"]:
    p = Path(ASSETS_DIR) / f
    if p.exists(): display(Image(str(p), width=700))

# %% [markdown]
# ## Stage 7b — Canal 3D overlay
# Designed section rendered over reconstructed terrain + ortho background,
# D8 thalweg overlaid. Output: `assets/canal_3d_overlay.png`

# %%
reach_geo = None
if not SKIP_CANAL_VIZ:
    from pipeline import run_canal_3d_viz
    reach_geo = run_canal_3d_viz(
        canal_params, Path(OUT_DIR), ORTHO_TIF, Path(ASSETS_DIR), d8_result,
    )
else:
    print("Stage 7b skipped")

# %%
p = Path(ASSETS_DIR) / "canal_3d_overlay.png"
if p.exists(): display(Image(str(p), width=900))

# %% [markdown]
# ## Stage 7c — Canal 4-view CAD drawing
# Front / Side / Top / Isometric with dimension lines (IS 10430).
# Output: `assets/canal_cad_model.png`

# %%
if not SKIP_CANAL_VIZ:
    from pipeline import run_canal_cad
    run_canal_cad(canal_params, reach_geo, Path(ASSETS_DIR))
else:
    print("Stage 7c skipped")

# %%
p = Path(ASSETS_DIR) / "canal_cad_model.png"
if p.exists(): display(Image(str(p), width=900))

# %% [markdown]
# ## Stage 7d — JAX LSPIV surface velocity + discharge
# DLT orthorectification → FFT phase-correlation PIV → Gaussian RBF
# interpolation → Q = α ∫ V·h dl (α=0.9).
#
# Processes both bridges (IMG_1139 downstream + IMG_1142 upstream).
#
# Also generates three **opyflow-equivalent figures**:
#
# | opyflow original | JAX equivalent |
# |---|---|
# | `birdEyeTransf1139.png` | `assets/opyflow_birdeye.png` |
# | `1139.png` + `1142.png` | `assets/opyflow_velocity_field.png` |
# | `figure_Brague.png` | `assets/figure_brague.png` |

# %%
lspiv_result = None
if not SKIP_LSPIV:
    from pipeline import run_lspiv
    lspiv_result = run_lspiv(
        video_down=VIDEO_DOWN,
        video_up=VIDEO_UP,
        mnt_xyz=MNT_XYZ,
        ortho_tif=ORTHO_TIF,
        out_dir=Path(OUT_DIR),
        assets=Path(ASSETS_DIR),
    )
    if lspiv_result:
        print(f"Combined Q = {lspiv_result['Q']:.2f} m³/s")
else:
    print("Stage 7d skipped")

# %%
for f in ["lspiv_results.png", "opyflow_birdeye.png",
          "opyflow_velocity_field.png", "figure_brague.png"]:
    p = Path(ASSETS_DIR) / f
    if p.exists():
        print(f)
        display(Image(str(p), width=900))

# %% [markdown]
# ## Stage 8 — Annotated pipeline visualisation + future roadmap
# Eight-panel summary figure covering the full pipeline.
# Output: `assets/annotated_pipeline.png`, `assets/future_roadmap.png`

# %%
if not SKIP_VIZ:
    from pipeline import run_visualisation
    run_visualisation(Path(ASSETS_DIR))
else:
    print("Stage 8 skipped")

# %%
for f in ["annotated_pipeline.png", "future_roadmap.png"]:
    p = Path(ASSETS_DIR) / f
    if p.exists():
        print(f)
        display(Image(str(p), width=900))

# %% [markdown]
# ## Results summary

# %%
print("=" * 62)
print("  opyf_colab — Brague Flood 2019 · Key Results")
print("=" * 62)
print(f"  Flow depth h_mean        = {meta.get('h_final_mean', 0):.3f} m")
print(f"  Flow depth h_max         = {meta.get('h_final_max', 0):.3f} m")
if d8_result:
    g = d8_result["geometry"]
    print(f"  Thalweg length           = {g['reach_length_m']:.1f} m")
    print(f"  Thalweg slope            = {g['S_long']:.5f}")
print(f"  Canal bed width B        = {canal_params['bed_width_m']:.3f} m")
print(f"  Canal water depth D      = {canal_params['water_depth_m']:.3f} m")
print(f"  Canal Q (IS 10430)       = {canal_params['Q_calculated_m3s']:.2f} m³/s")
if lspiv_result:
    print(f"  LSPIV discharge Q        = {lspiv_result['Q']:.2f} m³/s")
    print(f"  opyflow paper reference  = 102 ± 20 m³/s")
print("=" * 62)
