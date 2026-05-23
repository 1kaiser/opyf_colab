"""
pipeline.py — single entry-point for the opyf_colab hydraulic analysis pipeline.

Stages
------
0.  Download release assets (video, weights, ortho, MNT) if missing
1.  Extract frames from event video
2.  Load Depth Pro weights
3.  Load ortho + MNT bed
4.  Per-frame depth inference + GCP alignment (LightGlue) + flow depth
5.  Aggregate frames (median) → flow_depth.tif
6.  Point cloud alignment + water volume → assets/alignment_*.png
6c. D8 thalweg extraction → assets/d8_thalweg.png
6d. Pointcloud × Ortho alignment check → assets/pointcloud_ortho_check.png
7.  Run JAX canal optimizer
7b. Canal 3D overlay + D8 overlay → assets/canal_3d_overlay.png
7c. Canal 4-view CAD drawing → assets/canal_cad_model.png
7d. JAX LSPIV surface-velocity + discharge (IMG_1139 + IMG_1142)
    → assets/lspiv_results.png
8.  Build annotated visualisation → assets/annotated_pipeline.png

Usage
-----
    JAX_PLATFORMS=cpu python3 pipeline.py [options]

Options
-------
--video        PATH  Downstream bridge video (default: data/brague/IMG_1139.MOV)
--video-up     PATH  Upstream bridge video   (default: data/brague/IMG_1142.MOV)
--mnt          PATH  MNT .xyz point cloud    (default: data/brague/MNT.xyz)
--ortho        PATH  Orthorectified GeoTIFF  (default: data/brague/Ortho.tif)
--weights      PATH  Depth Pro .msgpack weights (default: weights/depth_pro.msgpack)
--out-dir      PATH  Intermediate outputs directory (default: output/brague)
--assets       PATH  Final assets directory (default: assets)
--n-frames     N     Number of frames to sample (default: 5)
--skip-download      Skip automatic asset download
--skip-depth         Skip depth inference if output/brague/flow_depth.tif exists
--skip-align         Skip point cloud alignment + water volume visualisation
--skip-d8            Skip D8 thalweg extraction
--skip-pc-check      Skip pointcloud × ortho alignment check
--skip-canal         Skip canal optimisation if canal_design/canal_params.json exists
--skip-canal-viz     Skip canal 3D overlay and CAD drawing
--skip-lspiv         Skip JAX LSPIV discharge computation
--skip-viz           Skip annotated pipeline visualisation
--sp-weights   PATH  SuperPoint weights (default: weights/superpoint.msgpack)
--lg-weights   PATH  LightGlue weights  (default: weights/superpoint_lightglue.msgpack)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent

_RELEASE_BASE = "https://github.com/1kaiser/opyf_colab/releases/download/v1.0.0"

# All assets that can be auto-downloaded, mapped to their local paths
_RELEASE_ASSETS = {
    "depth_pro.msgpack":            "weights/depth_pro.msgpack",
    "superpoint.msgpack":           "weights/superpoint.msgpack",
    "superpoint_lightglue.msgpack": "weights/superpoint_lightglue.msgpack",
    "MNT.xyz":                      "data/brague/MNT.xyz",
    "Ortho.tif":                    "data/brague/Ortho.tif",
    "IMG_1139.MOV":                 "data/brague/IMG_1139.MOV",
    "IMG_1142.MOV":                 "data/brague/IMG_1142.MOV",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_file(path: Path, label: str):
    if not path.exists():
        print(f"  [ERROR] {label} not found: {path}")
        sys.exit(1)
    print(f"  {label}: {path}")


# ── stage 0: asset download ───────────────────────────────────────────────────

def download_assets(required_only: list[str] | None = None):
    """Download missing release assets via wget. Only downloads if file absent."""
    banner("STAGE 0: Download release assets")
    any_downloaded = False

    for filename, local_rel in _RELEASE_ASSETS.items():
        local = REPO / local_rel
        if required_only and local_rel not in required_only:
            continue
        if local.exists():
            print(f"  ✓ {local_rel}")
            continue

        url = f"{_RELEASE_BASE}/{filename}"
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {filename} ({url}) …")
        result = subprocess.run(
            ["wget", "--continue", "--quiet", "--show-progress", "-O", str(local), url],
            check=False,
        )
        if result.returncode != 0:
            print(f"  [WARN] wget failed for {filename} — check URL or download manually")
        else:
            print(f"  → {local}")
            any_downloaded = True

    if not any_downloaded:
        print("  All assets already present.")


# ── stage 1-6: depth pipeline ────────────────────────────────────────────────

def run_depth_pipeline(args) -> dict:
    """Run frames→flow_depth via depth_to_elevation module. Returns pipeline_meta dict."""
    from modules.depth_to_elevation import (
        extract_frames, load_depth_pro, estimate_dry_mask,
        load_mnt, rasterise_mnt, infer_depth,
        solve_depth_scale, compute_flow_depth, aggregate_frames,
    )
    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 – frames
    banner("STAGE 1: Extract event frames")
    frames = extract_frames(args.video, out_dir / "frames", args.n_frames)

    # Stage 2 – model
    banner("STAGE 2: Load Depth Pro")
    jit_fn, variables = load_depth_pro(args.weights)

    # Stage 3 – ortho / dry mask
    banner("STAGE 3: Load ortho → dry mask")
    dry_mask, transform, ortho_shape = estimate_dry_mask(args.ortho)
    print(f"  Ortho shape: {ortho_shape}  dry pixels: {dry_mask.sum():,} / {dry_mask.size:,}")

    # Stage 4 – MNT
    banner("STAGE 4: Load & rasterise MNT")
    X, Y, Z = load_mnt(args.mnt, subsample=5)
    z_bed = rasterise_mnt(X, Y, Z, transform, ortho_shape)
    print(f"  z_bed valid: {np.isfinite(z_bed).sum():,} / {z_bed.size:,}  "
          f"Z=[{np.nanmin(z_bed):.2f}, {np.nanmax(z_bed):.2f}] m")

    # Stage 5 – per-frame inference
    banner("STAGE 5: Per-frame depth inference + elevation")
    frame_metas = []
    for fp in sorted(frames):
        t0 = time.time()
        inv_depth, fov_deg = infer_depth(fp, jit_fn, variables)
        dt = time.time() - t0

        s, t_off = solve_depth_scale(inv_depth, z_bed, dry_mask)
        z_surface, h = compute_flow_depth(inv_depth, z_bed, s, t_off, ortho_shape)

        stem = Path(fp).stem
        import rasterio
        from rasterio.transform import from_bounds
        import rasterio.crs

        with rasterio.open(args.ortho) as src:
            crs = src.crs
            bounds = src.bounds
            tf = from_bounds(*bounds, z_surface.shape[1], z_surface.shape[0])

        profile = dict(driver="GTiff", dtype="float32", count=1,
                       crs=crs, transform=tf,
                       width=z_surface.shape[1], height=z_surface.shape[0],
                       compress="deflate")

        zs_path = out_dir / f"{stem}_z_surface.tif"
        hd_path  = out_dir / f"{stem}_flow_depth.tif"
        with rasterio.open(zs_path, "w", **profile) as dst:
            dst.write(z_surface.astype("float32"), 1)
        with rasterio.open(hd_path, "w", **profile) as dst:
            dst.write(h.astype("float32"), 1)

        h_mean = float(np.nanmean(h[h > 0]))
        h_max  = float(np.nanmax(h))
        wet_px = int((h > 0).sum())
        print(f"  [{stem}]  fov={fov_deg:.1f}°  s={s:.4f}  t={t_off:.4f}  "
              f"h_mean={h_mean:.3f} m  wet_px={wet_px:,}  ({dt:.1f}s)")

        frame_metas.append(dict(frame=str(fp), fov_deg=float(fov_deg),
                                scale_s=float(s), offset_t=float(t_off),
                                h_mean_m=h_mean, h_max_m=h_max,
                                wet_pixels=wet_px, infer_time_s=round(dt, 2)))

    # Stage 6 – aggregate
    banner("STAGE 6: Aggregate frames (median)")
    h_final = aggregate_frames(out_dir, ortho_shape, transform)
    h_final_mean = float(np.nanmean(h_final[h_final > 0]))
    h_final_max  = float(np.nanmax(h_final))
    print(f"  Aggregated {len(frames)} frames  h_mean={h_final_mean:.3f} m  h_max={h_final_max:.3f} m")

    meta = dict(
        frames=frame_metas,
        ortho_shape=list(ortho_shape),
        mnt_subsample=5,
        n_frames=len(frames),
        z_bed_mean=float(np.nanmean(z_bed)),
        h_final_mean=h_final_mean,
        h_final_max=h_final_max,
    )
    meta_path = out_dir / "pipeline_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")
    return meta


# ── stage 6b: point cloud alignment + water volume ───────────────────────────

def run_alignment_stage(
    out_dir: Path, z_bed, transform, X_mnt, Y_mnt, Z_mnt, assets: Path,
    sp_weights: str, lg_weights: str,
) -> dict | None:
    """Align the first extracted frame's Depth Pro output to the ortho/MNT."""
    banner("STAGE 6b: Point cloud alignment + water volume")

    from modules.align_pointclouds import run_alignment, visualise_alignment
    from modules.infer_features import load_feature_models

    # Use the first frame that has a saved inv_depth .npy
    frames_dir = out_dir / "frames"
    inv_depth_files = sorted(frames_dir.glob("*.npy")) if frames_dir.exists() else []
    if not inv_depth_files:
        # Check out_dir directly
        inv_depth_files = sorted(out_dir.glob("frame_*.npy"))

    if not inv_depth_files:
        print("  No inv_depth .npy found — skipping alignment")
        print("  Tip: depth_to_elevation saves .tif but not .npy; "
              "run infer_depth.py standalone to get .npy")
        return None

    # Find matching frame PNG
    inv_npy = inv_depth_files[0]
    frame_png = inv_npy.with_suffix(".png")
    if not frame_png.exists():
        frame_png = frames_dir / (inv_npy.stem + ".png")
    if not frame_png.exists():
        print(f"  Frame PNG not found for {inv_npy.name} — skipping alignment")
        return None

    import numpy as np
    inv_depth = np.load(inv_npy)
    print(f"  Using frame: {frame_png.name}  inv_depth: {inv_depth.shape}")

    sp_jit, sp_vars, lg_jit, lg_vars = load_feature_models(sp_weights, lg_weights)

    ortho_path = str(REPO / "data" / "brague" / "Ortho.tif")
    result = run_alignment(
        str(frame_png), inv_depth, ortho_path, z_bed, transform,
        X_mnt, Y_mnt, Z_mnt,
        sp_jit, sp_vars, lg_jit, lg_vars,
    )

    paths = visualise_alignment(result, str(frame_png), ortho_path, str(assets))
    print(f"  Water volume  : {result['volume_m3']:,.1f} m³")
    print(f"  Inundated area: {result['area_m2']:,.0f} m²")
    print(f"  Bank GCPs     : {len(result['gcp_xyz'])}")
    return result


# ── stage 7: canal optimiser ─────────────────────────────────────────────────

def run_canal_optimizer(meta: dict, out_dir_canal: Path) -> dict:
    """Run JAX canal optimiser using Q derived from pipeline meta."""
    banner("STAGE 7: JAX Canal Optimizer")

    from modules.canal_optimizer import optimise_canal

    h_mean = meta["h_final_mean"]
    # Estimate Q with alpha=0.9, V=1.0 m/s (from depth-mean velocity)
    # Q = alpha * V_surface * h * A_wet_m2
    # A_wet ~ wet_pixels * (2.4e-3)^2
    px_area = (2.4e-3) ** 2  # 2.4 mm pixel
    avg_wet = sum(f["wet_pixels"] for f in meta["frames"]) / len(meta["frames"])
    A_wet   = avg_wet * px_area
    alpha   = 0.9
    V_est   = 1.0   # conservative surface velocity estimate
    Q_target = alpha * V_est * h_mean * (A_wet ** 0.5) * 0.1  # rough Q
    Q_target = max(Q_target, 50.0)   # ensure meaningful design flood
    print(f"  Estimated Q_target ≈ {Q_target:.1f} m³/s")

    params = optimise_canal(Q_target=Q_target)
    out_dir_canal.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_canal / "canal_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"  Canal params → {out_path}")
    print(f"  B={params['bed_width']:.2f} m  D={params['water_depth']:.2f} m  "
          f"Q_calc={params['calculated_discharge']:.2f} m³/s")
    return params


# ── stage 6c: D8 thalweg extraction ─────────────────────────────────────────

def run_d8_thalweg(out_dir: Path, assets: Path) -> dict | None:
    banner("STAGE 6c: D8 thalweg extraction")

    flow_tif = out_dir / "flow_depth.tif"
    if not flow_tif.exists():
        print("  flow_depth.tif not found — skipping D8")
        return None

    z_surface_tifs = sorted(out_dir.glob("*_z_surface.tif"))
    if not z_surface_tifs:
        print("  No z_surface.tif found — skipping D8")
        return None

    from modules.d8_thalweg import extract_d8_thalweg
    result = extract_d8_thalweg(
        str(flow_tif),
        str(z_surface_tifs[0]),
        sub=10,
        depth_thresh=0.10,
    )

    geo = result["geometry"]
    print(f"  Thalweg length : {geo['length_m']:.1f} m")
    print(f"  Mean slope     : {geo['slope_mean']:.5f}")
    print(f"  ΔZ invert      : {geo.get('z_drop_m', 0.0):.3f} m")

    # Quick thalweg plan-view save
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        acc = result["acc_np"]
        ax.imshow(
            np.log1p(acc), origin="upper", cmap="Blues",
            alpha=0.8,
        )
        cx = result.get("centerline_col_sub", [])
        cy = result.get("centerline_row_sub", [])
        if len(cx) > 1:
            ax.plot(cx, cy, "r-", lw=1.5, label="D8 thalweg")
        ax.set_title(
            f"D8 Thalweg  L={geo['length_m']:.0f} m  "
            f"slope={geo['slope_mean']:.4f}",
            color="#e8e8e8", fontsize=10,
        )
        ax.legend(fontsize=8, facecolor="#1a1a2a", edgecolor="#3a3a4a",
                  labelcolor="#e8e8e8")
        ax.tick_params(colors="#808080")
        for sp in ax.spines.values():
            sp.set_color("#303040")

        assets.mkdir(parents=True, exist_ok=True)
        out = assets / "d8_thalweg.png"
        fig.savefig(out, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out}")
    except Exception as exc:
        print(f"  [WARN] thalweg plot failed: {exc}")

    return result


# ── stage 6d: pointcloud × ortho alignment check ─────────────────────────────

def run_pc_ortho_check(mnt_xyz: str, ortho_tif: str, assets: Path):
    banner("STAGE 6d: Pointcloud × Ortho alignment check")
    from modules.pointcloud_ortho_check import generate_pointcloud_ortho_check

    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "pointcloud_ortho_check.png"
    generate_pointcloud_ortho_check(
        mnt_xyz=mnt_xyz,
        ortho_tif=ortho_tif,
        out_path=str(out),
        stride=100,
    )


# ── stage 7b: canal 3D overlay ───────────────────────────────────────────────

def run_canal_3d_viz(
    canal_params: dict, out_dir: Path, ortho_tif: str,
    assets: Path, d8_result: dict | None,
):
    banner("STAGE 7b: Canal 3D overlay")

    flow_tif = out_dir / "flow_depth.tif"
    z_surface_tifs = sorted(out_dir.glob("*_z_surface.tif"))
    if not flow_tif.exists() or not z_surface_tifs:
        print("  Missing rasters — skipping 3D overlay")
        return

    from modules.extract_reach_geometry import extract_reach
    from modules.canal_3d_viz import generate_canal_3d_overlay

    reach_geo = extract_reach(str(flow_tif))
    print(f"  Centreline pts : {len(reach_geo['centerline_x'])}")

    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "canal_3d_overlay.png"
    generate_canal_3d_overlay(
        canal_params=canal_params,
        reach_geo=reach_geo,
        flow_depth_tif=str(flow_tif),
        z_surface_tif=str(z_surface_tifs[0]),
        ortho_tif=ortho_tif,
        d8_result=d8_result,
        out_path=str(out),
    )
    return reach_geo


# ── stage 7c: canal 4-view CAD drawing ───────────────────────────────────────

def run_canal_cad(canal_params: dict, reach_geo: dict | None, assets: Path):
    banner("STAGE 7c: Canal 4-view CAD drawing")
    from modules.canal_3d_viz import generate_canal_cad_figure

    if reach_geo is None:
        from modules.extract_reach_geometry import extract_reach
        flow_tifs = list((REPO / "output" / "brague").glob("flow_depth.tif"))
        if flow_tifs:
            reach_geo = extract_reach(str(flow_tifs[0]))
        else:
            reach_geo = {"centerline_x": [0, 100], "centerline_y": [0, 0],
                         "h_mean": 1.15}

    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "canal_cad_model.png"
    generate_canal_cad_figure(
        canal_params=canal_params,
        reach_geo=reach_geo,
        out_path=str(out),
    )
    print(f"  Saved → {out}")


# ── stage 7d: JAX LSPIV discharge ────────────────────────────────────────────

def run_lspiv(
    video_down: str, video_up: str,
    mnt_xyz: str, ortho_tif: str,
    out_dir: Path, assets: Path,
) -> dict | None:
    banner("STAGE 7d: JAX LSPIV — surface velocity + discharge")

    import cv2
    import numpy as np
    from modules.jax_lspiv import (
        run_jax_lspiv,
        GCP_IMAGE_1139, GCP_MODEL_1139,
        GCP_IMAGE_1142, GCP_MODEL_1142,
    )
    from modules.lspiv_viz import plot_lspiv_results

    def _extract_consec(video_path: str, start_frame: int,
                        n: int, dest_dir: Path) -> list[str]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            p = dest_dir / f"frame_{start_frame + i:05d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(str(p))
        cap.release()
        return paths

    results = {}

    # ── 1139 downstream bridge ─────────────────────────────────────────
    frames_1139_dir = out_dir / "frames_1139_consec"
    frames_1139 = sorted(frames_1139_dir.glob("*.png")) if frames_1139_dir.exists() else []
    if len(frames_1139) < 2:
        if Path(video_down).exists():
            print("  Extracting consecutive frames from IMG_1139.MOV …")
            frames_1139 = _extract_consec(video_down, 200, 6, frames_1139_dir)
        else:
            print(f"  [WARN] {video_down} not found — skipping 1139")
            frames_1139 = []

    if len(frames_1139) >= 2:
        print("  Running 1139 (downstream bridge) …")
        r1139 = run_jax_lspiv(
            frame_paths=[str(p) for p in frames_1139],
            image_points=GCP_IMAGE_1139,
            model_points=GCP_MODEL_1139,
            mnt_xyz_path=mnt_xyz,
        )
        results["1139"] = r1139
        print(f"  1139: {len(r1139['X']):,} vectors  "
              f"Q={r1139['Q']:.2f} m³/s")

    # ── 1142 upstream bridge ──────────────────────────────────────────
    frames_1142_dir = out_dir / "frames_1142_consec"
    frames_1142 = sorted(frames_1142_dir.glob("*.png")) if frames_1142_dir.exists() else []
    if len(frames_1142) < 2:
        if Path(video_up).exists():
            print("  Extracting consecutive frames from IMG_1142.MOV …")
            frames_1142 = _extract_consec(video_up, 0, 6, frames_1142_dir)
        else:
            print(f"  [WARN] {video_up} not found — skipping 1142")
            frames_1142 = []

    if len(frames_1142) >= 2:
        print("  Running 1142 (upstream bridge) …")
        r1142 = run_jax_lspiv(
            frame_paths=[str(p) for p in frames_1142],
            image_points=GCP_IMAGE_1142,
            model_points=GCP_MODEL_1142,
            mnt_xyz_path=mnt_xyz,
        )
        results["1142"] = r1142
        print(f"  1142: {len(r1142['X']):,} vectors  "
              f"Q={r1142['Q']:.2f} m³/s")

    if not results:
        print("  No LSPIV results — skipping viz")
        return None

    # ── merge and visualise ────────────────────────────────────────────
    if len(results) == 2:
        import numpy as np
        r1 = results["1139"]
        r2 = results["1142"]
        merged = dict(r1)
        for k in ("X", "Y", "U", "V", "norm"):
            merged[k] = np.concatenate([r1[k], r2[k]])
        merged["Q"] = r1["Q"] + r2["Q"]
        merged["ortho_imgs"] = r1["ortho_imgs"]
        print(f"\n  Combined: {len(merged['X']):,} vectors  "
              f"Q_total = {merged['Q']:.2f} m³/s")
        result_to_plot = merged
    else:
        result_to_plot = list(results.values())[0]

    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lspiv_results.png"
    plot_lspiv_results(
        result=result_to_plot,
        ortho_tif=ortho_tif,
        out_path=str(out),
    )
    print(f"  Saved → {out}")
    return result_to_plot


# ── stage 8: annotated visualisation ────────────────────────────────────────

def run_visualisation(assets_dir: Path):
    banner("STAGE 8: Annotated pipeline visualisation")
    from modules.annotated_pipeline_viz import load_all, build_figure

    d = load_all()
    fig = build_figure(d)

    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "annotated_pipeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="opyf_colab hydraulic pipeline — frames to flow depth to canal design")
    p.add_argument("--video",         default="data/brague/IMG_1139.MOV",
                   help="Downstream bridge video (IMG_1139)")
    p.add_argument("--video-up",      default="data/brague/IMG_1142.MOV",
                   help="Upstream bridge video (IMG_1142)")
    p.add_argument("--mnt",           default="data/brague/MNT.xyz")
    p.add_argument("--ortho",         default="data/brague/Ortho.tif")
    p.add_argument("--weights",       default="weights/depth_pro.msgpack")
    p.add_argument("--out-dir",       default="output/brague")
    p.add_argument("--assets",        default="assets")
    p.add_argument("--n-frames",      type=int, default=5)
    p.add_argument("--sp-weights",    default="weights/superpoint.msgpack")
    p.add_argument("--lg-weights",    default="weights/superpoint_lightglue.msgpack")
    p.add_argument("--skip-download",  action="store_true")
    p.add_argument("--skip-depth",     action="store_true")
    p.add_argument("--skip-align",     action="store_true",
                   help="Skip point cloud alignment + water volume")
    p.add_argument("--skip-d8",        action="store_true",
                   help="Skip D8 thalweg extraction")
    p.add_argument("--skip-pc-check",  action="store_true",
                   help="Skip pointcloud × ortho alignment check")
    p.add_argument("--skip-canal",     action="store_true")
    p.add_argument("--skip-canal-viz", action="store_true",
                   help="Skip canal 3D overlay and CAD drawing")
    p.add_argument("--skip-lspiv",     action="store_true",
                   help="Skip JAX LSPIV discharge computation")
    p.add_argument("--skip-viz",       action="store_true")
    return p.parse_args()


def main():
    os.chdir(REPO)
    args = parse_args()

    out_dir   = Path(args.out_dir)
    assets    = Path(args.assets)
    canal_dir = REPO / "canal_design"

    t_total = time.time()

    # ── stage 0: asset download ─────────────────────────────────────────────
    if not args.skip_download:
        download_assets()
    else:
        banner("STAGE 0: Download (SKIPPED)")

    # ── depth pipeline ──────────────────────────────────────────────────────
    meta_path = out_dir / "pipeline_meta.json"
    flow_tif  = out_dir / "flow_depth.tif"

    if args.skip_depth and flow_tif.exists() and meta_path.exists():
        banner("STAGE 1-5: Depth pipeline (SKIPPED — using cached outputs)")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Loaded meta: {meta_path}")
        # Still need z_bed + transform for alignment stage
        _z_bed = _transform = _X_mnt = _Y_mnt = _Z_mnt = None
    else:
        check_file(Path(args.video),   "Video")
        check_file(Path(args.mnt),     "MNT")
        check_file(Path(args.ortho),   "Ortho")
        check_file(Path(args.weights), "Weights")
        meta = run_depth_pipeline(args)
        _z_bed = _transform = _X_mnt = _Y_mnt = _Z_mnt = None   # loaded lazily below

    # ── stage 6b: point cloud alignment + water volume ──────────────────────
    if not args.skip_align:
        import numpy as np, rasterio as _rio
        from modules.depth_to_elevation import load_mnt, rasterise_mnt

        if _z_bed is None:
            with _rio.open(args.ortho) as src:
                _transform = src.transform
                _ortho_shape = (src.height, src.width)
            _X_mnt, _Y_mnt, _Z_mnt = load_mnt(args.mnt, subsample=5)
            _z_bed = rasterise_mnt(_X_mnt, _Y_mnt, _Z_mnt, _transform, _ortho_shape)

        run_alignment_stage(
            out_dir, _z_bed, _transform, _X_mnt, _Y_mnt, _Z_mnt, assets,
            args.sp_weights, args.lg_weights,
        )
    else:
        banner("STAGE 6b: Point cloud alignment (SKIPPED)")

    # ── stage 6c: D8 thalweg ─────────────────────────────────────────────────
    d8_result = None
    if not args.skip_d8:
        d8_result = run_d8_thalweg(out_dir, assets)
    else:
        banner("STAGE 6c: D8 thalweg (SKIPPED)")

    # ── stage 6d: pointcloud × ortho check ───────────────────────────────────
    if not args.skip_pc_check:
        if Path(args.mnt).exists() and Path(args.ortho).exists():
            run_pc_ortho_check(args.mnt, args.ortho, assets)
        else:
            print("  MNT or Ortho missing — skipping PC check")
    else:
        banner("STAGE 6d: Pointcloud × Ortho check (SKIPPED)")

    # ── canal optimiser ─────────────────────────────────────────────────────
    cp_path = canal_dir / "canal_params.json"

    if args.skip_canal and cp_path.exists():
        banner("STAGE 7: Canal optimizer (SKIPPED — using cached params)")
        with open(cp_path) as f:
            canal_params = json.load(f)
        print(f"  Loaded: {cp_path}")
    else:
        canal_params = run_canal_optimizer(meta, canal_dir)

    # ── stage 7b: canal 3D overlay  +  stage 7c: CAD drawing ─────────────────
    reach_geo = None
    if not args.skip_canal_viz:
        reach_geo = run_canal_3d_viz(
            canal_params, out_dir, args.ortho, assets, d8_result,
        )
        run_canal_cad(canal_params, reach_geo, assets)
    else:
        banner("STAGE 7b/7c: Canal 3D viz + CAD (SKIPPED)")

    # ── stage 7d: JAX LSPIV discharge ────────────────────────────────────────
    lspiv_result = None
    if not args.skip_lspiv:
        lspiv_result = run_lspiv(
            video_down=args.video,
            video_up=args.video_up,
            mnt_xyz=args.mnt,
            ortho_tif=args.ortho,
            out_dir=out_dir,
            assets=assets,
        )
    else:
        banner("STAGE 7d: JAX LSPIV (SKIPPED)")

    # ── visualisation ────────────────────────────────────────────────────────
    if not args.skip_viz:
        run_visualisation(assets)
    else:
        banner("STAGE 8: Visualisation (SKIPPED)")

    # ── summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    banner("DONE")
    print(f"  Elapsed           : {elapsed:.1f} s")
    print(f"  flow_depth.tif    : {flow_tif}")
    print(f"  alignment_matches : {assets / 'alignment_matches.png'}")
    print(f"  d8_thalweg        : {assets / 'd8_thalweg.png'}")
    print(f"  pc_ortho_check    : {assets / 'pointcloud_ortho_check.png'}")
    print(f"  canal_params      : {cp_path}")
    print(f"  canal_3d_overlay  : {assets / 'canal_3d_overlay.png'}")
    print(f"  canal_cad_model   : {assets / 'canal_cad_model.png'}")
    print(f"  lspiv_results     : {assets / 'lspiv_results.png'}")
    print(f"  pipeline viz      : {assets / 'annotated_pipeline.png'}")
    print()
    print("  Key results:")
    print(f"    h_mean (flow depth)    = {meta['h_final_mean']:.3f} m")
    print(f"    h_max  (flow depth)    = {meta['h_final_max']:.3f} m")
    print(f"    Canal bed width        = {canal_params['bed_width_m']:.2f} m")
    print(f"    Canal water depth      = {canal_params['water_depth_m']:.2f} m")
    print(f"    Discharge (calculated) = {canal_params['Q_calculated_m3s']:.2f} m³/s")
    if lspiv_result is not None:
        print(f"    LSPIV discharge Q      = {lspiv_result['Q']:.2f} m³/s")


if __name__ == "__main__":
    main()
