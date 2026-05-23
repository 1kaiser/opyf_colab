"""
pipeline.py — single entry-point for the opyf_colab hydraulic analysis pipeline.

Stages
------
1. Extract frames from event video
2. Load Depth Pro weights
3. Load ortho + dry mask
4. Load & rasterise MNT bed model
5. Per-frame depth inference + scale solve + flow depth
6. Aggregate frames (median) → flow_depth.tif
7. Run JAX canal optimizer
8. Build annotated visualisation → assets/annotated_pipeline.png

Usage
-----
    JAX_PLATFORMS=cpu python3 pipeline.py [options]

Options
-------
--video   PATH   Event-day video (default: tests/Test_Brague_flood/IMG_1139.MOV)
--mnt     PATH   MNT .xyz point cloud (default: data/brague/MNT.xyz)
--ortho   PATH   Orthorectified GeoTIFF (default: data/brague/Ortho.tif)
--weights PATH   Depth Pro .msgpack weights (default: weights/depth_pro.msgpack)
--out-dir PATH   Intermediate outputs directory (default: output/brague)
--assets  PATH   Final assets directory (default: assets)
--n-frames N     Number of frames to sample (default: 5)
--skip-depth     Skip depth inference if output/brague/flow_depth.tif already exists
--skip-canal     Skip canal optimisation if canal_design/canal_params.json exists
--skip-viz       Skip visualisation step
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent

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
    p.add_argument("--video",    default="tests/Test_Brague_flood/IMG_1139.MOV")
    p.add_argument("--mnt",      default="data/brague/MNT.xyz")
    p.add_argument("--ortho",    default="data/brague/Ortho.tif")
    p.add_argument("--weights",  default="weights/depth_pro.msgpack")
    p.add_argument("--out-dir",  default="output/brague")
    p.add_argument("--assets",   default="assets")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--skip-depth",  action="store_true",
                   help="Skip depth inference if flow_depth.tif already exists")
    p.add_argument("--skip-canal",  action="store_true",
                   help="Skip canal optimisation if canal_params.json already exists")
    p.add_argument("--skip-viz",    action="store_true",
                   help="Skip visualisation step")
    return p.parse_args()


def main():
    os.chdir(REPO)
    args = parse_args()

    out_dir   = Path(args.out_dir)
    assets    = Path(args.assets)
    canal_dir = REPO / "canal_design"

    t_total = time.time()

    # ── depth pipeline ──────────────────────────────────────────────────────
    meta_path = out_dir / "pipeline_meta.json"
    flow_tif  = out_dir / "flow_depth.tif"

    if args.skip_depth and flow_tif.exists() and meta_path.exists():
        banner("STAGE 1-6: Depth pipeline (SKIPPED — using cached outputs)")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Loaded meta: {meta_path}")
    else:
        check_file(Path(args.video),   "Video")
        check_file(Path(args.mnt),     "MNT")
        check_file(Path(args.ortho),   "Ortho")
        check_file(Path(args.weights), "Weights")
        meta = run_depth_pipeline(args)

    # ── canal optimiser ─────────────────────────────────────────────────────
    cp_path = canal_dir / "canal_params.json"

    if args.skip_canal and cp_path.exists():
        banner("STAGE 7: Canal optimizer (SKIPPED — using cached params)")
        with open(cp_path) as f:
            canal_params = json.load(f)
        print(f"  Loaded: {cp_path}")
    else:
        canal_params = run_canal_optimizer(meta, canal_dir)

    # ── visualisation ────────────────────────────────────────────────────────
    if not args.skip_viz:
        run_visualisation(assets)
    else:
        banner("STAGE 8: Visualisation (SKIPPED)")

    # ── summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    banner("DONE")
    print(f"  Elapsed       : {elapsed:.1f} s")
    print(f"  flow_depth.tif: {flow_tif}")
    print(f"  canal_params  : {cp_path}")
    print(f"  pipeline viz  : {assets / 'annotated_pipeline.png'}")
    print()
    print("  Key results:")
    print(f"    h_mean (flow depth)    = {meta['h_final_mean']:.3f} m")
    print(f"    h_max  (flow depth)    = {meta['h_final_max']:.3f} m")
    print(f"    Canal bed width        = {canal_params['bed_width']:.2f} m")
    print(f"    Canal water depth      = {canal_params['water_depth']:.2f} m")
    print(f"    Discharge (calculated) = {canal_params['calculated_discharge']:.2f} m³/s")


if __name__ == "__main__":
    main()
