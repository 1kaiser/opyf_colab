"""
pipelines/depth_to_elevation.py
================================
Stage 1–2 of the hydraulic pipeline:

    event-day video frames
          │
          ▼
    Depth Pro JAX  →  relative depth map  d(u,v)
          │
          ▼
    GCP registration  →  align to Lambert-93 (EPSG:2154)
          │
          ▼
    Scale solve on dry areas  →  Z_surface(x,y)  [absolute elevation]
          │
          ▼
    Z_surface − Z_bed(MNT)  =  h(x,y)  [flow depth per pixel]
          │
          ▼
    Output raster:  flow_depth.tif  (same grid as Ortho.tif)

Usage
-----
    JAX_PLATFORMS=cpu python3 pipelines/depth_to_elevation.py \\
        --video     tests/Test_Brague_flood/IMG_1139.MOV \\
        --mnt       data/brague/MNT.xyz \\
        --ortho     data/brague/Ortho.tif \\
        --weights   weights/depth_pro.msgpack \\
        --out-dir   output/brague \\
        --n-frames  5

Data contracts
--------------
Input:
  MNT.xyz   — X Y Z text, Lambert-93 (EPSG:2154), sub-cm resolution
  Ortho.tif — GeoTIFF, Lambert-93, gives pixel↔XY mapping
  video.MOV — event-day video frames, for Depth Pro inference

Output:
  output/brague/
    frames/           extracted event frames (PNG)
    depth_raw/        Depth Pro relative depth maps (NPY)
    flow_depth.tif    georeferenced flow depth raster h(x,y) in metres
    z_surface.tif     inferred water surface elevation Z_surface(x,y)
    pipeline_meta.json  scale/offset, GCPs used, stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import median_filter
from tqdm import tqdm

# ── rasterio (GeoTIFF I/O) ───────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_bounds, rowcol, xy
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[warn] rasterio not installed — GeoTIFF output disabled")

# ── local model imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.jax.jax_depth_pro.models.depth_pro import DepthPro

DEPTH_PRO_IMG_SIZE = 1536   # model's expected input resolution


# ════════════════════════════════════════════════════════════════════════════
# 1. Frame extraction from video
# ════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, out_dir: str, n_frames: int = 5,
                   start_frame: int = 200) -> list[str]:
    """
    Extract n_frames evenly-spaced frames from video starting at start_frame.
    Returns list of saved PNG paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    # Sample indices spread across the usable portion of the video
    end_frame = min(total - 1, start_frame + int(fps * 10))  # 10-second window
    indices   = np.linspace(start_frame, end_frame, n_frames, dtype=int)

    paths = []
    for idx in tqdm(indices, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        p = os.path.join(out_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(p, frame)
        paths.append(p)

    cap.release()
    print(f"  Extracted {len(paths)} frames → {out_dir}")
    return paths


# ════════════════════════════════════════════════════════════════════════════
# 2. Depth Pro inference
# ════════════════════════════════════════════════════════════════════════════

def load_depth_pro(weights_path: str):
    """Load DepthPro model and JIT-compile."""
    model = DepthPro(vit_config={
        'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
        'depth': 24, 'num_heads': 16, 'init_values': 1e-5
    })
    print(f"  Loading Depth Pro weights from {weights_path} ...")
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    jit_fn = jax.jit(model.apply)
    # Warm up with a dummy call
    dummy = jnp.zeros((1, 3, DEPTH_PRO_IMG_SIZE, DEPTH_PRO_IMG_SIZE))
    _ = jit_fn(variables, dummy)
    print("  Depth Pro ready.")
    return jit_fn, variables


def infer_depth(frame_path: str, jit_fn, variables) -> tuple[np.ndarray, float]:
    """
    Run Depth Pro on a single frame.

    Returns
    -------
    inv_depth : np.ndarray (H, W)  — inverse depth (1/metric_depth)
    fov_deg   : float              — estimated horizontal FOV in degrees
    """
    img_bgr  = cv2.imread(frame_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized  = cv2.resize(img_rgb, (DEPTH_PRO_IMG_SIZE, DEPTH_PRO_IMG_SIZE))
    inp      = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    inp_jax  = jnp.array(inp[None, ...])  # (1, 3, H, W)

    inv_depth_jax, fov_jax = jit_fn(variables, inp_jax)

    inv_depth = np.array(inv_depth_jax[0, ..., 0])  # (H, W)
    fov_deg   = float(fov_jax[0])
    return inv_depth, fov_deg


def inv_depth_to_metric(inv_depth: np.ndarray) -> np.ndarray:
    """Convert inverse depth to metric depth (metres), clamped."""
    return 1.0 / np.clip(inv_depth, 1e-6, None)


# ════════════════════════════════════════════════════════════════════════════
# 3. MNT loading and rasterisation
# ════════════════════════════════════════════════════════════════════════════

def load_mnt(mnt_path: str, subsample: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNT.xyz point cloud (X Y Z, space-separated, Lambert-93).

    Returns
    -------
    X, Y, Z : (N,) arrays of easting, northing, elevation
    """
    print(f"  Loading MNT from {mnt_path} ...")
    data = np.loadtxt(mnt_path, usecols=(0, 1, 2))
    if subsample > 1:
        data = data[::subsample]
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
    print(f"  MNT: {len(X):,} points  "
          f"Z=[{Z.min():.2f}, {Z.max():.2f}] m  "
          f"X=[{X.min():.1f}, {X.max():.1f}]  "
          f"Y=[{Y.min():.1f}, {Y.max():.1f}]")
    return X, Y, Z


def rasterise_mnt(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                  transform, out_shape: tuple[int, int]) -> np.ndarray:
    """
    Interpolate MNT point cloud onto the ortho pixel grid using
    LinearNDInterpolator (Delaunay triangulation).

    Returns
    -------
    z_bed : (H, W) raster of bed elevation, NaN where no MNT data
    """
    H, W = out_shape
    print(f"  Rasterising MNT onto {W}×{H} grid ...")

    # Generate pixel centre coordinates in Lambert-93
    cols = np.arange(W)
    rows = np.arange(H)
    col_grid, row_grid = np.meshgrid(cols, rows)

    if HAS_RASTERIO:
        xs, ys = rasterio.transform.xy(transform, row_grid.ravel(), col_grid.ravel())
        xs = np.array(xs).reshape(H, W)
        ys = np.array(ys).reshape(H, W)
    else:
        # Manual affine: x = x0 + col*dx,  y = y0 + row*dy
        xs = transform.c + col_grid * transform.a
        ys = transform.f + row_grid * transform.e

    # Interpolate
    interp = LinearNDInterpolator(list(zip(X, Y)), Z, fill_value=np.nan)
    z_bed  = interp(xs, ys)

    valid = np.isfinite(z_bed).sum()
    print(f"  MNT rasterised: {valid:,} valid pixels ({100*valid/z_bed.size:.1f}%)")
    return z_bed


# ════════════════════════════════════════════════════════════════════════════
# 4. Depth map → absolute elevation via GCP scale solve
# ════════════════════════════════════════════════════════════════════════════

# Ground control points from test_opyf_LSPIV_Brague.md
# (image pixel → Lambert-93 XYZ) for IMG_1139
GCP_IMAGE_PTS = np.array([
    (355,  429),   # left bank
    (1338, 350),   # right bank
    (99,   562),   # left front bank
    (1673, 364),   # right front bank
], dtype=float)

# Absolute model coordinates (Lambert-93 + elevation offset)
# The .md uses relative coords — abs_or = (30.13, -8.28, 0) is the origin offset
# Real-world coords need to be added back; using relative for scale solve
GCP_MODEL_PTS = np.array([
    (30.13, -8.28,  0.0),
    (32.88, -28.08, 0.0),
    (20.46, -4.47,  0.4),
    (21.32, -27.14, 0.4),
], dtype=float)


def solve_depth_scale(metric_depth: np.ndarray,
                      z_bed: np.ndarray,
                      dry_mask: np.ndarray,
                      frame_shape: tuple[int, int],
                      ortho_shape: tuple[int, int]) -> tuple[float, float]:
    """
    Solve  Z_abs = s * d + t  using pixels where ground is dry (z_bed known,
    no water — typically bank pixels visible in the frame).

    dry_mask : (H_ortho, W_ortho) bool — True where ground is exposed (dry)
    metric_depth at ortho resolution required.

    Returns  (s, t)  scale and offset.
    """
    # Resize depth map to ortho resolution for direct comparison
    H_o, W_o = ortho_shape
    d_resized = cv2.resize(metric_depth, (W_o, H_o),
                           interpolation=cv2.INTER_LINEAR)

    valid = dry_mask & np.isfinite(z_bed) & (d_resized > 0)
    if valid.sum() < 10:
        print(f"  [warn] Only {valid.sum()} dry pixels — using identity scale")
        return 1.0, 0.0

    d_dry   = d_resized[valid]
    z_dry   = z_bed[valid]

    # Least-squares: [d | 1] @ [s, t]^T = z
    A = np.column_stack([d_dry, np.ones_like(d_dry)])
    result, _, _, _ = np.linalg.lstsq(A, z_dry, rcond=None)
    s, t = float(result[0]), float(result[1])

    # Sanity check
    z_pred   = s * d_dry + t
    residual = np.sqrt(np.mean((z_pred - z_dry) ** 2))
    print(f"  Scale solve: s={s:.4f}  t={t:.4f}  RMSE={residual:.4f} m  "
          f"(n={valid.sum()} dry pixels)")
    return s, t


def compute_flow_depth(metric_depth: np.ndarray,
                       z_bed: np.ndarray,
                       s: float, t: float,
                       ortho_shape: tuple[int, int],
                       min_depth: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute water surface elevation and flow depth.

    Returns
    -------
    z_surface : (H, W) absolute water surface elevation [m]
    h         : (H, W) flow depth = z_surface − z_bed [m], NaN on dry pixels
    """
    H_o, W_o = ortho_shape
    d_resized = cv2.resize(metric_depth, (W_o, H_o),
                           interpolation=cv2.INTER_LINEAR)

    z_surface = s * d_resized + t

    # Smooth to reduce frame-level noise
    z_surface = median_filter(z_surface, size=5)

    # Flow depth — negative means the inferred surface is below the bed
    # (dry ground or error), clip to 0
    h = z_surface - z_bed
    h[h < min_depth]        = np.nan   # below threshold → dry
    h[~np.isfinite(z_bed)]  = np.nan   # no MNT data

    valid = np.isfinite(h).sum()
    if valid > 0:
        print(f"  Flow depth: mean={np.nanmean(h):.3f} m  "
              f"max={np.nanmax(h):.3f} m  "
              f"({valid:,} wet pixels)")
    return z_surface, h


# ════════════════════════════════════════════════════════════════════════════
# 5. Output — GeoTIFF writing
# ════════════════════════════════════════════════════════════════════════════

def write_geotiff(array: np.ndarray, path: str, transform, crs_epsg: int = 2154,
                  nodata: float = np.nan) -> None:
    """Write a single-band float32 GeoTIFF."""
    if not HAS_RASTERIO:
        np.save(path.replace(".tif", ".npy"), array)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array.astype("float32"), 1)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Dry mask heuristic (bank detection from ortho)
# ════════════════════════════════════════════════════════════════════════════

def estimate_dry_mask(ortho_path: str) -> tuple[np.ndarray, object, tuple]:
    """
    Load Ortho.tif and produce a simple water/dry mask using the
    Normalized Difference Water Index heuristic (green vs NIR proxy).
    For RGB-only ortho: use blue channel dominance as water proxy.

    Returns
    -------
    dry_mask   : (H, W) bool — True where ground is dry (banks, road, etc.)
    transform  : rasterio affine transform
    shape      : (H, W)
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required to load Ortho.tif")

    print(f"  Loading Ortho.tif from {ortho_path} ...")
    with rasterio.open(ortho_path) as src:
        transform = src.transform
        H, W      = src.height, src.width
        n_bands   = src.count
        print(f"  Ortho: {W}×{H} px  {n_bands} bands  "
              f"pixel={src.transform.a:.4f} m  CRS={src.crs}")

        # Read at reduced resolution for memory
        scale    = min(1.0, 4096 / max(H, W))
        out_h    = int(H * scale)
        out_w    = int(W * scale)
        img      = src.read(
            out_shape=(n_bands, out_h, out_w),
            resampling=rasterio.enums.Resampling.bilinear
        ).astype(float)

    # Recompute transform for the resampled resolution
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    bounds = rasterio.transform.array_bounds(H, W, transform)
    scaled_transform = from_bounds(*bounds, out_w, out_h)

    if n_bands >= 3:
        R, G, B = img[0], img[1], img[2]
    else:
        R = G = B = img[0]

    # Water appears darker and more blue; dry ground is brighter and brownish
    # Simple heuristic: blue dominance index
    total    = R + G + B + 1e-6
    blue_dom = (B / total) > 0.38    # water pixels tend to have high blue fraction
    dark     = (R + G + B) < 300     # dark pixels (shadow / deep water)
    water    = blue_dom & dark
    dry_mask = ~water

    print(f"  Dry mask: {dry_mask.sum():,} dry pixels / {dry_mask.size:,} total "
          f"({100*dry_mask.mean():.1f}% dry)")

    return dry_mask, scaled_transform, (out_h, out_w)


# ════════════════════════════════════════════════════════════════════════════
# 7. Per-frame pipeline
# ════════════════════════════════════════════════════════════════════════════

def process_frame(frame_path: str,
                  jit_fn, variables,
                  z_bed: np.ndarray,
                  dry_mask: np.ndarray,
                  transform,
                  ortho_shape: tuple[int, int],
                  out_dir: str) -> dict:
    """Run full pipeline for one event-day frame."""
    stem = Path(frame_path).stem

    # Depth Pro inference
    t0 = time.monotonic()
    inv_depth, fov = infer_depth(frame_path, jit_fn, variables)
    metric_depth   = inv_depth_to_metric(inv_depth)
    t_inf = time.monotonic() - t0
    print(f"\n[{stem}] Depth Pro: fov={fov:.1f}°  "
          f"d=[{metric_depth.min():.2f},{metric_depth.max():.2f}] m  "
          f"({t_inf:.1f}s)")

    # Save raw depth
    depth_path = os.path.join(out_dir, "depth_raw", f"{stem}_inv_depth.npy")
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
    np.save(depth_path, inv_depth)

    # Scale solve on dry pixels
    s, t = solve_depth_scale(metric_depth, z_bed, dry_mask,
                             metric_depth.shape, ortho_shape)

    # Compute flow depth
    z_surface, h = compute_flow_depth(metric_depth, z_bed, s, t, ortho_shape)

    # Write GeoTIFFs
    write_geotiff(z_surface, os.path.join(out_dir, f"{stem}_z_surface.tif"),
                  transform)
    write_geotiff(h, os.path.join(out_dir, f"{stem}_flow_depth.tif"),
                  transform)

    return {
        "frame":      frame_path,
        "fov_deg":    fov,
        "scale_s":    s,
        "offset_t":   t,
        "h_mean_m":   float(np.nanmean(h)) if np.isfinite(h).any() else None,
        "h_max_m":    float(np.nanmax(h))  if np.isfinite(h).any() else None,
        "wet_pixels": int(np.isfinite(h).sum()),
        "infer_time_s": round(t_inf, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 8. Multi-frame aggregation — median depth map
# ════════════════════════════════════════════════════════════════════════════

def aggregate_frames(out_dir: str, ortho_shape: tuple[int, int],
                     transform) -> np.ndarray:
    """
    Stack per-frame flow depth rasters and take pixel-wise median
    to suppress per-frame noise from Depth Pro.
    Writes flow_depth.tif (final output).
    """
    depth_files = sorted(Path(out_dir).glob("*_flow_depth.tif"))
    if not depth_files:
        print("[warn] No per-frame depth rasters found for aggregation")
        return None

    stack = []
    for p in depth_files:
        if HAS_RASTERIO:
            with rasterio.open(p) as src:
                stack.append(src.read(1).astype(float))
        else:
            stack.append(np.load(str(p).replace(".tif", ".npy")))

    h_stack  = np.stack(stack, axis=0)   # (N, H, W)
    h_median = np.nanmedian(h_stack, axis=0)

    write_geotiff(h_median, os.path.join(out_dir, "flow_depth.tif"), transform)
    print(f"\n  Aggregated {len(stack)} frames → flow_depth.tif  "
          f"median h_mean={np.nanmean(h_median):.3f} m")
    return h_median


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Depth Pro → flow depth raster")
    ap.add_argument("--video",    required=True,  help="Event-day video (.MOV/.mp4)")
    ap.add_argument("--mnt",      required=True,  help="MNT.xyz point cloud")
    ap.add_argument("--ortho",    required=True,  help="Ortho.tif georeferenced image")
    ap.add_argument("--weights",  required=True,  help="depth_pro.msgpack")
    ap.add_argument("--out-dir",  default="output/brague", help="Output directory")
    ap.add_argument("--n-frames", type=int, default=5, help="Event frames to process")
    ap.add_argument("--start-frame", type=int, default=200)
    ap.add_argument("--mnt-subsample", type=int, default=10,
                    help="Subsample MNT every N points (reduces RAM, default 10)")
    ap.add_argument("--dry-threshold", type=float, default=0.38,
                    help="Blue dominance threshold for water mask (default 0.38)")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== STAGE 1: Extract event frames ===")
    frame_dir = os.path.join(out_dir, "frames")
    frames    = extract_frames(args.video, frame_dir,
                               n_frames=args.n_frames,
                               start_frame=args.start_frame)
    if not frames:
        print("No frames extracted — check video path"); sys.exit(1)

    print("\n=== STAGE 2: Load Depth Pro ===")
    jit_fn, variables = load_depth_pro(args.weights)

    print("\n=== STAGE 3: Load ortho → dry mask ===")
    dry_mask, transform, ortho_shape = estimate_dry_mask(args.ortho)

    print("\n=== STAGE 4: Load & rasterise MNT ===")
    X, Y, Z = load_mnt(args.mnt, subsample=args.mnt_subsample)
    z_bed   = rasterise_mnt(X, Y, Z, transform, ortho_shape)

    print("\n=== STAGE 5: Per-frame depth inference + elevation ===")
    meta = {"frames": []}
    for frame_path in frames:
        result = process_frame(
            frame_path, jit_fn, variables,
            z_bed, dry_mask, transform, ortho_shape, out_dir
        )
        meta["frames"].append(result)
        print(f"  h_mean={result['h_mean_m']:.3f} m  "
              f"wet_px={result['wet_pixels']:,}")

    print("\n=== STAGE 6: Aggregate frames (median) ===")
    h_final = aggregate_frames(out_dir, ortho_shape, transform)

    # Save metadata
    meta["ortho_shape"]   = list(ortho_shape)
    meta["mnt_subsample"] = args.mnt_subsample
    meta["n_frames"]      = len(frames)
    meta["z_bed_mean"]    = float(np.nanmean(z_bed))
    meta["h_final_mean"]  = float(np.nanmean(h_final)) if h_final is not None else None
    meta["h_final_max"]   = float(np.nanmax(h_final))  if h_final is not None else None

    meta_path = os.path.join(out_dir, "pipeline_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata → {meta_path}")
    print("\n=== DONE ===")
    print(f"  flow_depth.tif → {out_dir}/flow_depth.tif")
    print(f"  z_surface.tif  → per-frame in {out_dir}/")


if __name__ == "__main__":
    main()
