"""
modules/align_pointclouds.py
=============================
Align Depth Pro event-frame point cloud to the pre-event ortho/MNT using
SuperPoint + LightGlue bank keypoint matching, then compute water volume
as the gap between the two surfaces.

Pipeline
--------
event frame (PNG)     ortho image (Ortho.tif)
        │                      │
        └──── LightGlue ────────┘
               matched bank keypoints  (dry land features present in both)
                        │
                        ▼
            GCP set: (u,v)frame ↔ (X,Y)Lambert93
                        │
               ┌────────┴─────────────────────────────────────────┐
               │ homography H: frame pixels → ortho pixels         │
               │ MNT lookup: Z_bed at each GCP                     │
               │ scale solve:  s,t  from  Z_bed = s·d_DepthPro + t│
               └──────────────────────────────────────────────────┘
                        │
                        ▼
            Warp event depth → ortho grid
            Z_surface(X,Y) = s · d_warped(X,Y) + t
                        │
            Z_surface − Z_bed = h(X,Y)   [flow depth]
            Volume = ∫ h dA               [m³]
                        │
                        ▼
            4-view visualisation  (top / front / side / isometric)
            + keypoint match figure
            → assets/alignment_*.png

Public API
----------
    run_alignment(event_frame_path, inv_depth, ortho_path, z_bed, transform,
                  sp_jit, sp_vars, lg_jit, lg_vars) -> dict

    visualise_alignment(result, event_frame_path, ortho_path, out_dir)

CLI
---
    python3 modules/align_pointclouds.py \\
        --frame   output/brague/frames/frame_00200.png \\
        --depth   output/brague/frames/frame_00200_inv_depth.npy \\
        --ortho   data/brague/Ortho.tif \\
        --mnt     data/brague/MNT.xyz \\
        --sp      weights/superpoint.msgpack \\
        --lg      weights/superpoint_lightglue.msgpack \\
        --out-dir assets
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import median_filter
import rasterio
from rasterio.transform import rowcol as rc_transform

# ── internal imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.infer_features import (
    load_feature_models, match_images, estimate_homography,
)

# ── constants ─────────────────────────────────────────────────────────────────
_DISPLAY_PTS = 80_000     # subsample for 3D scatter
_MIN_MATCHES  = 12        # minimum GCPs required


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GCP-based scale solve
# ═══════════════════════════════════════════════════════════════════════════════

def _ortho_px_to_world(col, row, transform):
    """Convert ortho pixel coords → Lambert-93 (X, Y)."""
    X = transform.c + col * transform.a
    Y = transform.f + row * transform.e
    return X, Y


def _sample_inv_depth(inv_depth, u, v):
    """Bilinear sample inv_depth at floating-point (u, v) pixel coords."""
    H, W = inv_depth.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    fu, fv  = u - u0, v - v0
    return (inv_depth[v0, u0] * (1 - fu) * (1 - fv) +
            inv_depth[v0, u1] *      fu  * (1 - fv) +
            inv_depth[v1, u0] * (1 - fu) *      fv  +
            inv_depth[v1, u1] *      fu  *      fv)


def _interpolate_mnt(X_query, Y_query, X_mnt, Y_mnt, Z_mnt):
    """Interpolate MNT point cloud to query (X, Y) positions via Delaunay."""
    interp = LinearNDInterpolator(np.column_stack([X_mnt, Y_mnt]), Z_mnt)
    return interp(X_query, Y_query)


def gcp_scale_solve(
    mkpts_frame: np.ndarray,   # (K, 2)  pixel (u,v) in event frame
    mkpts_ortho: np.ndarray,   # (K, 2)  pixel (col,row) in ortho image
    inv_depth:   np.ndarray,   # (Hf, Wf) Depth Pro output
    transform,                  # rasterio affine — ortho → Lambert-93
    X_mnt, Y_mnt, Z_mnt,       # MNT point cloud arrays
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve  Z_bed = s · inv_depth + t  using GCP bank keypoints.

    Returns (s, t, X_gcp, Y_gcp, Z_gcp) where (X,Y,Z) are the bank GCPs
    in Lambert-93 used for the solve.
    """
    # Ortho pixel → world
    col_o = mkpts_ortho[:, 0].astype(float)
    row_o = mkpts_ortho[:, 1].astype(float)
    X_gcp, Y_gcp = _ortho_px_to_world(col_o, row_o, transform)

    # MNT elevation at GCP world positions
    Z_gcp = _interpolate_mnt(X_gcp, Y_gcp, X_mnt, Y_mnt, Z_mnt)

    # Depth Pro inv_depth at GCP frame pixels
    u_f = mkpts_frame[:, 0].astype(float)
    v_f = mkpts_frame[:, 1].astype(float)
    d_gcp = _sample_inv_depth(inv_depth, u_f, v_f)

    # Keep only GCPs with valid MNT and nonzero depth
    valid = np.isfinite(Z_gcp) & (d_gcp > 0)
    if valid.sum() < 4:
        print(f"  [warn] Only {valid.sum()} valid GCPs — using fallback scale=1")
        return 1.0, 0.0, X_gcp, Y_gcp, Z_gcp

    d_v, z_v = d_gcp[valid], Z_gcp[valid]
    A = np.column_stack([d_v, np.ones_like(d_v)])
    res, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    s, t = float(res[0]), float(res[1])

    rmse = float(np.sqrt(np.mean((s * d_v + t - z_v) ** 2)))
    print(f"  GCP scale solve: s={s:.4f}  t={t:.4f}  RMSE={rmse:.3f} m  "
          f"({valid.sum()}/{len(valid)} GCPs used)")
    return s, t, X_gcp[valid], Y_gcp[valid], Z_gcp[valid]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Event depth → ortho grid
# ═══════════════════════════════════════════════════════════════════════════════

def warp_depth_to_ortho(
    inv_depth: np.ndarray,     # (Hf, Wf)
    H_frame_to_ortho: np.ndarray,   # 3×3 homography
    s: float, t: float,
    ortho_shape: tuple[int, int],   # (H_o, W_o)
    smooth_px: int = 7,
) -> np.ndarray:
    """
    Warp Depth Pro inv_depth from frame space to ortho pixel space via H,
    apply scale (s, t), and return Z_surface raster (H_o, W_o).

    Uses cv2.warpPerspective with inverse homography so we sample from the
    depth map at the right locations.
    """
    Hf, Wf = inv_depth.shape
    H_o, W_o = ortho_shape

    # Resize depth map to full frame resolution if needed (it comes out 1536²)
    # then remap to ortho grid
    H_inv = np.linalg.inv(H_frame_to_ortho)   # ortho → frame

    # Build ortho pixel grid
    col_o, row_o = np.meshgrid(np.arange(W_o), np.arange(H_o))
    ones = np.ones_like(col_o)
    ortho_pts = np.stack([col_o, row_o, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    # Map ortho pixels → frame pixels
    frame_pts = H_inv @ ortho_pts
    frame_pts /= frame_pts[2:3]  # homogeneous divide
    u_f = frame_pts[0].reshape(H_o, W_o)
    v_f = frame_pts[1].reshape(H_o, W_o)

    # Sample inv_depth at (u_f, v_f) — bilinear
    u_f_c = np.clip(u_f, 0, Wf - 1)
    v_f_c = np.clip(v_f, 0, Hf - 1)
    in_bounds = (u_f >= 0) & (u_f < Wf) & (v_f >= 0) & (v_f < Hf)

    d_warped = _sample_inv_depth(inv_depth, u_f_c.ravel(), v_f_c.ravel())
    d_warped = d_warped.reshape(H_o, W_o)
    d_warped[~in_bounds] = np.nan

    # Apply scale
    Z_surface = s * d_warped + t
    Z_surface[~in_bounds] = np.nan

    # Spatial smoothing
    if smooth_px > 0:
        mask_nan  = ~np.isfinite(Z_surface)
        Z_surface = np.where(mask_nan, 0.0, Z_surface)
        Z_surface = median_filter(Z_surface, size=smooth_px)
        Z_surface[mask_nan] = np.nan

    return Z_surface


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Point clouds + water volume
# ═══════════════════════════════════════════════════════════════════════════════

def build_point_clouds(
    Z_surface: np.ndarray,   # (H_o, W_o)  event surface
    z_bed: np.ndarray,       # (H_o, W_o)  pre-event bed (MNT)
    transform,
    min_depth: float = 0.05,
    subsample: int = 1,
) -> dict:
    """
    Build event surface + bed point clouds and compute water stats.

    Returns dict with:
        event_xyz    (N, 3)  event surface point cloud
        bed_xyz      (N, 3)  MNT bed point cloud (same footprint)
        water_xyz    (M, 3)  water column centroids (event pixels where h > 0)
        h            (H, W)  flow depth raster (NaN = dry)
        volume_m3    float   total water volume
        area_m2      float   inundated area
        pixel_area   float   m²/pixel
    """
    H_o, W_o = Z_surface.shape
    pix_m = abs(transform.a)          # pixel size in metres (e.g. 2.4e-3)
    pixel_area = pix_m * pix_m

    # Build XY grids in Lambert-93
    col_g, row_g = np.meshgrid(np.arange(W_o), np.arange(H_o))
    X, Y = _ortho_px_to_world(col_g, row_g, transform)

    # Flow depth
    h = Z_surface - z_bed
    h[(h < min_depth) | ~np.isfinite(h)] = np.nan
    h[~np.isfinite(z_bed)] = np.nan

    # Event surface cloud (only where both surfaces known)
    valid_evt = np.isfinite(Z_surface) & np.isfinite(z_bed)
    if subsample > 1:
        valid_evt[::subsample, :] = False
        valid_evt[:, ::subsample] = False

    event_xyz = np.column_stack([
        X[valid_evt], Y[valid_evt], Z_surface[valid_evt],
    ])
    bed_xyz = np.column_stack([
        X[valid_evt], Y[valid_evt], z_bed[valid_evt],
    ])

    # Water column cloud (where h > 0)
    wet = np.isfinite(h)
    water_xyz = np.column_stack([
        X[wet], Y[wet], (Z_surface[wet] + z_bed[wet]) / 2.0,   # midpoint
    ])

    vol_m3  = float(np.nansum(h) * pixel_area)
    area_m2 = float(wet.sum() * pixel_area)

    print(f"  Water volume : {vol_m3:,.1f} m³")
    print(f"  Inundated area: {area_m2:,.0f} m²  ({area_m2/1e4:.2f} ha)")
    print(f"  Mean depth   : {np.nanmean(h):.3f} m   Max: {np.nanmax(h):.3f} m")

    return dict(
        event_xyz=event_xyz, bed_xyz=bed_xyz, water_xyz=water_xyz,
        h=h, volume_m3=vol_m3, area_m2=area_m2, pixel_area=pixel_area,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main alignment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment(
    event_frame_path: str,
    inv_depth: np.ndarray,          # (H, W) from Depth Pro
    ortho_path: str,
    z_bed: np.ndarray,              # (H_o, W_o) MNT rasterised to ortho grid
    transform,                       # rasterio affine transform of ortho
    X_mnt: np.ndarray, Y_mnt: np.ndarray, Z_mnt: np.ndarray,
    sp_jit, sp_vars,
    lg_jit, lg_vars,
    top_k: int = 2048,
    min_score: float = 0.15,
) -> dict:
    """
    Full alignment pipeline: match → GCP scale solve → warp → point clouds.

    Returns dict with keys:
        mkpts_frame    matched keypoints in event frame (u,v)
        mkpts_ortho    matched keypoints in ortho image (col,row)
        H              3×3 homography (frame → ortho)
        s, t           depth scale and offset
        gcp_xyz        (K, 3) GCP positions in Lambert-93
        Z_surface      (H_o, W_o) event surface elevation raster
        event_xyz      (N, 3) event point cloud
        bed_xyz        (N, 3) bed point cloud
        water_xyz      (M, 3) water column centroids
        h              (H_o, W_o) flow depth
        volume_m3      float
        area_m2        float
    """
    # ── 1. Feature matching ─────────────────────────────────────────────────
    print(f"  Matching features: {Path(event_frame_path).name} ↔ ortho …")
    mkpts_f, mkpts_o = match_images(
        event_frame_path, ortho_path,
        sp_jit, sp_vars, lg_jit, lg_vars,
        top_k=top_k, min_score=min_score,
    )
    print(f"  {len(mkpts_f)} matches found")

    if len(mkpts_f) < _MIN_MATCHES:
        raise RuntimeError(
            f"Too few matches ({len(mkpts_f)} < {_MIN_MATCHES}). "
            "Check image overlap or lower --min-score."
        )

    # ── 2. Homography frame → ortho ─────────────────────────────────────────
    H_mat = estimate_homography(mkpts_f, mkpts_o)
    if H_mat is None:
        raise RuntimeError("Homography estimation failed (RANSAC found no inliers).")

    # ── 3. GCP-based scale solve ─────────────────────────────────────────────
    s, t, X_gcp, Y_gcp, Z_gcp = gcp_scale_solve(
        mkpts_f, mkpts_o, inv_depth, transform, X_mnt, Y_mnt, Z_mnt,
    )

    # ── 4. Warp depth to ortho grid ──────────────────────────────────────────
    ortho_shape = z_bed.shape
    print(f"  Warping event depth to ortho grid {ortho_shape} …")
    Z_surface = warp_depth_to_ortho(inv_depth, H_mat, s, t, ortho_shape)

    # ── 5. Point clouds + water volume ───────────────────────────────────────
    print("  Building point clouds …")
    clouds = build_point_clouds(Z_surface, z_bed, transform)

    return dict(
        mkpts_frame=mkpts_f,
        mkpts_ortho=mkpts_o,
        H=H_mat,
        s=s, t=t,
        gcp_xyz=np.column_stack([X_gcp, Y_gcp, Z_gcp]),
        Z_surface=Z_surface,
        **clouds,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _subsample(xyz, n=_DISPLAY_PTS):
    if len(xyz) <= n:
        return xyz
    idx = np.random.choice(len(xyz), n, replace=False)
    return xyz[idx]


def _add_3d_view(ax, bed, event, water, gcp, elev, azim, title):
    """Draw a single 3D view panel."""
    ax.view_init(elev=elev, azim=azim)

    # Normalise Z for consistent colour across views
    z_min = min(bed[:, 2].min(), event[:, 2].min())
    z_max = event[:, 2].max()

    ax.scatter(bed[:, 0],   bed[:, 1],   bed[:, 2],
               c=(bed[:, 2] - z_min) / (z_max - z_min + 1e-6),
               cmap="terrain", s=0.3, alpha=0.4, rasterized=True)

    ax.scatter(event[:, 0], event[:, 1], event[:, 2],
               c=(event[:, 2] - z_min) / (z_max - z_min + 1e-6),
               cmap="RdYlBu_r", s=0.3, alpha=0.35, rasterized=True)

    if len(water):
        ax.scatter(water[:, 0], water[:, 1], water[:, 2],
                   c="cyan", s=0.4, alpha=0.5, rasterized=True)

    # GCPs — bank keypoints used for alignment
    ax.scatter(gcp[:, 0], gcp[:, 1], gcp[:, 2],
               c="yellow", s=20, marker="*", zorder=5, label="GCP (bank kpts)")

    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#888", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333")


def visualise_alignment(
    result: dict,
    event_frame_path: str,
    ortho_path: str,
    out_dir: str,
):
    """
    Produce two output figures:

    1. assets/alignment_matches.png
       Left: event frame with matched keypoints coloured by confidence rank
       Right: ortho image with corresponding keypoints

    2. assets/alignment_pointclouds.png
       2×2 grid: top / front / side / isometric 3D views of bed + event clouds
       with water volume as cyan points and GCPs as yellow stars

    Returns list of saved paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    mkpts_f  = result["mkpts_frame"]
    mkpts_o  = result["mkpts_ortho"]
    gcp_xyz  = result["gcp_xyz"]
    event_pc = _subsample(result["event_xyz"])
    bed_pc   = _subsample(result["bed_xyz"])
    water_pc = _subsample(result["water_xyz"])
    h        = result["h"]
    vol      = result["volume_m3"]
    area     = result["area_m2"]
    n_match  = len(mkpts_f)

    # ── Figure 1: keypoint matches ──────────────────────────────────────────
    frame_bgr = cv2.imread(event_frame_path)
    ortho_bgr = cv2.imread(ortho_path)

    # Resize ortho for display (can be huge)
    oh, ow = ortho_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    scale  = min(1.0, 1200 / max(ow, oh, fw, fh))
    if scale < 1:
        frame_disp = cv2.resize(frame_bgr, (int(fw*scale), int(fh*scale)))
        ortho_disp = cv2.resize(ortho_bgr, (int(ow*scale), int(oh*scale)))
        kf = mkpts_f * scale
        ko = mkpts_o * scale
    else:
        frame_disp, ortho_disp = frame_bgr, ortho_bgr
        kf, ko = mkpts_f, mkpts_o

    dh = max(frame_disp.shape[0], ortho_disp.shape[0])
    canvas = np.zeros((dh, frame_disp.shape[1] + ortho_disp.shape[1], 3), np.uint8)
    canvas[:frame_disp.shape[0], :frame_disp.shape[1]] = frame_disp
    canvas[:ortho_disp.shape[0], frame_disp.shape[1]:] = ortho_disp
    off_x = frame_disp.shape[1]

    colours = plt.cm.plasma(np.linspace(0, 1, len(kf)))
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(kf, ko)):
        c = tuple(int(v * 255) for v in colours[i][:3][::-1])
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + off_x, int(y2)), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), 4, c, -1)
        cv2.circle(canvas, (int(x2) + off_x, int(y2)), 4, c, -1)

    cv2.putText(canvas, f"{n_match} bank keypoint matches", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Event frame", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1)
    cv2.putText(canvas, "Ortho (dry, pre-event)", (off_x + 10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)

    fig_match, ax_m = plt.subplots(figsize=(16, 7), facecolor="#0d1117")
    ax_m.imshow(canvas[:, :, ::-1])
    ax_m.axis("off")
    ax_m.set_title(
        f"SuperPoint + LightGlue:  {n_match} bank GCPs  │  "
        f"scale s={result['s']:.4f}  t={result['t']:.3f} m  │  "
        f"Volume={vol:,.0f} m³",
        color="white", fontsize=10, pad=8,
    )
    match_path = str(Path(out_dir) / "alignment_matches.png")
    fig_match.savefig(match_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig_match)
    print(f"  Saved → {match_path}")

    # ── Figure 2: 4-view 3D point clouds ───────────────────────────────────
    VIEWS = [
        ("Top",       90,   0),
        ("Front",      5,   0),
        ("Side",       5,  90),
        ("Isometric", 30,  45),
    ]

    fig3d = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig3d.suptitle(
        f"Event frame vs Pre-event bed point clouds  │  "
        f"Water volume: {vol:,.0f} m³   Inundated area: {area:,.0f} m²   "
        f"Mean depth: {np.nanmean(h):.2f} m",
        color="white", fontsize=12, y=0.99,
    )

    for idx, (label, elev, azim) in enumerate(VIEWS):
        ax = fig3d.add_subplot(2, 2, idx + 1, projection="3d")
        _add_3d_view(ax, bed_pc, event_pc, water_pc, gcp_xyz,
                     elev=elev, azim=azim, title=label)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5a7d5a",
               markersize=6, label="Bed (MNT, pre-event)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff6b6b",
               markersize=6, label="Event surface (Depth Pro)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan",
               markersize=6, label="Water column"),
        Line2D([0], [0], marker="*", color="yellow",
               markersize=8, label="Bank GCPs (LightGlue)"),
    ]
    fig3d.legend(handles=legend_elements, loc="lower center",
                 ncol=4, fontsize=9, framealpha=0.2,
                 labelcolor="white", facecolor="#1a1a2e")

    # Inset: depth map colour bar
    ax_depth = fig3d.add_axes([0.92, 0.1, 0.015, 0.8])
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu_r",
        norm=plt.Normalize(vmin=np.nanmin(result["Z_surface"]),
                           vmax=np.nanmax(result["Z_surface"])),
    )
    sm.set_array([])
    cbar = fig3d.colorbar(sm, cax=ax_depth)
    cbar.set_label("Z surface (m)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    cloud_path = str(Path(out_dir) / "alignment_pointclouds.png")
    fig3d.savefig(cloud_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig3d)
    print(f"  Saved → {cloud_path}")

    # ── Figure 3: flow depth overlay on ortho ───────────────────────────────
    ortho_small = cv2.resize(ortho_bgr, (h.shape[1], h.shape[0]))
    h_norm = np.nan_to_num(h, nan=0.0)
    h_clip = np.clip(h_norm, 0, np.nanpercentile(h[np.isfinite(h)], 99))
    h_uint = (h_clip / (h_clip.max() + 1e-8) * 255).astype(np.uint8)
    depth_cm  = cv2.applyColorMap(h_uint, cv2.COLORMAP_JET)
    water_mask = (h_norm > 0).astype(np.uint8)
    overlay = ortho_small.copy()
    overlay[water_mask > 0] = (
        overlay[water_mask > 0] * 0.4 + depth_cm[water_mask > 0] * 0.6
    ).astype(np.uint8)

    fig_ov, ax_ov = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
    ax_ov.imshow(overlay[:, :, ::-1])
    ax_ov.scatter(mkpts_o[:, 0] * (h.shape[1] / ow),
                  mkpts_o[:, 1] * (h.shape[0] / oh),
                  c="yellow", s=25, marker="*", zorder=5,
                  label="Bank GCPs")
    ax_ov.set_title(
        f"Flow depth overlay on pre-event ortho  │  "
        f"Volume = {vol:,.0f} m³   Area = {area:,.0f} m²",
        color="white", fontsize=10, pad=6,
    )
    ax_ov.legend(fontsize=8, loc="upper right", framealpha=0.4)
    ax_ov.axis("off")

    sm2 = plt.cm.ScalarMappable(cmap="jet",
                                 norm=plt.Normalize(0, np.nanpercentile(h[np.isfinite(h)], 99)))
    sm2.set_array([])
    cb2 = fig_ov.colorbar(sm2, ax=ax_ov, fraction=0.03, pad=0.01)
    cb2.set_label("Flow depth h (m)", color="white", fontsize=9)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    overlay_path = str(Path(out_dir) / "alignment_depth_overlay.png")
    fig_ov.savefig(overlay_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig_ov)
    print(f"  Saved → {overlay_path}")

    return [match_path, cloud_path, overlay_path]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import rasterio

    p = argparse.ArgumentParser(description="Event-frame → ortho alignment + water volume")
    p.add_argument("--frame",   required=True, help="Event video frame PNG")
    p.add_argument("--depth",   required=True, help="inv_depth .npy from Depth Pro")
    p.add_argument("--ortho",   required=True, help="Ortho.tif (pre-event)")
    p.add_argument("--mnt",     required=True, help="MNT.xyz point cloud")
    p.add_argument("--sp",      default="weights/superpoint.msgpack")
    p.add_argument("--lg",      default="weights/superpoint_lightglue.msgpack")
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--top-k",   type=int, default=2048)
    p.add_argument("--min-score", type=float, default=0.15)
    args = p.parse_args()

    os.chdir(Path(__file__).parent.parent)

    # Load inputs
    inv_depth = np.load(args.depth)

    with rasterio.open(args.ortho) as src:
        transform = src.transform
        H_o, W_o = src.height, src.width

    print("Loading MNT …")
    data = np.loadtxt(args.mnt, max_rows=5_000_000)
    X_mnt, Y_mnt, Z_mnt = data[:, 0], data[:, 1], data[:, 2]

    from modules.depth_to_elevation import rasterise_mnt
    z_bed = rasterise_mnt(X_mnt, Y_mnt, Z_mnt, transform, (H_o, W_o))

    sp_jit, sp_vars, lg_jit, lg_vars = load_feature_models(args.sp, args.lg)

    result = run_alignment(
        args.frame, inv_depth, args.ortho, z_bed, transform,
        X_mnt, Y_mnt, Z_mnt,
        sp_jit, sp_vars, lg_jit, lg_vars,
        top_k=args.top_k, min_score=args.min_score,
    )

    print(f"\n  Water volume  : {result['volume_m3']:,.1f} m³")
    print(f"  Inundated area: {result['area_m2']:,.0f} m²")
    print(f"  GCPs used     : {len(result['gcp_xyz'])}")
    print(f"  Homography H  :\n{result['H']}")

    paths = visualise_alignment(result, args.frame, args.ortho, args.out_dir)
    print("\nOutputs:")
    for path in paths:
        print(f"  {path}")
