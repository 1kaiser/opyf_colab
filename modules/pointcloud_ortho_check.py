"""
modules/pointcloud_ortho_check.py
===================================
Check spatial alignment between MNT.xyz (pre-event terrain point cloud,
Lambert-93) and Ortho.tif (RGB orthophoto, same CRS) by colorising each
sampled MNT point with the Ortho pixel at that (X, Y) location.

A well-aligned pair produces a textured terrain where roads, vegetation and
the canal bed match their visible positions in the orthophoto.

Panels
------
  A — 3D coloured point cloud  (Ortho RGB draped on MNT Z)
  B — Top-down Ortho with MNT contours overlaid
  C — Elevation heatmap of sampled cloud (plan view)
  D — Z histogram split by dominant colour class (vegetation / urban / water)

Usage
-----
    python3 modules/pointcloud_ortho_check.py
    # or
    from modules.pointcloud_ortho_check import generate_pointcloud_ortho_check
    generate_pointcloud_ortho_check(
        mnt_xyz   = "data/brague/MNT.xyz",
        ortho_tif = "data/brague/Ortho.tif",
        out_path  = "assets/pointcloud_ortho_check.png",
        stride    = 100,
    )
"""

from pathlib import Path
import math
import subprocess
import numpy as np


def _sample_xyz(mnt_xyz: str, stride: int = 100) -> np.ndarray:
    """Return (N,3) float32 array by keeping every `stride`-th line."""
    # awk is faster than Python line-iteration for large files
    tmp = "/tmp/_mnt_sample.xyz"
    subprocess.run(
        f"awk 'NR % {stride} == 0' '{mnt_xyz}' > {tmp}",
        shell=True, check=True,
    )
    pts = np.loadtxt(tmp, dtype=np.float32)   # shape (N, 3): X Y Z
    return pts


def _sample_ortho_colors(pts_xy: np.ndarray, ortho_tif: str,
                          thumb_px: int = 1200) -> np.ndarray:
    """
    For each (X, Y) in pts_xy, return the RGB colour from Ortho.tif.
    Loads the ortho at `thumb_px` resolution to keep memory reasonable.

    Returns (N, 3) uint8 array.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(ortho_tif) as src:
        scale = max(src.width, src.height) / thumb_px
        scale = max(1.0, scale)
        w2 = int(src.width  / scale)
        h2 = int(src.height / scale)
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, h2, w2),
            resampling=Resampling.average,
        )                          # (3, H, W) uint8
        bounds = src.bounds

    # pixel resolution in the thumb
    res_x = (bounds.right  - bounds.left) / w2
    res_y = (bounds.top    - bounds.bottom) / h2

    # map Lambert-93 (X,Y) → pixel indices in thumb
    col_f = (pts_xy[:, 0] - bounds.left)   / res_x
    row_f = (bounds.top   - pts_xy[:, 1])  / res_y

    col_i = np.clip(col_f.astype(int), 0, w2 - 1)
    row_i = np.clip(row_f.astype(int), 0, h2 - 1)

    colors = rgb[:, row_i, col_i].T     # (N, 3) uint8
    return colors, rgb, bounds, w2, h2


def generate_pointcloud_ortho_check(
    mnt_xyz:   str = "data/brague/MNT.xyz",
    ortho_tif: str = "data/brague/Ortho.tif",
    out_path:  str = "assets/pointcloud_ortho_check.png",
    stride:    int = 100,
):
    """
    Generate 4-panel point-cloud × ortho alignment figure.
    stride=100 → ~320 K points from the 31.8 M-point Brague MNT.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgridspec
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D

    print(f"  Sampling MNT.xyz (stride={stride}) …")
    pts = _sample_xyz(mnt_xyz, stride=stride)          # (N, 3): X Y Z
    print(f"  Loaded {len(pts):,} points")

    print("  Sampling Ortho.tif colours …")
    colors, ortho_rgb, bounds, ow, oh = _sample_ortho_colors(
        pts[:, :2], ortho_tif, thumb_px=1200)          # colors (N,3) uint8

    colors_f = colors.astype(np.float32) / 255.0       # (N, 3) 0–1

    X = pts[:, 0];  Y = pts[:, 1];  Z = pts[:, 2]

    # ── colour-class labels (simple RGB heuristic) ──────────────────
    R, G, B_ch = colors_f[:, 0], colors_f[:, 1], colors_f[:, 2]
    is_veg   = (G > R) & (G > B_ch) & (G > 0.25)
    is_water = (B_ch > R) & (B_ch > G * 1.1) & (B_ch < 0.55)
    is_urban = ~is_veg & ~is_water
    labels   = np.where(is_veg, 0, np.where(is_water, 1, 2))
    label_names = ["Vegetation (G>R)", "Water/shadow (B>G)", "Urban/bare"]
    label_cols  = ["#30c060", "#3080d0", "#c09060"]

    # ── figure ────────────────────────────────────────────────────────
    BG = "#0d1117";  TC = "#e8e8e8";  DIM = "#a0a0a0"
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)
    gs = mgridspec.GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97, top=0.93, bottom=0.05,
        hspace=0.30, wspace=0.22,
    )

    def _spine(ax):
        for sp in ax.spines.values(): sp.set_color("#303040")
        ax.tick_params(colors="#808080", labelsize=6)
        ax.set_facecolor(BG)

    # ── Panel A: 3D coloured point cloud ─────────────────────────────
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    axA.set_facecolor(BG)
    for pane in (axA.xaxis.pane, axA.yaxis.pane, axA.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#252535")

    # subsample further for 3D scatter (max 80 K for responsiveness)
    k  = max(1, len(pts) // 80_000)
    sc = axA.scatter(
        X[::k], Y[::k], Z[::k],
        c=colors_f[::k], s=0.4, alpha=0.7, depthshade=True,
    )
    axA.set_xlabel("X Lambert-93 (m)", color=DIM, fontsize=7, labelpad=3)
    axA.set_ylabel("Y Lambert-93 (m)", color=DIM, fontsize=7, labelpad=3)
    axA.set_zlabel("Z (m)",            color=DIM, fontsize=7, labelpad=2)
    axA.tick_params(colors="#808080", labelsize=6)
    axA.view_init(elev=35, azim=-50)
    axA.set_title(
        f"A — 3D Coloured Point Cloud  (Ortho RGB draped on MNT Z)\n"
        f"{len(pts[::k]):,} pts shown of {len(pts):,} sampled  "
        f"(stride={stride}×  →  every {stride}th of 31.8 M)",
        color=TC, fontsize=8, pad=6,
    )

    # Z range annotation
    z_range = f"Z: {Z.min():.2f} – {Z.max():.2f} m   ΔZ={Z.max()-Z.min():.2f} m"
    axA.text2D(0.02, 0.02, z_range, transform=axA.transAxes,
               color=DIM, fontsize=7)

    # ── Panel B: Ortho image + MNT elevation contours (top-down) ─────
    axB = fig.add_subplot(gs[0, 1])
    _spine(axB)

    ortho_show = np.moveaxis(ortho_rgb[:3], 0, -1).astype(np.float32) / 255.0
    axB.imshow(
        ortho_show,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper", alpha=0.85, zorder=1,
    )

    # MNT Z as scatter (colour = Z, tiny dots)
    sc2 = axB.scatter(
        X, Y, c=Z, cmap="terrain", s=0.05,
        alpha=0.45, vmin=Z.min(), vmax=Z.max(), zorder=2,
    )
    cb = fig.colorbar(sc2, ax=axB, fraction=0.03, pad=0.02)
    cb.set_label("MNT elevation (m)", color=DIM, fontsize=7)
    cb.ax.tick_params(colors="#808080", labelsize=6)

    axB.set_xlabel("X Lambert-93 (m)", color=DIM, fontsize=7)
    axB.set_ylabel("Y Lambert-93 (m)", color=DIM, fontsize=7)
    axB.set_title("B — Top-down: Ortho + MNT elevation overlay",
                  color=TC, fontsize=9, pad=5)

    # ── Panel C: Plan-view elevation heatmap (MNT only) ──────────────
    axC = fig.add_subplot(gs[1, 0])
    _spine(axC)

    # bin points onto a 600×600 grid for a smooth heatmap
    nx = ny = 600
    xi = np.linspace(X.min(), X.max(), nx)
    yi = np.linspace(Y.min(), Y.max(), ny)
    xi_idx = np.clip(((X - X.min()) / (X.max() - X.min()) * (nx-1)).astype(int), 0, nx-1)
    yi_idx = np.clip(((Y - Y.min()) / (Y.max() - Y.min()) * (ny-1)).astype(int), 0, ny-1)
    zmap   = np.full((ny, nx), np.nan)
    np.maximum.at(zmap, (yi_idx, xi_idx), Z)   # max-Z per cell (top surface)

    im = axC.imshow(
        np.flipud(zmap),
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap="terrain", origin="lower", interpolation="bilinear",
    )
    cb2 = fig.colorbar(im, ax=axC, fraction=0.03, pad=0.02)
    cb2.set_label("Elevation (m)", color=DIM, fontsize=7)
    cb2.ax.tick_params(colors="#808080", labelsize=6)

    axC.set_xlabel("X Lambert-93 (m)", color=DIM, fontsize=7)
    axC.set_ylabel("Y Lambert-93 (m)", color=DIM, fontsize=7)
    axC.set_title("C — MNT elevation map (plan, terrain colormap)",
                  color=TC, fontsize=9, pad=5)

    # ── Panel D: Z histogram by colour class ─────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    _spine(axD)

    bins = np.linspace(Z.min(), Z.max(), 60)
    for lbl_idx, (name, col) in enumerate(zip(label_names, label_cols)):
        mask = labels == lbl_idx
        if mask.sum() > 10:
            axD.hist(Z[mask], bins=bins, color=col, alpha=0.6,
                     label=f"{name}  ({mask.sum():,} pts)", density=True)

    axD.set_xlabel("MNT elevation Z (m)", color=DIM, fontsize=8)
    axD.set_ylabel("Density", color=DIM, fontsize=8)
    axD.set_title(
        "D — Elevation distribution by Ortho colour class\n"
        "(confirms water/shadow pixels sit in lowest terrain zones)",
        color=TC, fontsize=9, pad=5,
    )
    axD.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor=TC)

    # ── overall stats box ─────────────────────────────────────────────
    stats = (
        f"MNT.xyz   {len(pts):,} pts sampled (stride {stride}×)   "
        f"Z: {Z.min():.3f}–{Z.max():.3f} m\n"
        f"Ortho.tif  {ow}×{oh} px (thumb)   "
        f"bounds match: ΔX={bounds.right-bounds.left:.2f} m "
        f"ΔY={bounds.top-bounds.bottom:.2f} m   EPSG:2154"
    )
    fig.suptitle(
        "MNT.xyz × Ortho.tif — 3D Point-Cloud / Orthophoto Alignment Check\n" + stats,
        color=TC, fontsize=9, y=0.98,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Point-cloud ortho check → {out_path}")


if __name__ == "__main__":
    generate_pointcloud_ortho_check()
