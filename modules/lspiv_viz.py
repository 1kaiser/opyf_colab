"""
modules/lspiv_viz.py
=====================
Visualization for the JAX LSPIV pipeline output — replicates the style of
figure_Brague.png from groussea/opyflow.

Layout (6 panels)
-----------------
  Row 0: [ 3D velocity cloud — frame pair 0 ] [ 3D velocity cloud — all pairs ]
  Row 1: [ Ortho background + velocity scatter + transect line (plan view) ]
         [ Cross-section: MNT bathymetry + velocity profile + Q annotation  ]
  Row 2: [ Orthorectified frame 0 (bird-eye view) ] [ Ortho.tif thumbnail   ]

Usage
-----
    from modules.lspiv_viz import plot_lspiv_results
    plot_lspiv_results(result, ortho_tif="data/brague/Ortho.tif",
                       out_path="assets/lspiv_results.png")
"""

from pathlib import Path
import numpy as np


# Ortho display shift (from opyflow apply_opyf_1139_1142.py)
# shiftX, shiftY centre the Ortho.tif in local-metre coords
ORTHO_SHIFT_X =  14.9397    # local_x of ortho centre
ORTHO_SHIFT_Y = -14.2917    # local_y of ortho centre  (note sign: opyflow −shiftY)
ORTHO_RES_M   =  0.002448   # m/px native resolution


def _load_ortho_local(ortho_tif: str, max_px: int = 800) -> tuple:
    """
    Load Ortho.tif at reduced resolution and convert its extent to local metres.
    Returns (rgb_uint8, extent_local) where extent = [x_min, x_max, y_min, y_max].
    """
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(ortho_tif) as src:
        scale  = max(src.width, src.height) / max_px
        scale  = max(1.0, scale)
        w2 = int(src.width  / scale)
        h2 = int(src.height / scale)
        rgb = src.read([1, 2, 3],
                       out_shape=(3, h2, w2),
                       resampling=Resampling.average)
        b   = src.bounds
        res = (b.right - b.left) / w2   # m/px in thumb

    rgb = np.moveaxis(rgb, 0, -1).astype(np.uint8)

    # Convert Lambert-93 bounds → local metres
    from modules.jax_lspiv import ORIGIN_X, ORIGIN_Y
    x_min_l = b.left   - ORIGIN_X;  x_max_l = b.right - ORIGIN_X
    y_min_l = b.bottom - ORIGIN_Y;  y_max_l = b.top   - ORIGIN_Y
    extent  = [x_min_l, x_max_l, y_min_l, y_max_l]
    return rgb, extent, res


def _make_cmap():
    """Velocity colormap similar to opyflow's custom colormap."""
    import matplotlib.colors as mc
    colors = ["#000080", "#0040ff", "#00cfff", "#80ff40",
              "#ffff00", "#ff8000", "#ff0000", "#8b0000"]
    return mc.LinearSegmentedColormap.from_list("opyflow", colors)


def plot_lspiv_results(result:   dict,
                       ortho_tif: str  = "data/brague/Ortho.tif",
                       out_path:  str  = "assets/lspiv_results.png",
                       v_max:     float = None):
    """
    6-panel figure replicating the opyflow figure_Brague.png style.

    Parameters
    ----------
    result    : dict returned by jax_lspiv.run_jax_lspiv()
    ortho_tif : path to Ortho.tif
    out_path  : output PNG path
    v_max     : colour scale max velocity (m/s); None = auto 95th percentile
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgridspec
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D

    X      = result["X"];     Y    = result["Y"]
    U      = result["U"];     V    = result["V"]
    norm   = result["norm"]
    x_grid = result["x_grid"]; y_grid = result["y_grid"]
    ortho_frames = result["ortho_frames"]

    terrain_xy = result["terrain_xy"];  terrain_z = result["terrain_z"]
    trans_pts  = result["transect_pts"]
    dist_m     = result["dist_m"]
    z_bed_t    = result["z_bed_transect"]
    norm_t     = result["norm_transect"]
    Q          = result["Q"]
    z_water    = result["z_water"]

    from modules.jax_lspiv import TRANSECT_L, TRANSECT_R

    cmap  = _make_cmap()
    v_max = v_max or float(np.percentile(norm, 95)) or 3.0
    BG = "#0d1117";  TC = "#e8e8e8";  DIM = "#a0a0a0"

    # ── load ortho thumbnail ──────────────────────────────────────────
    ortho_rgb, ortho_extent, _ = _load_ortho_local(ortho_tif)

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs = mgridspec.GridSpec(
        3, 2, figure=fig,
        left=0.06, right=0.97, top=0.95, bottom=0.04,
        hspace=0.32, wspace=0.22,
        height_ratios=[1.2, 1.4, 1.0],
    )

    def _spine(ax):
        for sp in ax.spines.values(): sp.set_color("#303040")
        ax.tick_params(colors="#808080", labelsize=7)
        ax.set_facecolor(BG)

    # ── Panel A: 3D velocity cloud (pair 0) ──────────────────────────
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    axA.set_facecolor(BG)
    for pane in (axA.xaxis.pane, axA.yaxis.pane, axA.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#252535")
    k = max(1, len(X) // 30_000)
    sc = axA.scatter(X[::k], Y[::k], norm[::k], c=norm[::k],
                     cmap=cmap, vmin=0, vmax=v_max,
                     s=0.8, alpha=0.8, depthshade=True)
    axA.set_xlabel("X local (m)", color=DIM, fontsize=7, labelpad=2)
    axA.set_ylabel("Y local (m)", color=DIM, fontsize=7, labelpad=2)
    axA.set_zlabel("|V| (m/s)",   color=DIM, fontsize=7, labelpad=2)
    axA.tick_params(colors="#808080", labelsize=6)
    axA.view_init(elev=30, azim=-60)
    axA.set_title("A — 3D velocity scatter  (JAX-PIV, all pairs)",
                  color=TC, fontsize=9, pad=5)

    cb = fig.colorbar(sc, ax=axA, fraction=0.025, pad=0.08, shrink=0.7)
    cb.set_label("|V| (m/s)", color=DIM, fontsize=7)
    cb.ax.tick_params(colors="#808080", labelsize=6)

    # ── Panel B: 3D terrain coloured by velocity interpolated onto MNT ─
    axB = fig.add_subplot(gs[0, 1], projection="3d")
    axB.set_facecolor(BG)
    for pane in (axB.xaxis.pane, axB.yaxis.pane, axB.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#252535")
    # Subsample terrain for display
    kt = max(1, len(terrain_z) // 30_000)
    scT = axB.scatter(terrain_xy[::kt, 0], terrain_xy[::kt, 1],
                      terrain_z[::kt],
                      c=terrain_z[::kt], cmap="terrain",
                      s=0.5, alpha=0.7, depthshade=True)
    # Transect line
    axB.plot([TRANSECT_L[0], TRANSECT_R[0]],
             [TRANSECT_L[1], TRANSECT_R[1]],
             [z_water, z_water],
             color="#00ff80", linewidth=2.5, zorder=6, label="Transect")
    # Water surface plane
    axB.plot_surface(
        np.array([[x_grid.min(), x_grid.max()],
                  [x_grid.min(), x_grid.max()]]),
        np.array([[y_grid.min(), y_grid.min()],
                  [y_grid.max(), y_grid.max()]]),
        np.full((2, 2), z_water),
        alpha=0.15, color="#3080d0", zorder=2,
    )
    axB.set_xlabel("X local (m)", color=DIM, fontsize=7, labelpad=2)
    axB.set_ylabel("Y local (m)", color=DIM, fontsize=7, labelpad=2)
    axB.set_zlabel("Z (m MSL)",   color=DIM, fontsize=7, labelpad=2)
    axB.tick_params(colors="#808080", labelsize=6)
    axB.view_init(elev=30, azim=-55)
    axB.set_title(f"B — MNT terrain + water plane (z={z_water} m) + transect",
                  color=TC, fontsize=9, pad=5)
    axB.legend(fontsize=7, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor=TC, loc="upper left")

    # ── Panel C: Plan view — Ortho + velocity scatter + transect ─────
    axC = fig.add_subplot(gs[1, 0])
    _spine(axC)
    axC.imshow(ortho_rgb, extent=ortho_extent, origin="upper", alpha=0.8, zorder=1)
    sc2 = axC.scatter(X, Y, c=norm, cmap=cmap, vmin=0, vmax=v_max,
                      s=1.0, alpha=0.6, zorder=3)
    # Transect
    axC.plot([TRANSECT_L[0], TRANSECT_R[0]],
             [TRANSECT_L[1], TRANSECT_R[1]],
             "-x", color="#00ff80", linewidth=3, markersize=12,
             markeredgewidth=2, zorder=5, label="Transect")
    axC.text(TRANSECT_L[0] - 1.5, TRANSECT_L[1] + 0.3, "L",
             color="white", fontsize=13, fontweight="bold",
             bbox=dict(fc="purple", alpha=0.6), zorder=6)
    axC.text(TRANSECT_R[0] - 1.5, TRANSECT_R[1] - 0.6, "R",
             color="white", fontsize=13, fontweight="bold",
             bbox=dict(fc="purple", alpha=0.6), zorder=6)
    axC.set_xlim(ortho_extent[0], ortho_extent[1])
    axC.set_ylim(ortho_extent[2], ortho_extent[3])
    axC.set_aspect("equal")
    axC.set_xlabel("X local (m)", color=DIM, fontsize=8)
    axC.set_ylabel("Y local (m)", color=DIM, fontsize=8)
    axC.set_title("C — Plan view: Ortho + JAX-PIV velocity + transect",
                  color=TC, fontsize=9, pad=5)
    axC.legend(fontsize=8, facecolor="#1a1a2a", edgecolor="#3a3a4a", labelcolor=TC)
    cb2 = fig.colorbar(sc2, ax=axC, fraction=0.03, pad=0.02, shrink=0.8)
    cb2.set_label("|V| (m/s)", color=DIM, fontsize=7)
    cb2.ax.tick_params(colors="#808080", labelsize=6)

    # ── Panel D: Cross-section (MNT bathymetry + velocity profile) ───
    axD = fig.add_subplot(gs[1, 1])
    _spine(axD)

    h_bathy  = z_bed_t - z_water                         # negative below water
    wet_mask = h_bathy <= 0

    axD.plot(dist_m, h_bathy, "-.", color="#40e080", linewidth=1.8,
             label="Bathymetry (m rel. water surface)")
    axD.plot(dist_m[wet_mask], norm_t[wet_mask], "--",
             color="#60b0ff", linewidth=1.5,
             label="|V| velocity (m/s)")
    axD.fill_between(dist_m, h_bathy, 0,
                     where=wet_mask, color="#1a4080", alpha=0.4)
    axD.axhline(0, color="#30b0e0", lw=1.2, ls="--", alpha=0.8, label="Water surface")

    axD.spines["bottom"].set_position(("data", 0))
    axD.spines["left"].set_position(("data", dist_m[0]))
    axD.spines["right"].set_color("none")
    axD.spines["top"].set_color("none")

    Q_txt = f"Q = {Q:.1f} m³/s\n(α={result['alpha']:.1f},  z_w={z_water} m)"
    axD.text(0.97, 0.97, Q_txt, transform=axD.transAxes,
             color=TC, fontsize=10, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2a", ec="#40c080", alpha=0.9))
    axD.text(dist_m[0] - 0.3, 0.0, "L",
             color="white", fontsize=13, fontweight="bold",
             bbox=dict(fc="purple", alpha=0.6))
    axD.text(dist_m[-1] + 0.1, 0.0, "R",
             color="white", fontsize=13, fontweight="bold",
             bbox=dict(fc="purple", alpha=0.6))

    axD.set_xlabel("Distance along transect (m)", color=DIM, fontsize=8)
    axD.set_ylabel("Relative elevation / |V| (m or m/s)", color=DIM, fontsize=8)
    axD.set_title("D — Cross-section: bathymetry + velocity → discharge Q",
                  color=TC, fontsize=9, pad=5)
    axD.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor=TC, loc="lower center")
    axD.set_xlim(dist_m[0] - 0.5, dist_m[-1] + 0.5)

    # ── Panel E: Orthorectified frame (bird-eye view) ─────────────────
    axE = fig.add_subplot(gs[2, 0])
    _spine(axE)
    if len(ortho_frames) > 0:
        axE.imshow(ortho_frames[0], cmap="gray", origin="upper",
                   extent=[x_grid[0], x_grid[-1], y_grid[-1], y_grid[0]],
                   vmin=0, vmax=1)
    axE.set_xlabel("X local (m)", color=DIM, fontsize=8)
    axE.set_ylabel("Y local (m)", color=DIM, fontsize=8)
    axE.set_title("E — Orthorectified frame 0 (bird-eye view, DLT homography)",
                  color=TC, fontsize=9, pad=5)

    # ── Panel F: Ortho.tif thumbnail (reference) ─────────────────────
    axF = fig.add_subplot(gs[2, 1])
    _spine(axF)
    axF.imshow(ortho_rgb, extent=ortho_extent, origin="upper")
    axF.set_aspect("equal")
    axF.set_xlabel("X local (m)", color=DIM, fontsize=8)
    axF.set_ylabel("Y local (m)", color=DIM, fontsize=8)
    axF.set_title("F — Ortho.tif reference (pre-event aerial photo, Lambert-93)",
                  color=TC, fontsize=9, pad=5)

    # ── overall title ─────────────────────────────────────────────────
    n_vec = len(X)
    fig.suptitle(
        f"JAX LSPIV — Brague Flood 23 Nov 2019 · Biot Bridge  ·  "
        f"{n_vec:,} velocity vectors  ·  Q = {Q:.1f} m³/s  ·  "
        f"z_water = {z_water} m MSL",
        color=TC, fontsize=10, y=0.98,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  LSPIV results figure → {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from modules.jax_lspiv import run_jax_lspiv, GCP_IMAGE_1139, GCP_MODEL_1139
    import glob

    frame_paths = sorted(glob.glob("output/brague/frames/frame_0*.png"))[:4]
    print(f"Using {len(frame_paths)} frames: {[str(p) for p in frame_paths]}")

    result = run_jax_lspiv(
        frame_paths   = frame_paths,
        image_points  = GCP_IMAGE_1139,
        model_points  = GCP_MODEL_1139,
        mnt_xyz_path  = "data/brague/MNT.xyz",
        win_size      = 32,
        step          = 16,
        fps           = 30.0,
        res_m         = 0.02,
        interp_radius = 1.5,
        mnt_stride    = 50,
    )

    plot_lspiv_results(
        result,
        ortho_tif = "data/brague/Ortho.tif",
        out_path  = "assets/lspiv_results.png",
    )
    print("Done.")
