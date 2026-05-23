"""
modules/opyflow_style_viz.py
============================
JAX-equivalent of the three figure-generation code blocks in opyflow's
Test_Brague_flood notebook (apply_opyf_1139_1142.py).

Original reference:
  https://github.com/groussea/opyflow/blob/master/tests/Test_Brague_flood/
  apply_opyf_1139_1142.py  +  figure_Brague.png

Figures produced
----------------
  opyflow_birdeye.png      —  orthorectified frame(s) with GCP markers + axes
                              [birdEyeTransf1139.png style]
  opyflow_velocity_field.png — interpolated velocity colour field on ortho
                              [1139.png + 1142.png style, 2-panel]
  figure_brague.png        —  scatter on ortho + transect cross-section inset + Q
                              [figure_Brague.png — the paper figure]

All three use the local Lambert-93 coordinate system
(origin subtracted: x0=1030760.6875, y0=6289057.0).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BG  = "#0d1117"
TC  = "#e8e8e8"
DIM = "#a0a0a0"

# ── colormap matching opyflow's make_cmap_customized() ────────────────────────

def _make_cmap():
    """8-stop velocity colormap replicating opyflow's custom map."""
    stops = [
        (0.00, (0.0,  0.0,  0.5)),   # dark blue
        (0.15, (0.0,  0.3,  1.0)),   # blue
        (0.30, (0.0,  0.85, 0.9)),   # cyan
        (0.45, (0.0,  0.85, 0.2)),   # green
        (0.60, (0.9,  0.9,  0.0)),   # yellow
        (0.75, (1.0,  0.55, 0.0)),   # orange
        (0.88, (0.9,  0.1,  0.0)),   # red
        (1.00, (0.5,  0.0,  0.5)),   # purple
    ]
    cdict = {"red": [], "green": [], "blue": []}
    for pos, (r, g, b) in stops:
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    return mcolors.LinearSegmentedColormap("opyflow", cdict)


CMAP = _make_cmap()


# ── ortho loader ──────────────────────────────────────────────────────────────

def _load_ortho(ortho_tif: str, max_px: int = 1000):
    """
    Load Ortho.tif, downsample to max_px, return (H,W,3) uint8 + local extent.
    Local coords: subtract (x0=1030760.6875, y0=6289057.0).
    """
    import rasterio
    from rasterio.enums import Resampling

    x0, y0 = 1030760.6875, 6289057.0

    with rasterio.open(ortho_tif) as src:
        scale = max(src.width, src.height) / max_px
        scale = max(1.0, scale)
        w2    = int(src.width  / scale)
        h2    = int(src.height / scale)
        rgb   = src.read([1, 2, 3], out_shape=(3, h2, w2),
                         resampling=Resampling.average)   # (3,H,W)
        bounds = src.bounds

    img = np.moveaxis(rgb, 0, -1)   # (H,W,3) uint8
    extent_local = [
        bounds.left   - x0,
        bounds.right  - x0,
        bounds.bottom - y0,
        bounds.top    - y0,
    ]
    return img, extent_local


def _spine(ax):
    for sp in ax.spines.values():
        sp.set_color("#303040")
    ax.set_facecolor(BG)
    ax.tick_params(colors="#808080", labelsize=7)


# ── Figure 1 : bird-eye orthorectified frames ─────────────────────────────────

def plot_birdeye_frames(
    results:     dict,            # keyed by "1139" and/or "1142"
    gcp_image:   dict,            # {"1139": (N,2), "1142": (N,2)}
    gcp_model:   dict,            # {"1139": (N,3), "1142": (N,3)}  local metres
    out_path:    str = "assets/opyflow_birdeye.png",
):
    """
    Orthorectified first frame for each bridge with GCP markers.
    Equivalent of opyflow's birdEyeTransf1139.png.
    """
    keys   = [k for k in ("1139", "1142") if k in results]
    n_pan  = len(keys)
    if n_pan == 0:
        return

    fig, axes = plt.subplots(1, n_pan, figsize=(7 * n_pan, 6))
    if n_pan == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    titles = {"1139": "IMG_1139 — downstream bridge (DLT orthorectified)",
              "1142": "IMG_1142 — upstream bridge  (DLT orthorectified)"}
    gcp_colors = ["#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff"]

    for ax, key in zip(axes, keys):
        r = results[key]
        _spine(ax)

        ortho_frames = r.get("ortho_imgs", [])
        xg = r.get("x_grid")
        yg = r.get("y_grid")

        if ortho_frames and xg is not None:
            frame0 = np.asarray(ortho_frames[0])
            if frame0.dtype != np.uint8:
                frame0 = (frame0.clip(0, 1) * 255).astype(np.uint8)
            extent = [float(xg.min()), float(xg.max()),
                      float(yg.min()), float(yg.max())]
            ax.imshow(frame0, extent=extent, origin="lower", aspect="equal",
                      alpha=0.92)

        # GCP markers
        if key in gcp_image and key in gcp_model:
            mp = np.asarray(gcp_model[key])
            for i, (pt, col) in enumerate(zip(mp, gcp_colors)):
                ax.plot(pt[0], pt[1], "o", color=col, ms=8,
                        markeredgecolor="w", markeredgewidth=0.8, zorder=5)
                ax.annotate(f"GCP{i+1}\n({pt[0]:.1f},{pt[1]:.1f})",
                            xy=(pt[0], pt[1]),
                            xytext=(pt[0] + 0.5, pt[1] + 0.5),
                            color=col, fontsize=6.5,
                            arrowprops=dict(arrowstyle="-", color=col, lw=0.6))

        ax.set_xlabel("X local (m)", color=DIM, fontsize=8)
        ax.set_ylabel("Y local (m)", color=DIM, fontsize=8)
        ax.set_title(titles.get(key, key), color=TC, fontsize=9, pad=5)
        ax.set_aspect("equal")

    fig.suptitle("Bird-eye (DLT orthorectified) frames — Brague flood 2019",
                 color=TC, fontsize=10, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  opyflow_birdeye → {out_path}")


# ── Figure 2 : velocity colour field on ortho (1139.png / 1142.png style) ────

def plot_velocity_field(
    results:   dict,
    ortho_tif: str,
    out_path:  str = "assets/opyflow_velocity_field.png",
    v_max:     float = 5.0,
):
    """
    Interpolated velocity colour field on orthorectified background —
    one panel per bridge.  Equivalent of opyflow's 1139.png + 1142.png.
    """
    keys   = [k for k in ("1139", "1142") if k in results]
    n_pan  = len(keys)
    if n_pan == 0:
        return

    ortho_img, ortho_extent = _load_ortho(ortho_tif)

    fig, axes = plt.subplots(1, n_pan, figsize=(7 * n_pan, 6))
    if n_pan == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    titles = {"1139": "IMG_1139 — downstream (surface velocity)",
              "1142": "IMG_1142 — upstream  (surface velocity)"}

    for ax, key in zip(axes, keys):
        r = results[key]
        _spine(ax)

        # ortho background
        ax.imshow(ortho_img, extent=ortho_extent,
                  origin="upper", aspect="equal", alpha=0.55, zorder=1)

        X, Y, norm = r["X"], r["Y"], r["norm"]

        # thin gridded vector field using quiver
        sc = ax.scatter(X, Y, c=norm, cmap=CMAP,
                        vmin=0, vmax=v_max,
                        s=4, alpha=0.85, zorder=2, linewidths=0)

        # quiver (sub-sampled)
        k = max(1, len(X) // 800)
        U_n = r["U"][::k]; V_n = r["V"][::k]
        speed = np.hypot(U_n, V_n).clip(1e-6)
        ax.quiver(X[::k], Y[::k], U_n / speed, V_n / speed,
                  norm[::k], cmap=CMAP, clim=(0, v_max),
                  scale=30, width=0.003, alpha=0.6, zorder=3)

        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(r"$\|\vec{U}\|$ (m/s)", color=DIM, fontsize=8)
        cb.ax.tick_params(colors="#808080", labelsize=7)

        ax.set_xlabel("X local (m)", color=DIM, fontsize=8)
        ax.set_ylabel("Y local (m)", color=DIM, fontsize=8)
        ax.set_title(titles.get(key, key), color=TC, fontsize=9, pad=5)
        ax.set_aspect("equal")

        n_vec = len(X)
        ax.text(0.02, 0.02,
                f"{n_vec:,} vectors   v_max={norm.max():.2f} m/s",
                transform=ax.transAxes, color=DIM, fontsize=7)

    fig.suptitle("Surface velocity field — Brague flood 2019 (JAX LSPIV)",
                 color=TC, fontsize=10, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  opyflow_velocity_field → {out_path}")


# ── Figure 3 : figure_Brague.png — scatter + ortho + cross-section + Q ───────

def plot_figure_brague(
    results:   dict,
    ortho_tif: str,
    out_path:  str = "assets/figure_brague.png",
    v_max:     float = 6.0,
    alpha_vel: float = 0.9,
):
    """
    Replicates figure_Brague.png from the opyflow paper:

      Left panel  — velocity scatter on ortho background, transect L→R line
      Right inset — cross-section: velocity profile + bathymetry + Q
      Colorbar    — between the two panels

    Uses combined velocity scatter from both bridges (1139 + 1142) if available.
    Transect data is taken from whichever result has a non-trivial Q.
    """
    ortho_img, ortho_extent = _load_ortho(ortho_tif, max_px=800)

    # ── merge scatter from both bridges ──────────────────────────────────────
    Xtot = np.concatenate([results[k]["X"] for k in results])
    Ytot = np.concatenate([results[k]["Y"] for k in results])
    Ntot = np.concatenate([results[k]["norm"] for k in results])

    # ── pick transect from the result with larger Q ───────────────────────────
    best_key = max(results, key=lambda k: results[k].get("Q", 0))
    r_tr = results[best_key]
    dist_m       = r_tr["dist_m"]
    z_bed_tr     = r_tr["z_bed_transect"]   # relative to water surface
    V_tr         = r_tr["V_transect"]
    transect_pts = r_tr["transect_pts"]     # (N,2)
    Q            = r_tr["Q"]

    # wet mask: where bed is below water surface (z_bed < 0)
    wet = z_bed_tr < 0

    OR_L = transect_pts[0]
    OR_R = transect_pts[-1]

    # ── layout matching figure_Brague.png ─────────────────────────────────────
    fig = plt.figure(figsize=(9, 5))
    fig.patch.set_facecolor(BG)

    ax   = fig.add_axes([0.07, 0.13, 0.42, 0.80])   # scatter + ortho
    axR  = fig.add_axes([0.60, 0.20, 0.35, 0.72])   # cross-section
    axc  = fig.add_axes([0.51, 0.13, 0.012, 0.80])  # colorbar

    for a in (ax, axR):
        _spine(a)

    # ── left panel: scatter on ortho ─────────────────────────────────────────
    ax.imshow(ortho_img, extent=ortho_extent,
              origin="upper", aspect="equal", alpha=0.75, zorder=1)

    sc = ax.scatter(Xtot, Ytot, c=Ntot, cmap=CMAP,
                    vmin=0, vmax=v_max, s=1.5, alpha=alpha_vel,
                    zorder=2, linewidths=0)

    # transect line  (thick green, dashed black overlay — opyflow style)
    ax.plot([OR_L[0], OR_R[0]], [OR_L[1], OR_R[1]],
            "-", lw=8, color=(0.1, 1.0, 0.1, 0.45), zorder=3)
    ax.plot([OR_L[0], OR_R[0]], [OR_L[1], OR_R[1]],
            "--x", lw=1.5, ms=12, color="k", markeredgewidth=1.5,
            zorder=4)

    # L / R labels
    for xy, lbl in ((OR_L, "L"), (OR_R, "R")):
        ax.text(xy[0] - 1.5, xy[1], lbl, color="w", fontsize=14, fontweight="bold",
                bbox=dict(facecolor="purple", alpha=0.65, boxstyle="round,pad=0.2"),
                zorder=5)

    ax.set_xlabel("X local (m)", color=DIM, fontsize=8)
    ax.set_ylabel("Y local (m)", color=DIM, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(colors="#808080", labelsize=7)

    n_total = len(Xtot)
    ax.text(0.02, 0.02,
            f"{n_total:,} vectors  (1139 + 1142)",
            transform=ax.transAxes, color=DIM, fontsize=7)

    fig.colorbar(sc, cax=axc,
                 label=r"$\|\vec{U}\|$ [ m/s ]")
    axc.tick_params(colors="#808080", labelsize=7)
    axc.yaxis.label.set_color(DIM)

    # ── right panel: cross-section (opyflow style) ───────────────────────────
    if wet.sum() > 5:
        d_wet = dist_m[wet]
        V_wet = V_tr[wet]
        z_wet = z_bed_tr[wet]
    else:
        d_wet = dist_m
        V_wet = V_tr
        z_wet = z_bed_tr

    axR.plot(d_wet, V_wet, "--",  lw=1.5, color="#4488ff",
             label="Surface velocity (m/s)")
    axR.plot(d_wet, z_wet, "-.", lw=1.5, color="#88cc44",
             label="Bathymetry rel. water (m)")
    axR.axhline(0, color="#808080", lw=0.8, ls=":")

    # Q annotation (opyflow style)
    d_mid = d_wet.mean() if len(d_wet) > 0 else dist_m.mean()
    v_mid = max(V_wet.max() * 0.7, 0.5) if len(V_wet) > 0 else 1.0
    axR.text(d_mid, v_mid,
             f"Q = {Q:.1f} m³ s⁻¹",
             fontsize=10, color=TC, fontweight="bold",
             bbox=dict(facecolor="#1a1a2a", edgecolor="#3a3a4a",
                       alpha=0.85, boxstyle="round,pad=0.3"))

    # L / R labels on cross-section x-axis
    for d_pos, lbl in ((d_wet[0] if len(d_wet) > 0 else dist_m[0], "L"),
                       (d_wet[-1] if len(d_wet) > 0 else dist_m[-1], "R")):
        axR.text(d_pos, z_wet.min() - 0.3 if len(z_wet) > 0 else -0.5,
                 lbl, color="w", fontsize=13, fontweight="bold",
                 bbox=dict(facecolor="purple", alpha=0.65, boxstyle="round,pad=0.2"))

    # spine style matching opyflow (zero-crossing axes)
    axR.spines["bottom"].set_position(("data", 0))
    axR.spines["left"].set_position(("data", d_wet[0] - 1 if len(d_wet) > 0 else 0))
    axR.spines["right"].set_color("none")
    axR.spines["top"].set_color("none")
    axR.set_xlabel("Distance along transect (m)", color=DIM, fontsize=8)
    axR.xaxis.set_label_coords(0.55, -0.08)
    axR.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor=TC, loc="upper right")
    axR.tick_params(colors="#808080", labelsize=7)
    axR.minorticks_on()
    axR.grid(alpha=0.25)

    fig.suptitle(
        "Brague flood 23/11/2019 — JAX LSPIV (equivalent of opyflow figure_Brague.png)\n"
        f"Q_total ≈ {sum(r.get('Q',0) for r in results.values()):.1f} m³/s  "
        f"(α=0.9 · opyflow paper: 102 ± 20 m³/s)",
        color=TC, fontsize=9, y=1.01,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  figure_brague → {out_path}")


# ── convenience: generate all three from a results dict ──────────────────────

def generate_opyflow_figures(
    results:     dict,
    ortho_tif:   str,
    gcp_image:   dict = None,
    gcp_model:   dict = None,
    assets_dir:  str  = "assets",
):
    """
    Generate all three opyflow-style figures.

    Parameters
    ----------
    results   : {"1139": lspiv_result_dict, "1142": lspiv_result_dict}
    ortho_tif : path to Ortho.tif
    gcp_image : {"1139": (N,2) px, "1142": (N,2) px}  — optional
    gcp_model : {"1139": (N,3) local-m, "1142": (N,3) local-m}  — optional
    """
    d = Path(assets_dir)
    d.mkdir(parents=True, exist_ok=True)

    gcp_image = gcp_image or {}
    gcp_model = gcp_model or {}

    plot_birdeye_frames(
        results, gcp_image, gcp_model,
        out_path=str(d / "opyflow_birdeye.png"),
    )
    plot_velocity_field(
        results, ortho_tif,
        out_path=str(d / "opyflow_velocity_field.png"),
    )
    if results:
        plot_figure_brague(
            results, ortho_tif,
            out_path=str(d / "figure_brague.png"),
        )
