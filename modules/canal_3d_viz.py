"""
modules/canal_3d_viz.py
========================
Geo-referenced 3D visualization of the canal design placed on the actual
terrain derived from flow_depth.tif + z_surface.tif.

Four-panel figure
-----------------
  Panel A (3D)   — terrain Z_bed surface, flood surface, and the designed
                   canal section extruded along the reach centreline
  Panel B (plan) — ortho image with wet mask, centreline, and canal footprint
  Panel C (long) — longitudinal profile: Z_bed, flood surface, canal invert
  Panel D (xsec) — mid-reach cross-section: terrain + flood + designed canal

Public API
----------
  generate_canal_3d_overlay(
      flow_depth_tif, z_surface_tif, ortho_tif,
      canal_params, reach_geo, out_path)
"""

from pathlib import Path
import math
import numpy as np


def _load_rasters(flow_depth_tif: str, z_surface_tif: str, sub: int = 6):
    """Load and subsample the depth rasters; derive Z_bed."""
    import rasterio
    with rasterio.open(flow_depth_tif) as src:
        h_full  = src.read(1).astype(np.float32)
        t       = src.transform
        bounds  = src.bounds
    with rasterio.open(z_surface_tif) as src2:
        zs_full = src2.read(1).astype(np.float32)

    h  = h_full [::sub, ::sub]
    zs = zs_full[::sub, ::sub]
    zb = np.where((h > 0.05) & (zs > 0), zs - h, np.nan)

    res  = abs(float(t.a)) * sub
    x0, y0 = float(bounds.left), float(bounds.top)
    nrows, ncols = h.shape
    xs   = x0 + np.arange(ncols) * res
    ys   = y0 - np.arange(nrows) * res
    X, Y = np.meshgrid(xs, ys)
    return h, zs, zb, X, Y, res, t, bounds


def _load_ortho_thumb(ortho_tif: str, max_px: int = 800):
    """Load Ortho.tif downsampled to ≤ max_px on the longest side."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(ortho_tif) as src:
        factor = max(src.width, src.height) / max_px
        factor = max(1, int(factor))
        w2 = src.width  // factor
        h2 = src.height // factor
        rgb = src.read([1, 2, 3],
                       out_shape=(3, h2, w2),
                       resampling=Resampling.average)
        bounds = src.bounds
    rgb = np.moveaxis(rgb, 0, -1)
    rgb = np.clip(rgb / rgb.max(), 0, 1)
    return rgb, bounds


def _canal_footprint(centerline_x, centerline_y, top_width_m):
    """
    Return left and right edge coordinates of the canal footprint (plan view).
    Computed as perpendicular offset ±top_width/2 from each centreline point.
    """
    left_x, left_y   = [], []
    right_x, right_y = [], []
    n = len(centerline_x)
    for i in range(n):
        # local tangent direction
        i0, i1 = max(0, i - 1), min(n - 1, i + 1)
        dx = centerline_x[i1] - centerline_x[i0]
        dy = centerline_y[i1] - centerline_y[i0]
        length = math.hypot(dx, dy) or 1e-9
        nx, ny = -dy / length, dx / length   # normal (perpendicular)
        half = top_width_m / 2
        left_x.append(centerline_x[i] + nx * half)
        left_y.append(centerline_y[i] + ny * half)
        right_x.append(centerline_x[i] - nx * half)
        right_y.append(centerline_y[i] - ny * half)
    return (np.array(left_x),  np.array(left_y),
            np.array(right_x), np.array(right_y))


def _canal_section_xy(B, D, m, fb, z_invert):
    """Trapezoidal section vertices in (width_offset, elevation) space."""
    total = D + fb
    pts = np.array([
        [-(B/2 + m*total), z_invert + total],
        [-(B/2 + m*D),     z_invert + D],
        [-B/2,             z_invert],
        [ B/2,             z_invert],
        [ B/2 + m*D,       z_invert + D],
        [ B/2 + m*total,   z_invert + total],
    ])
    return pts


def generate_canal_3d_overlay(
    flow_depth_tif: str,
    z_surface_tif:  str,
    ortho_tif:      str,
    canal_params:   dict,
    reach_geo:      dict,
    out_path:       str = "assets/canal_3d_overlay.png",
    sub:            int = 6,
    d8_result:      dict | None = None,
):
    """
    Parameters
    ----------
    flow_depth_tif : str
    z_surface_tif  : str   one frame's z_surface.tif (used for Z_bed)
    ortho_tif      : str
    canal_params   : dict  output of optimise_canal()
    reach_geo      : dict  output of extract_reach()
    sub            : int   spatial subsampling factor
    d8_result      : dict  optional output of extract_d8_thalweg() — when
                           provided, the D8 thalweg + accumulation are used for
                           the curvature panel and the plan-view overlay
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from modules.extract_reach_geometry import longitudinal_profile

    B   = canal_params["bed_width_m"]
    D   = canal_params["water_depth_m"]
    m   = canal_params["side_slope"]
    fb  = canal_params["freeboard_m"]
    Q   = canal_params["Q_calculated_m3s"]
    V   = canal_params["velocity_ms"]
    Tw  = canal_params["top_width_m"]

    cx  = reach_geo["centerline_x"]
    cy  = reach_geo["centerline_y"]

    # ── load rasters ─────────────────────────────────────────────────
    print("  Loading rasters …")
    h, zs, zb, X, Y, res, t_full, bounds = _load_rasters(
        flow_depth_tif, z_surface_tif, sub=sub)
    ortho_rgb, ortho_bounds = _load_ortho_thumb(ortho_tif, max_px=600)

    # fill NaN in zb with nearest valid (for surface plot)
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(zb)
    if nan_mask.any():
        idx = distance_transform_edt(nan_mask, return_distances=False,
                                     return_indices=True)
        zb_filled = zb[tuple(idx)]
    else:
        zb_filled = zb

    # ── longitudinal profile ─────────────────────────────────────────
    import rasterio as _rio
    with _rio.open(flow_depth_tif) as src:
        t_full_tf = src.transform
    h_full_arr = np.where(h > 0.05, h, 0)
    zb_full    = np.where(h > 0.05, zs - h, np.nan)

    # use subsampled zb for profile
    dist_m, z_prof, S_long_meas = longitudinal_profile(
        cx, cy, zb_filled,
        _rio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            zb_filled.shape[1], zb_filled.shape[0]))

    z_flood_prof   = z_prof + np.interp(
        dist_m,
        np.linspace(0, dist_m[-1], len(cx)),
        np.clip(h[reach_geo["centerline_row"][::max(1, len(reach_geo["centerline_row"])//len(cx))]
                  [:len(cx)],
                  reach_geo["centerline_col"][::max(1, len(reach_geo["centerline_col"])//len(cx))]
                  [:len(cx)]], 0, None)) \
        if False else z_prof + reach_geo.get("h_mean", 1.15)

    # canal invert follows terrain slope (cut to design depth at each point)
    z_invert_prof  = z_prof - 0.0   # bed = terrain (canal placed at surface level)
    S_design       = canal_params.get("long_slope", S_long_meas)

    # ── curvature: prefer D8 thalweg geometry if available ────────────
    if d8_result is not None:
        geo_d8 = d8_result["geometry"]
        cx_curv = d8_result["centerline_x"]
        cy_curv = d8_result["centerline_y"]
        R_curve = geo_d8["radius_m"]
        dist_curv = geo_d8["dist_m"]
        S_long_meas = geo_d8["S_long"]
    else:
        from modules.extract_reach_geometry import curvature_profile
        kappa   = curvature_profile(cx, cy)
        with np.errstate(divide="ignore", invalid="ignore"):
            R_curve = np.where(kappa > 1e-9, 1.0 / kappa, np.inf)
        dist_curv = np.concatenate([[0.0], np.cumsum(
            np.sqrt(np.diff(cx)**2 + np.diff(cy)**2))])
        cx_curv, cy_curv = cx, cy
        S_long_meas = 0.0002   # fallback

    R_min   = float(np.nanmin(R_curve[np.isfinite(R_curve)])) if np.any(np.isfinite(R_curve)) else np.inf
    R_IS    = canal_params.get("min_curve_radius_m", 1000.0)

    # ── figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#0d1117")

    # 3-column top row (A, B, C) + 2-column bottom row (D, E)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.30)

    # Panel A: 3D terrain + flood + canal section
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_facecolor("#0d1117")

    # terrain surface
    zb_plot = np.ma.masked_invalid(zb_filled)
    terrain_norm = mcolors.Normalize(vmin=np.nanmin(zb_filled),
                                     vmax=np.nanmax(zb_filled))
    ax3d.plot_surface(X, Y, zb_filled,
                      facecolors=plt.cm.terrain(terrain_norm(zb_filled)),
                      alpha=0.85, linewidth=0, antialiased=False, zorder=1)

    # flood surface
    flood_z = np.where(h > 0.05, zb_filled + h, np.nan)
    ax3d.plot_surface(X, Y, np.nan_to_num(flood_z, nan=np.nanmin(zb_filled)),
                      color="#1a6fa8", alpha=0.35, linewidth=0, zorder=2)

    # canal cross-section at mid-reach (representative cross-section)
    mid = len(cx) // 2
    z_bed_mid = float(np.nanmedian(zb_filled))
    sec = _canal_section_xy(B, D, m, fb, z_invert=z_bed_mid)
    # orient section perpendicular to centreline at mid
    if mid > 0 and mid < len(cx) - 1:
        tdx = cx[mid+1] - cx[mid-1]
        tdy = cy[mid+1] - cy[mid-1]
        tlen = math.hypot(tdx, tdy) or 1
        nx, ny = -tdy/tlen, tdx/tlen
    else:
        nx, ny = 1.0, 0.0

    sec_x3d = cx[mid] + sec[:, 0] * nx
    sec_y3d = cy[mid] + sec[:, 0] * ny
    sec_z3d = sec[:, 1]
    ax3d.plot(sec_x3d, sec_y3d, sec_z3d,
              color="#f0c040", lw=2.5, zorder=5, label=f"Canal B={B:.1f}m D={D:.1f}m")
    verts3d = [list(zip(sec_x3d, sec_y3d, sec_z3d))]
    poly3d = Poly3DCollection(verts3d, alpha=0.45,
                              facecolor="#1a6fa8", edgecolor="none")
    ax3d.add_collection3d(poly3d)

    # centreline
    z_cl = np.interp(np.linspace(0, 1, len(cx)),
                     np.linspace(0, 1, zb_filled.shape[0]),
                     np.nanmedian(zb_filled, axis=1))
    ax3d.plot(cx, cy, z_cl + 0.05,
              color="#ff6060", lw=1.5, ls="--", zorder=4, label="Centreline")

    ax3d.set_xlabel("X Lambert-93 (m)", color="#a0a0a0", fontsize=7, labelpad=4)
    ax3d.set_ylabel("Y Lambert-93 (m)", color="#a0a0a0", fontsize=7, labelpad=4)
    ax3d.set_zlabel("Elevation (m)", color="#a0a0a0", fontsize=7, labelpad=2)
    ax3d.set_title("A — 3D Terrain + Flood + Canal Section",
                   color="#e0e0e0", fontsize=9, pad=6)
    ax3d.tick_params(colors="#808080", labelsize=6)
    ax3d.legend(fontsize=7, facecolor="#1a1a2a", edgecolor="#3a3a4a",
                labelcolor="#e0e0e0", loc="upper left")
    ax3d.view_init(elev=30, azim=-60)

    # ── Panel B: Plan view — ortho + wet mask + canal footprint ───────
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.set_facecolor("#0d1117")
    ax2d.imshow(ortho_rgb,
                extent=[ortho_bounds.left, ortho_bounds.right,
                        ortho_bounds.bottom, ortho_bounds.top],
                origin="upper", aspect="equal", alpha=0.75)

    # wet mask outline
    wet_mask = h > 0.05
    ax2d.contour(X, Y, wet_mask.astype(float), levels=[0.5],
                 colors=["#56c8ff"], linewidths=1.2, linestyles="-",
                 alpha=0.7)

    # canal footprint
    lx, ly, rx, ry = _canal_footprint(cx, cy, Tw)
    fp_x = np.concatenate([lx, rx[::-1], [lx[0]]])
    fp_y = np.concatenate([ly, ry[::-1], [ly[0]]])
    ax2d.fill(fp_x, fp_y, color="#f0c040", alpha=0.35, label=f"Canal footprint\nTop width {Tw:.1f} m")
    ax2d.plot(np.concatenate([lx, rx[::-1], [lx[0]]]),
              np.concatenate([ly, ry[::-1], [ly[0]]]),
              color="#f0c040", lw=1.5)
    # D8 flow accumulation heatmap in plan view
    if d8_result is not None:
        acc_sub = d8_result["acc_np"]
        wet_sub = d8_result["wet_np"]
        t_sub   = d8_result["transform_sub"]
        r_sub   = d8_result["res_m"]
        H_s, W_s = acc_sub.shape
        xs_d8 = float(t_sub.c) + (np.arange(W_s) + 0.5) * r_sub
        ys_d8 = float(t_sub.f) - (np.arange(H_s) + 0.5) * r_sub
        acc_show = np.where(wet_sub, np.log1p(acc_sub), np.nan)
        ax2d.pcolormesh(xs_d8, ys_d8, acc_show,
                        cmap="Blues", alpha=0.55, zorder=2,
                        shading="nearest")

    # colour the centreline by curvature radius (clipped to [R_IS/4, 10*R_IS])
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mc2
    R_clamp = np.clip(R_curve, R_IS / 4, R_IS * 4)
    R_clamp[~np.isfinite(R_clamp)] = R_IS * 4
    seg_pts = np.array([cx_curv, cy_curv]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([seg_pts[:-1], seg_pts[1:]], axis=1)
    norm_c  = mc2.Normalize(vmin=R_IS / 4, vmax=R_IS * 4)
    lc = LineCollection(segs, cmap="RdYlGn",
                        norm=norm_c, linewidth=2.5, zorder=5)
    lc.set_array((R_clamp[:-1] + R_clamp[1:]) / 2)
    ax2d.add_collection(lc)
    plt.colorbar(lc, ax=ax2d, orientation="vertical", fraction=0.03, pad=0.02,
                 label="Radius of curvature (m)")

    # IS 5968 radius label
    ax2d.plot([], [], color="#ff4040", lw=2, ls="--",
              label=f"R_IS5968 min = {R_IS:.0f} m")

    # dimension annotation at mid-reach
    ax2d.annotate("", xy=(cx[mid]+Tw/2*1.0, cy[mid]),
                  xytext=(cx[mid]-Tw/2*1.0, cy[mid]),
                  arrowprops=dict(arrowstyle="<->", color="#f0c040", lw=1.5))
    ax2d.text(cx[mid], cy[mid] - 0.8,
              f"Top width = {Tw:.1f} m", color="#f0c040",
              fontsize=7, ha="center")

    ax2d.set_xlim(bounds.left - 1, bounds.right + 1)
    ax2d.set_ylim(bounds.bottom - 1, bounds.top + 1)
    ax2d.set_title("B — Plan View: Ortho + Wet Mask + Canal Footprint",
                   color="#e0e0e0", fontsize=9)
    ax2d.set_xlabel("X Lambert-93 (m)", color="#a0a0a0", fontsize=8)
    ax2d.set_ylabel("Y Lambert-93 (m)", color="#a0a0a0", fontsize=8)
    ax2d.tick_params(colors="#a0a0a0", labelsize=7)
    ax2d.legend(fontsize=7, facecolor="#1a1a2a", edgecolor="#3a3a4a",
                labelcolor="#e0e0e0", loc="upper right")
    for sp in ax2d.spines.values():
        sp.set_color("#3a3a4a")

    # ── Panel C: Curvature radius profile ────────────────────────────
    axK = fig.add_subplot(gs[0, 2])
    axK.set_facecolor("#161b22")

    dist_cl = dist_curv

    R_plot = np.where(np.isfinite(R_curve), R_curve, R_IS * 8)
    R_plot = np.clip(R_plot, 0, R_IS * 6)

    axK.fill_between(dist_cl, 0, R_plot,
                     color="#1a6fa8", alpha=0.4, label="Radius of curvature")
    axK.plot(dist_cl, R_plot, color="#56c8ff", lw=1.5)
    axK.axhline(R_IS, color="#ff4040", lw=2, ls="--",
                label=f"IS 5968 R_min = {R_IS:.0f} m  (Q={Q:.0f} m³/s)")

    # annotate the tightest bend
    tight_idx = np.argmin(R_curve[np.isfinite(R_curve)]) \
        if np.any(np.isfinite(R_curve)) else 0
    axK.annotate(f"R_min≈{R_min:.0f} m",
                 xy=(dist_cl[tight_idx], R_plot[tight_idx]),
                 xytext=(dist_cl[tight_idx] + dist_cl[-1]*0.05,
                         R_plot[tight_idx] + R_IS * 0.4),
                 color="#ffaa40", fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color="#ffaa40", lw=1.2))

    ok_colour = "#40c040" if R_min >= R_IS else "#ff4040"
    ok_text   = f"{'PASS' if R_min >= R_IS else 'FAIL'}: R_min={R_min:.0f} m"
    axK.text(0.98, 0.97, ok_text, transform=axK.transAxes,
             ha="right", va="top", color=ok_colour, fontsize=8,
             fontweight="bold")

    axK.set_xlabel("Distance along centreline (m)", color="#a0a0a0", fontsize=9)
    axK.set_ylabel("Radius of curvature (m)", color="#a0a0a0", fontsize=9)
    axK.set_title("C — Centreline Curvature vs IS 5968 Min Radius",
                  color="#e0e0e0", fontsize=9)
    axK.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor="#e0e0e0")
    axK.tick_params(colors="#a0a0a0", labelsize=8)
    for sp in axK.spines.values():
        sp.set_color("#3a3a4a")

    # ── Panel D: Longitudinal profile ─────────────────────────────────
    axL = fig.add_subplot(gs[1, 0:2])
    axL.set_facecolor("#161b22")
    z_valid = np.isfinite(z_prof)
    axL.fill_between(dist_m[z_valid], np.nanmin(z_prof) - 1, z_prof[z_valid],
                     color="#5a5a3a", alpha=0.6, label="Terrain (Z_bed)")
    axL.plot(dist_m[z_valid], z_prof[z_valid],
             color="#c0a040", lw=1.5)

    # flood surface
    z_flood_line = z_prof + reach_geo.get("h_mean", 1.152)
    axL.fill_between(dist_m[z_valid], z_prof[z_valid], z_flood_line[z_valid],
                     color="#1a6fa8", alpha=0.5, label=f"Flood h_mean={reach_geo.get('h_mean',1.15):.2f} m")
    axL.plot(dist_m[z_valid], z_flood_line[z_valid], color="#56c8ff", lw=1.2)

    # canal invert line (design slope from upstream end)
    z_up  = z_prof[z_valid][0]
    z_inv = z_up - S_design * dist_m[z_valid]
    axL.plot(dist_m[z_valid], z_inv, color="#f0c040", lw=2,
             ls="--", label=f"Canal invert  S=1:{int(1/max(S_design,1e-6))}")
    axL.fill_between(dist_m[z_valid], z_inv - D, z_inv,
                     color="#1a6fa8", alpha=0.35)

    # water depth indicator at mid-reach
    mid_d = len(dist_m) // 2
    axL.annotate("", xy=(dist_m[mid_d], z_prof[mid_d]),
                 xytext=(dist_m[mid_d], z_inv[mid_d]),
                 arrowprops=dict(arrowstyle="<->", color="#a0d0ff", lw=1.3))
    axL.text(dist_m[mid_d] + 0.3, (z_prof[mid_d] + z_inv[mid_d])/2,
             f"D={D:.2f} m", color="#a0d0ff", fontsize=7.5)

    axL.set_xlabel("Distance along centreline (m)", color="#a0a0a0", fontsize=9)
    axL.set_ylabel("Elevation (m a.s.l.)", color="#a0a0a0", fontsize=9)
    axL.set_title(f"D — Longitudinal Profile: Terrain · Flood · Canal Invert   "
                  f"(S_meas=1:{int(1/max(S_long_meas,1e-9))}  S_design=1:{int(1/max(S_design,1e-9))})",
                  color="#e0e0e0", fontsize=9)
    axL.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor="#e0e0e0")
    axL.tick_params(colors="#a0a0a0", labelsize=8)
    for sp in axL.spines.values():
        sp.set_color("#3a3a4a")

    # ── Panel E: Mid-reach cross-section ──────────────────────────────
    axX = fig.add_subplot(gs[1, 2])
    axX.set_facecolor("#161b22")

    # terrain cross-section at mid-Y row
    mid_row = zb_filled.shape[0] // 2
    x_row   = X[mid_row, :]
    z_row   = zb_filled[mid_row, :]
    h_row   = h[mid_row, :]
    valid_x = np.isfinite(z_row)

    axX.fill_between(x_row[valid_x] - x_row[valid_x].mean(),
                     np.nanmin(z_row) - 0.5,
                     z_row[valid_x],
                     color="#5a5a3a", alpha=0.7, label="Terrain bed")
    axX.plot(x_row[valid_x] - x_row[valid_x].mean(), z_row[valid_x],
             color="#c0a040", lw=1.5)

    # existing flood surface
    z_flood_row = np.where(h_row > 0.05, z_row + h_row, np.nan)
    axX.fill_between(x_row[valid_x] - x_row[valid_x].mean(),
                     z_row[valid_x],
                     np.where(h_row[valid_x] > 0.05,
                               z_flood_row[valid_x], z_row[valid_x]),
                     color="#1a6fa8", alpha=0.5, label="Flood water")

    # designed canal section (placed at median bed elevation)
    z_bed_mid = float(np.nanmedian(z_row[valid_x]))
    sec       = _canal_section_xy(B, D, m, fb, z_invert=z_bed_mid)
    sec_fill_w = sec[:, 0]
    sec_fill_z = sec[:, 1]
    axX.fill(sec_fill_w, sec_fill_z, color="#1a6fa8", alpha=0.5,
             label="Designed canal (water)")
    axX.fill(np.concatenate([sec_fill_w, [sec_fill_w[-1], sec_fill_w[0]]]),
             np.concatenate([sec_fill_z,
                              [z_bed_mid - 0.5, z_bed_mid - 0.5]]),
             color="#5a5a6a", alpha=0.6, label="Concrete lining")
    axX.plot(sec_fill_w, sec_fill_z, color="#f0c040", lw=2.5)

    # dimension arrows
    axX.annotate("", xy=(B/2, z_bed_mid), xytext=(-B/2, z_bed_mid),
                 arrowprops=dict(arrowstyle="<->", color="#f0c040", lw=1.5))
    axX.text(0, z_bed_mid - 0.25, f"B={B:.2f} m", color="#f0c040",
             fontsize=8, ha="center")
    axX.annotate("", xy=(B/2 + m*D + 0.3, z_bed_mid + D),
                 xytext=(B/2 + m*D + 0.3, z_bed_mid),
                 arrowprops=dict(arrowstyle="<->", color="#56c8ff", lw=1.5))
    axX.text(B/2 + m*D + 0.7, z_bed_mid + D/2,
             f"D={D:.2f} m", color="#56c8ff", fontsize=8)

    # existing mean width indicator
    mean_w = reach_geo["mean_width_m"]
    axX.annotate("", xy=(mean_w/2, z_bed_mid + D + fb + 0.3),
                 xytext=(-mean_w/2, z_bed_mid + D + fb + 0.3),
                 arrowprops=dict(arrowstyle="<->", color="#56c8ff", lw=1.2,
                                 linestyle="dashed"))
    axX.text(0, z_bed_mid + D + fb + 0.5,
             f"Existing flood width ≈ {mean_w:.1f} m",
             color="#56c8ff", fontsize=7.5, ha="center", style="italic")

    axX.set_xlim(-mean_w/2 - 1, mean_w/2 + 3)
    axX.set_xlabel("Width from centreline (m)", color="#a0a0a0", fontsize=9)
    axX.set_ylabel("Elevation (m a.s.l.)", color="#a0a0a0", fontsize=9)
    axX.set_title("E — Mid-reach Cross-Section: Terrain · Flood · Canal Design",
                  color="#e0e0e0", fontsize=9)
    axX.legend(fontsize=7.5, facecolor="#1a1a2a", edgecolor="#3a3a4a",
               labelcolor="#e0e0e0", loc="upper right")
    axX.tick_params(colors="#a0a0a0", labelsize=8)
    for sp in axX.spines.values():
        sp.set_color("#3a3a4a")

    # ── title ─────────────────────────────────────────────────────────
    curv_status = f"R_min={R_min:.0f} m {'≥' if R_min>=R_IS else '<'} IS5968 {R_IS:.0f} m"
    fig.suptitle(
        f"Canal Design on Geo-referenced Terrain  |  "
        f"Q={Q:.0f} m³/s  B={B:.2f} m  D={D:.2f} m  V={V:.2f} m/s  "
        f"Reach≈{reach_geo['reach_length_m']:.0f} m  |  {curv_status}",
        color="#e0e0e0", fontsize=11, y=1.005)

    plt.tight_layout(rect=[0, 0, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Canal 3D overlay → {out_path}")


def generate_canal_cad_figure(
    canal_params: dict,
    reach_geo:    dict,
    out_path:     str   = "assets/canal_cad_model.png",
    display_len:  float = None,   # metres of canal to render; None = auto
):
    """
    4-view engineering drawing: Front · Side · Top · Isometric, each with
    dimension lines.  Uses local (canal-axis) coordinates so all views are
    self-consistent regardless of the Lambert-93 centreline orientation.

    Layout
    ------
      Row 0:  [ FRONT VIEW (cross-section) ]  [ SIDE VIEW (longitudinal) ]
      Row 1:  [  TOP VIEW  (plan)          ]  [   ISOMETRIC (3-D)        ]
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as mgridspec
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ── canal parameters ─────────────────────────────────────────────
    B   = canal_params["bed_width_m"]
    D   = canal_params["water_depth_m"]
    m   = canal_params["side_slope"]
    fb  = canal_params["freeboard_m"]
    Q   = canal_params.get("Q_calculated_m3s", 50.0)
    V   = canal_params.get("velocity_ms", 1.3)
    n_m = canal_params.get("manning_n", 0.018)
    S   = canal_params.get("long_slope", 1.0 / 5000)
    Tw  = canal_params.get("top_width_m", B + 2 * m * (D + fb))
    H   = D + fb           # total lining height
    L   = reach_geo["reach_length_m"]

    # display length: show a ~3-canal-width section, min 10 m
    if display_len is None:
        display_len = float(np.clip(3.0 * Tw, 10.0, 80.0))

    # vertical exaggeration for side view (make slope visible)
    drop = S * display_len
    # target drop to occupy ~25 % of panel height
    VE = max(1, int(math.ceil(0.25 * H / drop))) if drop > 1e-9 else 1
    VE = min(VE, 200)   # cap to avoid absurd labels

    # ── canonical section vertices (local: x=width, z=elev) ─────────
    # Outer profile (6 pts, closed)
    sx = np.array([-(B/2 + m*H), -(B/2 + m*D), -B/2,
                    B/2,           B/2 + m*D,    B/2 + m*H])
    sz = np.array([H, D, 0, 0, D, H])
    # Water body outline (4 pts)
    wx = np.array([-(B/2 + m*D), -B/2, B/2, B/2 + m*D])
    wz = np.array([D, 0, 0, D])

    # ── colour palette ────────────────────────────────────────────────
    TC     = "#e8e8e8"
    DIM    = "#a0a0a0"
    C_WALL = "#1a3a5a"
    C_WAT  = "#1a6080"
    C_BED  = "#2a2a3a"
    C_LINE = "#5090d0"
    C_B    = "#e0c040"    # bed-width dim
    C_D    = "#e0a030"    # depth dim
    C_FB   = "#c0c040"    # freeboard dim
    C_TW   = "#a0e0a0"    # top-width dim
    C_L    = "#d060c0"    # length dim
    C_S    = "#60c0d0"    # slope dim
    C_CL   = "#e0c040"    # centreline
    BG     = "#0d1117"

    # ── dimension-line helpers ────────────────────────────────────────
    def _dim_h(ax, x1, x2, y_base, y_off, label, col, fs=7.5, va="bottom"):
        """Horizontal dimension with extension lines."""
        ey = y_base + y_off * 0.6
        dy = y_base + y_off
        ax.plot([x1, x1], [y_base, ey], color=col, lw=0.7, ls="--", alpha=0.6)
        ax.plot([x2, x2], [y_base, ey], color=col, lw=0.7, ls="--", alpha=0.6)
        ax.annotate("", xy=(x2, dy), xytext=(x1, dy),
                    arrowprops=dict(arrowstyle="<->", color=col,
                                    lw=1.1, mutation_scale=8))
        yo = dy + abs(y_off) * 0.18 if va == "bottom" else dy - abs(y_off) * 0.18
        ax.text((x1 + x2) / 2, yo, label, color=col, fontsize=fs,
                ha="center", va=va)

    def _dim_v(ax, y1, y2, x_base, x_off, label, col, fs=7.5):
        """Vertical dimension with extension lines."""
        ex = x_base + x_off * 0.6
        dx = x_base + x_off
        ax.plot([x_base, ex], [y1, y1], color=col, lw=0.7, ls="--", alpha=0.6)
        ax.plot([x_base, ex], [y2, y2], color=col, lw=0.7, ls="--", alpha=0.6)
        ax.annotate("", xy=(dx, y2), xytext=(dx, y1),
                    arrowprops=dict(arrowstyle="<->", color=col,
                                    lw=1.1, mutation_scale=8))
        ax.text(dx + abs(x_off) * 0.18, (y1 + y2) / 2, label, color=col,
                fontsize=fs, va="center", ha="left")

    def _spine_style(ax):
        for sp in ax.spines.values():
            sp.set_color("#303040")
        ax.tick_params(colors="#808080", labelsize=6)
        ax.set_facecolor(BG)

    # ── figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)
    gs = mgridspec.GridSpec(
        2, 2, figure=fig,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
        hspace=0.38, wspace=0.28,
        width_ratios=[1, 2.2],
        height_ratios=[1, 1],
    )

    # ════════════════════════════════════════════════════════════════
    # PANEL A — FRONT VIEW  (cross-section, looking upstream: X vs Z)
    # ════════════════════════════════════════════════════════════════
    axF = fig.add_subplot(gs[0, 0])
    _spine_style(axF)

    # fill concrete lining (polygon)
    poly_x = list(sx) + [sx[0]]
    poly_z = list(sz) + [sz[0]]
    axF.fill(poly_x, poly_z, color=C_WALL, alpha=0.85, zorder=2)
    # fill water body
    axF.fill(list(wx) + [wx[0]], list(wz) + [wz[0]],
             color=C_WAT, alpha=0.7, zorder=3)
    # water surface line
    axF.axhline(D, color="#30b0e0", lw=1.2, ls="--", alpha=0.85, zorder=4)
    # section outline
    axF.plot(poly_x, poly_z, color=C_LINE, lw=2.0, zorder=5)
    # centreline tick
    axF.axvline(0, color=C_CL, lw=0.8, ls=":", alpha=0.5, zorder=4)

    # ── dimension lines ──────────────────────────────────────────────
    pad_x = 0.12 * Tw;   pad_z = 0.12 * H
    # Bed width B
    _dim_h(axF, -B/2, B/2, 0.0, -(pad_z * 1.0), f"B = {B:.2f} m", C_B)
    # Top width Tw
    _dim_h(axF, -Tw/2, Tw/2, H, pad_z * 1.0, f"Tw = {Tw:.2f} m", C_TW)
    # Water depth D
    _dim_v(axF, 0.0, D, Tw/2, pad_x * 1.1, f"D = {D:.2f} m", C_D)
    # Freeboard fb
    _dim_v(axF, D, H,  Tw/2, pad_x * 1.1, f"fb = {fb:.2f} m", C_FB)
    # Side slope label
    axF.text(-(B/2 + m*D/2) - 0.03*(Tw/2), D/2,
             f"1 : {m:.1f}", color=TC, fontsize=7.5,
             ha="right", va="center", style="italic")
    axF.text( (B/2 + m*D/2) + 0.03*(Tw/2), D/2,
             f"1 : {m:.1f}", color=TC, fontsize=7.5,
             ha="left", va="center", style="italic")

    axF.set_xlim(-Tw/2 - pad_x * 2.5, Tw/2 + pad_x * 3.5)
    axF.set_ylim(-pad_z * 2.2, H + pad_z * 2.2)
    axF.set_aspect("equal", adjustable="box")
    axF.set_xlabel("Width  (m)", color=DIM, fontsize=7)
    axF.set_ylabel("Elevation  (m)", color=DIM, fontsize=7)
    axF.set_title("A — FRONT VIEW  (cross-section)", color=TC, fontsize=9,
                  pad=5)

    # ════════════════════════════════════════════════════════════════
    # PANEL B — SIDE VIEW  (longitudinal, looking from bank: Y vs Z)
    # ════════════════════════════════════════════════════════════════
    axS = fig.add_subplot(gs[0, 1])
    _spine_style(axS)

    # y from 0 → display_len; z_bed drops with slope (scaled by VE)
    ys  = np.array([0.0, display_len])
    zb  = np.array([0.0, -S * display_len]) * VE      # bed (VE applied)
    zt  = zb + H * VE                                   # top of lining

    # lining: bottom edge, top edge, filled rectangle (sloped)
    side_y = [0, display_len, display_len, 0]
    side_z = [zb[0], zb[1], zt[1], zt[0]]
    axS.fill(side_y, side_z, color=C_WALL, alpha=0.8, zorder=2)

    # water body: from z_bed to z_bed + D*VE
    water_side_y = [0, display_len, display_len, 0]
    water_side_z = [zb[0], zb[1], zb[1] + D*VE, zb[0] + D*VE]
    axS.fill(water_side_y, water_side_z, color=C_WAT, alpha=0.7, zorder=3)

    # outline
    axS.plot(side_y + [side_y[0]], side_z + [side_z[0]],
             color=C_LINE, lw=1.8, zorder=5)
    # water surface dashed line
    axS.plot([0, display_len], [zb[0] + D*VE, zb[1] + D*VE],
             color="#30b0e0", lw=1.2, ls="--", alpha=0.9, zorder=4)

    pL = 0.06 * display_len
    pH = 0.12 * H * VE
    # Length dimension
    _dim_h(axS, 0, display_len, zb[0], -(pH * 1.0), f"L = {display_len:.0f} m", C_L)
    # Total height dimension at upstream end
    _dim_v(axS, zb[0], zt[0], 0, -pL * 1.1,
           f"H = {H:.2f} m" + (f" (×{VE} vert)" if VE > 1 else ""), TC)
    # Slope annotation
    mid_y = display_len / 2
    mid_zb = (zb[0] + zb[1]) / 2
    slope_str = f"S = 1:{int(round(1/max(S,1e-9)))}"
    if VE > 1:
        slope_str += f"  (VE {VE}×)"
    axS.text(mid_y, mid_zb - pH * 0.5, slope_str,
             color=C_S, fontsize=8, ha="center", va="top")
    # slope arrow along bed
    axS.annotate("", xy=(display_len * 0.8, zb[0] + (zb[1]-zb[0])*0.8 - pH*0.05),
                 xytext=(display_len * 0.2, zb[0] + (zb[1]-zb[0])*0.2 - pH*0.05),
                 arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.2), zorder=6)

    axS.set_xlim(-pL * 1.5, display_len + pL * 1.0)
    axS.set_ylim(zb[1] - pH * 2.8, zt[0] + pH * 2.5)
    axS.set_xlabel("Distance along canal  (m)", color=DIM, fontsize=7)
    axS.set_ylabel(f"Elevation  (m{', VE'+str(VE)+'×' if VE>1 else ''})",
                   color=DIM, fontsize=7)
    axS.set_title("B — SIDE VIEW  (longitudinal profile)", color=TC, fontsize=9,
                  pad=5)

    # ════════════════════════════════════════════════════════════════
    # PANEL C — TOP VIEW  (plan, looking down: X vs Y)
    # ════════════════════════════════════════════════════════════════
    axT = fig.add_subplot(gs[1, 0])
    _spine_style(axT)

    # In plan: the canal rectangle is display_len × Tw
    # inner bed shown dashed (B wide)
    axT.fill([-Tw/2, Tw/2, Tw/2, -Tw/2],
             [0, 0, display_len, display_len],
             color=C_WALL, alpha=0.75, zorder=2)
    # water / bed area (water surface projection = Tw_w × L)
    Tw_w = B + 2 * m * D   # water top-width
    axT.fill([-Tw_w/2, Tw_w/2, Tw_w/2, -Tw_w/2],
             [0, 0, display_len, display_len],
             color=C_WAT, alpha=0.6, zorder=3)
    # bed outline dashed
    axT.plot([-B/2, -B/2, B/2, B/2, -B/2],
             [0, display_len, display_len, 0, 0],
             color=C_B, lw=1.0, ls="--", zorder=4, alpha=0.9)
    # outer walls
    for xv in [-Tw/2, Tw/2]:
        axT.plot([xv, xv], [0, display_len], color=C_LINE, lw=1.8, zorder=5)
    axT.plot([-Tw/2, Tw/2], [0, 0],           color=C_LINE, lw=1.8, zorder=5)
    axT.plot([-Tw/2, Tw/2], [display_len]*2,  color=C_LINE, lw=1.8, zorder=5)
    # centreline
    axT.plot([0, 0], [0, display_len], color=C_CL, lw=1.0, ls=":", zorder=4)

    pxT = 0.12 * Tw;   pyT = 0.07 * display_len
    # Top width Tw
    _dim_h(axT, -Tw/2, Tw/2, display_len, pyT * 1.2, f"Tw = {Tw:.2f} m", C_TW)
    # Bed width B (below)
    _dim_h(axT, -B/2, B/2, 0, -(pyT * 1.2), f"B = {B:.2f} m", C_B, va="top")
    # Length L
    _dim_v(axT, 0, display_len, Tw/2, pxT * 1.2,
           f"L = {display_len:.0f} m", C_L)

    axT.set_xlim(-Tw/2 - pxT * 2.5, Tw/2 + pxT * 3.5)
    axT.set_ylim(-pyT * 3.5, display_len + pyT * 3.0)
    axT.set_aspect("equal", adjustable="box")
    axT.set_xlabel("Width  (m)", color=DIM, fontsize=7)
    axT.set_ylabel("Distance along canal  (m)", color=DIM, fontsize=7)
    axT.set_title("C — TOP VIEW  (plan)", color=TC, fontsize=9, pad=5)

    # ════════════════════════════════════════════════════════════════
    # PANEL D — ISOMETRIC  (3-D extruded section, local coords)
    # ════════════════════════════════════════════════════════════════
    axI = fig.add_subplot(gs[1, 1], projection="3d")
    axI.set_facecolor(BG)
    for pane in (axI.xaxis.pane, axI.yaxis.pane, axI.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#252535")

    n_seg = 30
    ys_3d = np.linspace(0, display_len, n_seg + 1)

    for i in range(n_seg):
        y0, y1 = ys_3d[i], ys_3d[i + 1]
        z_inv0 = -S * y0;  z_inv1 = -S * y1

        # bed quad (flat bottom)
        bv = [[(x, y0, z_inv0) for x in [-B/2, B/2]] +
              [(B/2, y1, z_inv1), (-B/2, y1, z_inv1)]]
        axI.add_collection3d(Poly3DCollection(bv, facecolor=C_BED,
                             edgecolor="#404050", lw=0.3, alpha=0.9))
        # left and right walls
        for sign in (-1, 1):
            wv = [[
                (sign * B/2,          y0, z_inv0),
                (sign * (B/2 + m*H),  y0, z_inv0 + H),
                (sign * (B/2 + m*H),  y1, z_inv1 + H),
                (sign * B/2,          y1, z_inv1),
            ]]
            axI.add_collection3d(Poly3DCollection(wv, facecolor=C_WALL,
                                 edgecolor="#2a4a6a", lw=0.3, alpha=0.92))
        # water surface quad
        wv = [[
            (-(B/2 + m*D), y0, z_inv0 + D),
            ( (B/2 + m*D), y0, z_inv0 + D),
            ( (B/2 + m*D), y1, z_inv1 + D),
            (-(B/2 + m*D), y1, z_inv1 + D),
        ]]
        axI.add_collection3d(Poly3DCollection(wv, facecolor=C_WAT,
                             edgecolor=None, alpha=0.55))

    # upstream end-cap outline
    cap_x = list(sx) + [sx[0]]
    cap_z = [z + 0.0 for z in list(sz)] + [sz[0]]
    axI.plot(cap_x, [0.0] * len(cap_x), cap_z, color=C_LINE, lw=2.0, zorder=6)

    # downstream end-cap
    zd = -S * display_len
    axI.plot(cap_x, [display_len] * len(cap_x), [z + zd for z in cap_z],
             color=C_LINE, lw=1.2, ls="--", zorder=5)

    # centreline
    axI.plot([0, 0], [0, display_len], [0, -S * display_len],
             color=C_CL, lw=1.2, ls="--", zorder=7)

    # ── isometric dimension annotations (3D text) ────────────────────
    def _ann3d(ax, x1, y1, z1, x2, y2, z2, label, col):
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=col, lw=0.9, ls="--", alpha=0.7)
        ax.text((x1+x2)/2, (y1+y2)/2, (z1+z2)/2, label, color=col, fontsize=7.5,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.7))

    _ann3d(axI, -B/2, 0, 0, B/2, 0, 0, f"B = {B:.2f} m", C_B)
    _ann3d(axI,  Tw/2, display_len/2, 0, Tw/2, display_len/2, H,
           f"H = {H:.2f} m", C_D)
    _ann3d(axI,  Tw/2, display_len/2, H, -Tw/2, display_len/2, H,
           f"Tw = {Tw:.2f} m", C_TW)
    _ann3d(axI, -Tw/2, 0, H*0.5, -Tw/2, display_len, H*0.5 - S*display_len,
           f"L = {display_len:.0f} m", C_L)

    axI.set_xlabel("Width  (m)",  color=DIM, fontsize=7, labelpad=3)
    axI.set_ylabel("Length  (m)", color=DIM, fontsize=7, labelpad=3)
    axI.set_zlabel("Elev  (m)",   color=DIM, fontsize=7, labelpad=2)
    axI.tick_params(colors="#808080", labelsize=6)
    axI.view_init(elev=25, azim=-50)
    axI.set_title("D — ISOMETRIC VIEW", color=TC, fontsize=9, pad=5)

    # legend
    leg_patches = [
        mpatches.Patch(color=C_WALL, label=f"Concrete lining  m={m}:1"),
        mpatches.Patch(color=C_WAT,  label=f"Water  D={D:.2f} m"),
        mpatches.Patch(color=C_BED,  label="Canal bed"),
    ]
    axI.legend(handles=leg_patches, fontsize=7, facecolor="#1a1a2a",
               edgecolor="#3a3a4a", labelcolor=TC, loc="upper left")

    # ── title block ───────────────────────────────────────────────────
    title = (f"Canal CAD Model — IS 10430 · IS 5968  ·  "
             f"B = {B:.2f} m · D = {D:.2f} m · Tw = {Tw:.2f} m · "
             f"Q = {Q:.0f} m³/s · V = {V:.2f} m/s · n = {n_m:.3f} · "
             f"S = 1:{int(round(1/max(S,1e-9)))}")
    fig.suptitle(title, color=TC, fontsize=10, y=0.97)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Canal CAD figure → {out_path}")


if __name__ == "__main__":
    import json, sys, os
    sys.path.insert(0, ".")
    from modules.canal_optimizer import optimise_canal
    from modules.extract_reach_geometry import extract_reach
    from modules.d8_thalweg import extract_d8_thalweg

    FP  = "output/brague/flow_depth.tif"
    ZSF = "output/brague/frame_00200_z_surface.tif"

    geo = extract_reach(FP)
    geo["h_mean"] = 1.152

    # D8 thalweg for informed slope + curvature
    d8 = extract_d8_thalweg(FP, ZSF, sub=10)
    S_meas = max(d8["geometry"]["S_long"], 1e-5)
    params = optimise_canal(Q_target=50.0, S_long=S_meas)

    print(f"\nDesign with measured S={S_meas:.5f} (1:{int(1/S_meas)}):")
    print(f"  B={params['bed_width_m']:.2f} m  D={params['water_depth_m']:.2f} m  "
          f"V={params['velocity_ms']:.2f} m/s")

    generate_canal_3d_overlay(
        flow_depth_tif = FP,
        z_surface_tif  = ZSF,
        ortho_tif      = "data/brague/Ortho.tif",
        canal_params   = params,
        reach_geo      = geo,
        out_path       = "assets/canal_3d_overlay.png",
        sub            = 6,
        d8_result      = d8,
    )

    generate_canal_cad_figure(
        canal_params = params,
        reach_geo    = geo,
        out_path     = "assets/canal_cad_model.png",
    )
