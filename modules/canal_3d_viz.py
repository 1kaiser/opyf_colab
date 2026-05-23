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
