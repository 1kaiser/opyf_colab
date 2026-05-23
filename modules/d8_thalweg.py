"""
modules/d8_thalweg.py
=====================
JAX-accelerated D8 flow routing and thalweg extraction.

Algorithm
---------
  1. fill_sinks(z, wet)         — iterative Planchon-Darboux sink filling (JAX scan)
  2. d8_flow_direction(z, wet)  — steepest-descent D8 direction per cell (JAX vmap)
  3. flow_accumulation(fdir, wet, n) — upstream area via repeated scatter (JAX scan)
  4. trace_thalweg(acc, fdir, wet) — follow max-acc inflow from outlet → source (NumPy)
  5. thalweg_geometry(rows, cols, z_bed, transform, res)
        → slope, bearing, curvature, reach length, IS 5968 check

Public API
----------
  extract_d8_thalweg(flow_depth_tif, z_surface_tif, sub=10) → dict

D8 direction encoding (0–7, counter-clockwise from East):
  0=E   1=SE  2=S   3=SW
  4=W   5=NW  6=N   7=NE
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

# ── D8 direction table ─────────────────────────────────────────────────────────
#   direction index:   0   1   2   3   4   5   6   7
_DR   = np.array(  [  0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int32)
_DC   = np.array(  [  1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int32)
_DIST = np.array(  [1.0, math.sqrt(2), 1.0, math.sqrt(2),
                    1.0, math.sqrt(2), 1.0, math.sqrt(2)], dtype=np.float64)
_OPP  = np.array(  [  4,  5,  6,  7,  0,  1,  2,  3], dtype=np.int32)  # opposite direction

# bearing from North, clockwise (degrees) — for flow-direction visualisation
_BEARING = np.array([90., 135., 180., 225., 270., 315., 0., 45.], dtype=np.float64)

_DR_J   = jnp.array(_DR)
_DC_J   = jnp.array(_DC)
_DIST_J = jnp.array(_DIST)
_OPP_J  = jnp.array(_OPP)


# ── 1. Sink filling ───────────────────────────────────────────────────────────

def fill_sinks(z: jnp.ndarray, wet: jnp.ndarray,
               eps: float = 1e-4, n_iter: int = 200) -> jnp.ndarray:
    """
    Iterative Planchon-Darboux sink filling (JAX scan).

    Boundary anchor: wet cells that are adjacent to a dry cell (the perimeter
    of the wet mask) keep their original terrain elevation. Interior wet cells
    start at +inf and are filled down to the minimum drainage path.

    Parameters
    ----------
    z      : (H, W) float64 JAX array — Z_bed values (dry cells = large sentinel)
    wet    : (H, W) bool JAX array
    eps    : float — tiny increment added each step to drain flat / pit regions
    n_iter : int   — number of scan iterations (use ≥ max(H,W) for convergence)

    Returns
    -------
    z_filled : (H, W) float64 — pit-free Z_bed in wet cells; dry = large value
    """
    z_max_wet = jnp.nanmax(jnp.where(wet, z, jnp.nan))
    barrier   = z_max_wet + 1000.0          # large value for dry cells
    z_floor   = jnp.where(wet, z, barrier)  # terrain floor (can't go below)

    # Wet-mask perimeter: wet cells with at least one dry 4-connected neighbour
    def _has_dry_nb(w):
        dry = ~w
        return (jnp.roll(dry, 1, 0) | jnp.roll(dry, -1, 0) |
                jnp.roll(dry, 1, 1) | jnp.roll(dry, -1, 1))

    perimeter = wet & _has_dry_nb(wet)

    # Initialise: perimeter cells keep Z, interior wet cells start at z_max+1
    w_init = jnp.where(wet,
                       jnp.where(perimeter, z, z_max_wet + 1.0),
                       barrier)

    def _step(w, _):
        w_min = w
        for d in range(8):
            nb = jnp.roll(jnp.roll(w, -int(_DR[d]), axis=0),
                          -int(_DC[d]), axis=1)
            w_min = jnp.minimum(w_min, nb + eps)
        # Cannot go below original terrain elevation
        return jnp.maximum(w_min, z_floor), None

    w_filled, _ = lax.scan(_step, w_init, None, length=n_iter)
    return w_filled


# ── 2. D8 flow direction ──────────────────────────────────────────────────────

@jax.jit
def d8_flow_direction(z_filled: jnp.ndarray,
                      wet: jnp.ndarray) -> jnp.ndarray:
    """
    Steepest-descent D8 flow direction.

    Returns an int8 array with values 0–7 (direction index) or -1 (flat/pit).
    Dry cells return -1.
    """
    # Compute slope to each of 8 neighbours
    slopes = []
    for d in range(8):
        nb = jnp.roll(jnp.roll(z_filled, -int(_DR[d]), axis=0),
                      -int(_DC[d]), axis=1)
        # Only slope toward wet neighbours
        nb_valid = jnp.roll(jnp.roll(wet, -int(_DR[d]), axis=0),
                            -int(_DC[d]), axis=1)
        s = jnp.where(nb_valid, (z_filled - nb) / _DIST[d], -jnp.inf)
        slopes.append(s)

    slopes_stack = jnp.stack(slopes, axis=-1)       # (H, W, 8)
    max_slope    = jnp.max(slopes_stack, axis=-1)
    best_dir     = jnp.argmax(slopes_stack, axis=-1).astype(jnp.int8)

    # -1 for flat cells (no positive slope) or dry cells
    fdir = jnp.where(wet & (max_slope > 0), best_dir, jnp.int8(-1))
    return fdir


# ── 3. Flow accumulation ──────────────────────────────────────────────────────

def flow_accumulation(flow_dir: jnp.ndarray,
                      wet: jnp.ndarray,
                      n_iter: int = 600) -> jnp.ndarray:
    """
    Upstream contributing area (cell count) via repeated scatter.

    Each scan step propagates accumulation one cell downstream.
    After n_iter steps, cells up to n_iter hops from the outlet are correct.
    Use n_iter ≥ max(H, W) for full convergence.

    Returns float64 array of accumulation values (≥ 1 for wet cells).
    """
    acc_init = jnp.where(wet, 1.0, 0.0)

    def _step(acc, _):
        incoming = jnp.zeros_like(acc)
        for d in range(8):
            # Cells with flow_dir == d contribute their acc to the cell at +dr,+dc
            contributes = jnp.where(flow_dir == d, acc, 0.0)
            # Shift contribution to destination cell
            shifted = jnp.roll(jnp.roll(contributes,
                                        int(_DR[d]), axis=0),
                                int(_DC[d]), axis=1)
            incoming = incoming + shifted
        # Each wet cell contributes 1 to itself (base area)
        new_acc = jnp.where(wet, incoming + 1.0, 0.0)
        return new_acc, None

    acc_final, _ = lax.scan(_step, acc_init, None, length=n_iter)
    return acc_final


# ── 4. Thalweg tracing (NumPy — sequential by nature) ────────────────────────

def trace_thalweg(acc_np: np.ndarray,
                  flow_dir_np: np.ndarray,
                  wet_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace the thalweg from the outlet (highest-accumulation boundary cell)
    upstream to the source by always following the highest-accumulation
    upstream neighbour that drains into the current cell.

    Returns
    -------
    rows, cols : ordered 1-D int arrays from outlet to source
    """
    H, W = acc_np.shape

    # Outlet: wet cell with the global maximum flow accumulation.
    # This is the true convergence point of the D8 flow network — all upstream
    # paths drain here, so it naturally falls at the downstream end of the reach.
    cand_acc = np.where(wet_np, acc_np, -1.0)
    outlet_flat = int(np.argmax(cand_acc))
    outlet_r, outlet_c = divmod(outlet_flat, W)

    path_r = [outlet_r]
    path_c = [outlet_c]
    visited = {(outlet_r, outlet_c)}

    current_r, current_c = outlet_r, outlet_c

    while True:
        best_acc  = -1
        best_r, best_c = current_r, current_c

        for d in range(8):
            nr = current_r + _DR[d]
            nc = current_c + _DC[d]
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not wet_np[nr, nc]:
                continue
            if (nr, nc) in visited:
                continue
            # This neighbour must drain INTO current cell
            nd = int(flow_dir_np[nr, nc])
            if nd < 0:
                continue
            dest_r = nr + _DR[nd]
            dest_c = nc + _DC[nd]
            if (dest_r, dest_c) == (current_r, current_c):
                a = acc_np[nr, nc]
                if a > best_acc:
                    best_acc = a
                    best_r, best_c = nr, nc

        if best_acc < 0:
            break   # no upstream neighbour found

        path_r.append(best_r)
        path_c.append(best_c)
        visited.add((best_r, best_c))
        current_r, current_c = best_r, best_c

    return np.array(path_r, dtype=np.int32), np.array(path_c, dtype=np.int32)


# ── 5. Thalweg geometry ───────────────────────────────────────────────────────

def thalweg_geometry(rows: np.ndarray,
                     cols: np.ndarray,
                     z_bed_np: np.ndarray,
                     res_m: float) -> dict:
    """
    Compute reach geometry from the ordered thalweg pixel sequence.

    Parameters
    ----------
    rows, cols  : ordered thalweg pixel indices (outlet → source)
    z_bed_np    : (H, W) float array of bed elevations
    res_m       : pixel resolution (m/px)

    Returns
    -------
    dict with keys:
        dist_m          — cumulative distance along thalweg (m)
        z_profile       — bed elevation at each thalweg pixel
        slope_local     — local slope at each point (dZ/ds, positive = descending)
        bearing_deg     — bearing from North (°) of each thalweg segment
        curvature_1m    — absolute curvature κ (1/m) at each point
        radius_m        — radius of curvature (m)  [inf where straight]
        reach_length_m  — total arc length (m)
        S_long          — best-fit longitudinal slope (polyfit)
        min_radius_m    — tightest bend radius (m)
        mean_radius_m   — mean radius of finite-radius segments (m)
    """
    if len(rows) < 3:
        raise ValueError("Thalweg has fewer than 3 pixels — check wet mask / D8 params")

    # Reverse so path runs outlet→source (downstream→upstream = ascending elevation)
    # Then flip to upstream→downstream for conventional flow convention
    rows = rows[::-1].copy()
    cols = cols[::-1].copy()

    # Pixel coords → metric (relative, for geometry only — Lambert-93 done outside)
    x_raw = cols.astype(float) * res_m
    y_raw = -rows.astype(float) * res_m   # row↓ = Y decreasing

    # Smooth D8 staircase: Gaussian σ = 2 m to remove 45° kink artefacts
    from scipy.ndimage import gaussian_filter1d
    sigma_px = max(3, int(2.0 / res_m))
    x = gaussian_filter1d(x_raw, sigma=sigma_px)
    y = gaussian_filter1d(y_raw, sigma=sigma_px)

    # Cumulative distance
    dx  = np.diff(x);   dy  = np.diff(y)
    seg = np.sqrt(dx**2 + dy**2)
    dist_m = np.concatenate([[0.0], np.cumsum(seg)])

    # Bed elevation
    z_profile = z_bed_np[rows, cols].astype(float)
    nan_mask  = ~np.isfinite(z_profile)
    if nan_mask.any():
        idx = np.arange(len(z_profile))
        good = idx[~nan_mask]
        z_profile[nan_mask] = np.interp(idx[nan_mask], good, z_profile[good])

    # Local slope (dZ / ds, positive = descending)
    slope_local = -np.gradient(z_profile, dist_m)

    # Bearing (° clockwise from North) of each step
    bearing = np.degrees(np.arctan2(dx, dy))  # N=0, E=90
    bearing = np.concatenate([[bearing[0]], bearing])

    # Curvature: κ = |x'y'' − y'x''| / (x'^2 + y'^2)^1.5
    xd  = np.gradient(x,  dist_m)
    yd  = np.gradient(y,  dist_m)
    xdd = np.gradient(xd, dist_m)
    ydd = np.gradient(yd, dist_m)
    denom = (xd**2 + yd**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-12, np.abs(xd*ydd - yd*xdd) / denom, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_m = np.where(kappa > 1e-9, 1.0 / kappa, np.inf)

    finite_R = radius_m[np.isfinite(radius_m)]
    min_R    = float(finite_R.min())  if finite_R.size else np.inf
    mean_R   = float(finite_R.mean()) if finite_R.size else np.inf

    # Best-fit longitudinal slope (linear regression)
    # Path runs upstream→downstream, so elevation decreases; coeffs[0] < 0 → S > 0
    valid = np.isfinite(z_profile)
    if valid.sum() > 2:
        coeffs  = np.polyfit(dist_m[valid], z_profile[valid], 1)
        S_long  = float(abs(coeffs[0]))   # always positive: drop per unit length
    else:
        S_long = 0.0

    return {
        "dist_m":          dist_m,
        "z_profile":       z_profile,
        "slope_local":     slope_local,
        "bearing_deg":     bearing,
        "curvature_1m":    kappa,
        "radius_m":        radius_m,
        "reach_length_m":  float(dist_m[-1]),
        "S_long":          S_long,
        "min_radius_m":    min_R,
        "mean_radius_m":   mean_R,
    }


# ── 6. Top-level entry point ──────────────────────────────────────────────────

def extract_d8_thalweg(flow_depth_tif: str,
                       z_surface_tif: str,
                       sub: int = 10,
                       depth_thresh: float = 0.10,
                       fill_iter: int = 200,
                       acc_iter: int = 0) -> dict:
    """
    Full D8 thalweg pipeline: load → fill → D8 → accumulate → trace → geometry.

    Parameters
    ----------
    flow_depth_tif : str   path to flow_depth.tif
    z_surface_tif  : str   path to a z_surface.tif frame
    sub            : int   spatial subsampling factor (default 10 → ~91 mm/px)
    depth_thresh   : float minimum depth to be considered wet (m)
    fill_iter      : int   sink-filling iterations
    acc_iter       : int   flow-accumulation iterations (0 = auto: max(H,W))

    Returns
    -------
    dict with keys:
        centerline_x, centerline_y  — Lambert-93 coordinates (full-res sub-pixel)
        geometry                    — output of thalweg_geometry()
        flow_dir_np                 — (H_sub, W_sub) D8 direction array
        acc_np                      — (H_sub, W_sub) accumulation array
        transform_sub               — rasterio Affine of the subsampled grid
        sub                         — subsampling factor used
    """
    import rasterio
    from rasterio.transform import from_bounds

    print(f"  [D8] Loading rasters (sub={sub}) …")
    with rasterio.open(flow_depth_tif) as src:
        h_full  = src.read(1).astype(np.float32)
        t_orig  = src.transform
        bounds  = src.bounds
    with rasterio.open(z_surface_tif) as src2:
        zs_full = src2.read(1).astype(np.float32)

    # Subsample
    h  = h_full [::sub, ::sub]
    zs = zs_full[::sub, ::sub]
    H, W  = h.shape
    res_m = abs(float(t_orig.a)) * sub

    # Z_bed: valid only where wet
    wet_np = h > depth_thresh
    zb_np  = np.where(wet_np & (zs > 0), (zs - h).astype(float), np.nan)

    if acc_iter == 0:
        acc_iter = max(H, W) + 50

    print(f"  [D8] Grid {H}×{W}  res={res_m:.3f} m/px  "
          f"wet={wet_np.sum()}px  acc_iter={acc_iter}")

    # Replace NaN with large sentinel for JAX (dry cells become barriers)
    z_sentinel = float(np.nanmax(zb_np[np.isfinite(zb_np)])) + 1000.0 \
        if np.any(np.isfinite(zb_np)) else 100.0
    zb_jax  = jnp.array(np.nan_to_num(zb_np, nan=z_sentinel))
    wet_jax = jnp.array(wet_np)

    print("  [D8] Filling sinks …")
    z_filled_jax = fill_sinks(zb_jax, wet_jax, n_iter=fill_iter)

    print("  [D8] Computing D8 flow directions …")
    fdir_jax = d8_flow_direction(z_filled_jax, wet_jax)

    print(f"  [D8] Accumulating flow ({acc_iter} iterations) …")
    acc_jax = flow_accumulation(fdir_jax, wet_jax, n_iter=acc_iter)

    acc_np   = np.array(acc_jax)
    fdir_np  = np.array(fdir_jax)
    z_sub_np = np.array(z_filled_jax)
    z_sub_np[~wet_np] = np.nan

    print("  [D8] Tracing thalweg …")
    t_rows, t_cols = trace_thalweg(acc_np, fdir_np, wet_np)
    print(f"  [D8] Thalweg: {len(t_rows)} pixels  "
          f"rows {t_rows[0]}→{t_rows[-1]}, cols {t_cols[0]}→{t_cols[-1]}")

    # Lambert-93 coordinates (pixel centre in subsampled grid)
    x0 = float(bounds.left)
    y0 = float(bounds.top)
    cl_x = x0 + (t_cols + 0.5) * res_m
    cl_y = y0 - (t_rows + 0.5) * res_m   # row↓ = Y decreasing

    geo = thalweg_geometry(t_rows, t_cols, z_sub_np, res_m)

    t_sub = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, W, H)

    return {
        "centerline_x":   cl_x,
        "centerline_y":   cl_y,
        "thalweg_rows":   t_rows,
        "thalweg_cols":   t_cols,
        "geometry":       geo,
        "flow_dir_np":    fdir_np,
        "acc_np":         acc_np,
        "z_sub_np":       z_sub_np,
        "wet_np":         wet_np,
        "transform_sub":  t_sub,
        "res_m":          res_m,
        "sub":            sub,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")

    fp = "output/brague/flow_depth.tif"
    zs = "output/brague/frame_00200_z_surface.tif"

    if not os.path.exists(fp) or not os.path.exists(zs):
        print("ERROR: output files not found — run pipeline.py first")
        sys.exit(1)

    result = extract_d8_thalweg(fp, zs, sub=10)
    geo    = result["geometry"]

    print()
    print("─" * 50)
    print(f"Reach length      : {geo['reach_length_m']:.1f} m")
    print(f"Longitudinal slope: {geo['S_long']:.6f}  (1:{int(1/max(abs(geo['S_long']),1e-9))})")
    print(f"Min curve radius  : {geo['min_radius_m']:.1f} m")
    print(f"Mean curve radius : {geo['mean_radius_m']:.1f} m")
    print(f"Thalweg pixels    : {len(result['thalweg_rows'])}")
    print()

    # IS 5968 check
    from modules.canal_optimizer import _is_min_radius
    Q_design = 50.0
    R_IS = _is_min_radius(Q_design)
    status = "PASS" if geo['min_radius_m'] >= R_IS else "WARN — realignment needed"
    print(f"IS 5968 R_min (Q={Q_design:.0f} m³/s): {R_IS:.0f} m  → {status}")

    # re-run canal design with measured slope
    from modules.canal_optimizer import optimise_canal
    S_meas = abs(geo['S_long'])
    if S_meas < 1e-5:
        S_meas = 0.0002   # fallback
    params = optimise_canal(Q_target=Q_design, S_long=S_meas)
    print()
    print("Canal design with measured S_long:")
    is_ok = params['IS_velocity_ok'] and params['IS_discharge_ok']
    print(f"  S = {S_meas:.6f}  B = {params['bed_width_m']:.2f} m  "
          f"D = {params['water_depth_m']:.2f} m  V = {params['velocity_ms']:.2f} m/s  "
          f"{'PASS' if is_ok else 'FAIL'}")
