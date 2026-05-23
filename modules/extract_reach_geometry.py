"""
modules/extract_reach_geometry.py
===================================
Extract canal reach geometry from the flow depth raster and MNT terrain.

Provides
--------
  extract_reach(flow_depth_tif)
      → centerline (Lambert-93 XY), reach length, width profile, S_long

  longitudinal_profile(centerline_xy, z_bed_raster, transform)
      → distance array, elevation array along centerline

The centerline is found by skeletonizing the wet mask, then ordering the
skeleton pixels from upstream (high Y) to downstream (low Y).
"""

from pathlib import Path
import math
import numpy as np


def _skeleton_order(skel_yx):
    """
    Order skeleton pixels along the longest end-to-end path using BFS.
    Handles branched skeletons by pruning branch pixels and then finding
    the pair of endpoints with maximum geodesic distance.
    Returns ordered (row, col) arrays.
    """
    ys, xs = skel_yx
    if len(ys) == 0:
        return np.array([]), np.array([])

    pts = set(zip(ys.tolist(), xs.tolist()))

    def neighbours(r, c):
        return [(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r+dr, c+dc) in pts]

    # ── BFS from a seed to find the farthest reachable pixel ──────────
    def bfs_farthest(seed):
        from collections import deque
        visited = {seed: None}  # node → parent
        queue   = deque([seed])
        last    = seed
        while queue:
            node = queue.popleft()
            last = node
            for nb in neighbours(*node):
                if nb not in visited:
                    visited[nb] = node
                    queue.append(nb)
        return last, visited

    # Double-BFS: find true endpoints (start → farthest end → farthest end)
    seed      = min(pts, key=lambda p: p[0])   # top-most pixel
    endA, _   = bfs_farthest(seed)
    endB, par = bfs_farthest(endA)

    # Trace back the path from endB to endA
    path, node = [], endB
    while node is not None:
        path.append(node)
        node = par[node]
    path.reverse()

    arr = np.array(path)
    return arr[:, 0], arr[:, 1]


def extract_reach(flow_depth_tif: str, threshold: float = 0.10):
    """
    Parameters
    ----------
    flow_depth_tif : str
        Path to flow_depth.tif produced by the depth pipeline.
    threshold : float
        Minimum flow depth (m) to be considered wet.

    Returns
    -------
    dict with keys:
        centerline_x, centerline_y   – Lambert-93 coords of ordered centerline
        reach_length_m               – total arc length of centerline (m)
        width_profile_m              – perpendicular wet-width at each CL point
        mean_width_m
        max_width_m
        S_long                       – estimated longitudinal slope from Z span
        pixel_res_m
        bounds                       – rasterio BoundingBox
        wet_area_m2
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")
    import rasterio

    with rasterio.open(flow_depth_tif) as src:
        h   = src.read(1).astype(np.float32)
        t   = src.transform
        bounds = src.bounds
        res = abs(float(t.a))

    from scipy.ndimage import binary_fill_holes, binary_closing, label
    mask_raw = (h > threshold)

    # Small closing (5-px radius) to bridge pixel-scale gaps in the depth map,
    # then fill enclosed holes. Keep the closing physically small (< 5 % of
    # channel width) so the gross shape is preserved.
    disk_r = 5   # pixels ≈ 5 × pixel_res_m = ~45 mm at 9 mm/px
    struct = np.ones((disk_r * 2 + 1, disk_r * 2 + 1), dtype=bool)
    mask = binary_fill_holes(binary_closing(mask_raw, structure=struct))

    # ── D8-based centreline via distance-transform ridge ─────────────
    # For each row compute the centroid column of the wet pixels (weighted by
    # local distance-transform value = distance from dry boundary).
    # This follows the "hydraulic thalweg" — the deepest / widest path —
    # which is equivalent to a D8 flow-accumulation ridge in 2-D.
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)   # distance to nearest dry pixel (px)

    # Row-wise centroid weighted by dist² (emphasises channel centre)
    valid_rows = np.where(mask.any(axis=1))[0]
    cl_rows = np.array([], dtype=int)
    cl_cols = np.array([], dtype=int)
    for r in valid_rows:
        w = dist[r, :] ** 2
        total = w.sum()
        if total > 0:
            col_c = int(round((np.arange(h.shape[1]) * w).sum() / total))
            cl_rows = np.append(cl_rows, r)
            cl_cols = np.append(cl_cols, col_c)

    if len(cl_rows) == 0:
        raise ValueError("No valid wet rows found — check threshold or input")

    row_ord, col_ord = cl_rows, cl_cols

    if len(row_ord) == 0:
        raise ValueError("Skeleton is empty — check threshold or input raster")

    # Convert to Lambert-93
    cx = float(t.c) + (col_ord + 0.5) * float(t.a)
    cy = float(t.f) + (row_ord + 0.5) * float(t.e)   # t.e is negative

    # Arc length along centreline
    dx  = np.diff(cx)
    dy  = np.diff(cy)
    seg = np.sqrt(dx**2 + dy**2)
    reach_length = float(seg.sum())

    # Width at each centreline row (simple perpendicular estimate via row widths)
    row_widths = mask.sum(axis=1) * res        # metres, per image row
    # sample width at each skeleton row
    width_profile = row_widths[row_ord].astype(float)

    # Longitudinal slope: linear regression on row index vs mean Z_bed
    # (estimated as the slope implied by the vertical extent of the wet region
    # in pixel space — rough; a proper slope uses Z_bed along centreline)
    wet_rows = np.where(mask.any(axis=1))[0]
    # approximate: use pixel rows as proxy for distance
    span_m = (wet_rows[-1] - wet_rows[0]) * res if len(wet_rows) > 1 else reach_length

    # Smooth the centreline to remove pixel-quantization kinks before curvature
    from scipy.ndimage import gaussian_filter1d
    sigma_px = max(10, int(1.0 / res))   # 1-m Gaussian smoothing
    cx = gaussian_filter1d(cx.astype(float), sigma=sigma_px)
    cy = gaussian_filter1d(cy.astype(float), sigma=sigma_px)

    curv = curvature_profile(cx, cy)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_arr = np.where(curv > 1e-9, 1.0 / curv, np.inf)
    finite_R = R_arr[np.isfinite(R_arr)]
    min_R   = float(finite_R.min())  if finite_R.size else float("inf")
    mean_R  = float(finite_R.mean()) if finite_R.size else float("inf")

    return {
        "centerline_x":       cx,
        "centerline_y":       cy,
        "centerline_row":     row_ord,
        "centerline_col":     col_ord,
        "reach_length_m":     reach_length,
        "width_profile_m":    width_profile,
        "mean_width_m":       float(width_profile[width_profile > 0].mean()),
        "max_width_m":        float(width_profile.max()),
        "pixel_res_m":        res,
        "bounds":             bounds,
        "wet_area_m2":        float(mask.sum() * res**2),
        "S_long_approx":      None,    # filled in by longitudinal_profile()
        "curvature_1_per_m":  curv,
        "radius_m":           R_arr,
        "min_radius_m":       min_R,
        "mean_radius_m":      mean_R,
    }


def curvature_profile(centerline_x: np.ndarray,
                      centerline_y: np.ndarray) -> np.ndarray:
    """
    Compute the absolute curvature κ (1/m) at each centreline point using
    second-order finite differences.

    κ = |x' y'' − y' x''| / (x'^2 + y'^2)^(3/2)

    Returns array of length N (κ=0 at endpoints).
    """
    n = len(centerline_x)
    kappa = np.zeros(n)
    if n < 3:
        return kappa

    # first derivatives (central differences, forward/backward at ends)
    dx = np.gradient(centerline_x.astype(float))
    dy = np.gradient(centerline_y.astype(float))
    # second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-12,
                         np.abs(dx * d2y - dy * d2x) / denom,
                         0.0)
    return kappa


def longitudinal_profile(centerline_x, centerline_y,
                          z_bed: np.ndarray, transform) -> tuple:
    """
    Sample Z_bed elevation along the centerline and compute longitudinal profile.

    Parameters
    ----------
    centerline_x, centerline_y : 1-D arrays  Lambert-93
    z_bed : 2-D array  aligned with `transform`
    transform : rasterio.Affine

    Returns
    -------
    dist_m   : cumulative distance along centreline (m)
    z_profile: bed elevation at each centreline point (m)
    S_long   : best-fit longitudinal slope (rise/run, positive = descending)
    """
    nrows, ncols = z_bed.shape
    a, e = float(transform.a), float(transform.e)
    c, f = float(transform.c), float(transform.f)

    cols = np.clip(((centerline_x - c) / a).astype(int), 0, ncols - 1)
    rows = np.clip(((centerline_y - f) / e).astype(int), 0, nrows - 1)
    z_profile = z_bed[rows, cols].astype(float)

    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    dist_m = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])

    # least-squares slope
    valid = np.isfinite(z_profile) & (z_profile > 0)
    if valid.sum() > 2:
        coeffs = np.polyfit(dist_m[valid], z_profile[valid], 1)
        S_long = float(-coeffs[0])   # negative because elevation decreases downstream
    else:
        S_long = 0.0002              # fallback

    return dist_m, z_profile, S_long


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    geo = extract_reach("output/brague/flow_depth.tif")
    print(f"Reach length    : {geo['reach_length_m']:.1f} m")
    print(f"Mean width      : {geo['mean_width_m']:.1f} m")
    print(f"Max width       : {geo['max_width_m']:.1f} m")
    print(f"Wet area        : {geo['wet_area_m2']:.0f} m²")
    print(f"Centreline pts  : {len(geo['centerline_x'])}")
    print(f"Min curve radius: {geo['min_radius_m']:.1f} m")
    print(f"Mean curve radius: {geo['mean_radius_m']:.1f} m")

    # Show slope from Z_bed if z_surface available
    import rasterio
    import os
    z_surf_path = "output/brague/frame_00200_z_surface.tif"
    if os.path.exists(z_surf_path):
        with rasterio.open("output/brague/flow_depth.tif") as src:
            h_arr = src.read(1).astype("float32")
        with rasterio.open(z_surf_path) as src2:
            zs_arr = src2.read(1).astype("float32")
            tf     = src2.transform
        zb_arr = zs_arr - h_arr
        zb_arr[h_arr < 0.1] = float("nan")
        dist_m, z_prof, S_long = longitudinal_profile(
            geo["centerline_x"], geo["centerline_y"], zb_arr, tf)
        print(f"Measured S_long : {S_long:.6f}  (1:{int(1/max(S_long,1e-9))})")
        print(f"Elevation span  : {z_prof[0]:.2f} → {z_prof[-1]:.2f} m")
