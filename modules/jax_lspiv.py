"""
modules/jax_lspiv.py
=====================
JAX re-implementation of the opyflow LSPIV pipeline (Brague flood).

Pipeline stages
---------------
  1. Stabilisation      — iterative Lucas-Kanade tracker + homography warp
  2. Orthorectification — DLT homography → bird-eye-view (bilinear warp)
  3. PIV                — FFT phase-correlation on interrogation windows (vmap)
  4. Accumulation       — stack velocity clouds from all frame pairs
  5. Gaussian interp    — sparse → dense field (Gaussian RBF, matches opyflow VTK)
  6. Discharge          — Q = α ∫ V·h dl  along transect  (trapezoidal rule)

Reference
---------
  Rousseau G. (2019). opyflow — https://github.com/groussea/opyflow
  Vigoureux et al. SimHydro 2021.
  apply_opyf_1139_1142.py in groussea/opyflow

Coordinate system
-----------------
  All world coordinates are in LOCAL metres relative to the bridge origin
  used by opyflow:
      x0 = 1030760.6875  (Lambert-93 easting)
      y0 = 6289057.0     (Lambert-93 northing)
  i.e.  local_x = lambert93_X - x0
        local_y = lambert93_Y - y0

GCPs from opyflow (IMG_1139, downstream bridge)
-----------------------------------------------
  image_points  = [(355,429),(1338,350),(99,562),(1673,364)]   # px (col,row)
  model_points  = [(30.13,-8.28,0),(32.88,-28.08,0),
                   (20.46,-4.47,0.4),(21.32,-27.14,0.4)]       # local XYZ (m)
"""

import math
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  GEOMETRY — DLT homography + JAX bilinear warp
# ═══════════════════════════════════════════════════════════════════════════════

# Local Lambert-93 origin used throughout opyflow
ORIGIN_X = 1030760.6875
ORIGIN_Y = 6289057.0


def compute_homography_dlt(src_pts: np.ndarray,
                            dst_pts: np.ndarray) -> np.ndarray:
    """
    Direct Linear Transform (DLT) for planar homography.

    Parameters
    ----------
    src_pts : (N≥4, 2) float   source points  (e.g. pixel col, row)
    dst_pts : (N≥4, 2) float   destination    (e.g. world X, Y)

    Returns
    -------
    H : (3, 3) float  homography  dst_h = H @ src_h  (homogeneous)
    """
    assert len(src_pts) >= 4
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)

    # Normalise src
    s_mean = src.mean(0);  s_scale = np.sqrt(2) / (np.linalg.norm(src - s_mean, axis=1).mean() + 1e-9)
    Ts = np.array([[s_scale, 0, -s_scale*s_mean[0]],
                   [0, s_scale, -s_scale*s_mean[1]],
                   [0, 0, 1]])
    # Normalise dst
    d_mean = dst.mean(0);  d_scale = np.sqrt(2) / (np.linalg.norm(dst - d_mean, axis=1).mean() + 1e-9)
    Td = np.array([[d_scale, 0, -d_scale*d_mean[0]],
                   [0, d_scale, -d_scale*d_mean[1]],
                   [0, 0, 1]])

    src_n = (Ts @ np.column_stack([src, np.ones(len(src))]).T).T
    dst_n = (Td @ np.column_stack([dst, np.ones(len(dst))]).T).T

    A = []
    for (xi, yi, _), (Xi, Yi, _) in zip(src_n, dst_n):
        A.append([-xi, -yi, -1,  0,   0,  0, Xi*xi, Xi*yi, Xi])
        A.append([ 0,   0,  0, -xi, -yi, -1, Yi*xi, Yi*yi, Yi])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1].reshape(3, 3)
    # De-normalise
    H = np.linalg.inv(Td) @ h @ Ts
    return H / H[2, 2]


@functools.partial(jax.jit, static_argnums=(2,))
def warp_perspective(image: jnp.ndarray,
                     H_inv: jnp.ndarray,
                     out_shape: tuple) -> jnp.ndarray:
    """
    Apply perspective warp to `image` using the INVERSE homography H_inv.

    Parameters
    ----------
    image     : (H_in, W_in) or (H_in, W_in, C)  float32
    H_inv     : (3, 3)  inverse homography  src_coords = H_inv @ dst_h
    out_shape : (H_out, W_out)

    Returns
    -------
    warped : same ndim as image, shape out_shape (+ C if colour)
    """
    oh, ow = out_shape
    # Build output pixel grid (col=x, row=y)
    col = jnp.arange(ow, dtype=jnp.float32)
    row = jnp.arange(oh, dtype=jnp.float32)
    gc, gr = jnp.meshgrid(col, row)           # (oh, ow)
    ones  = jnp.ones_like(gc)
    pts_h = jnp.stack([gc.ravel(), gr.ravel(), ones.ravel()])   # (3, oh*ow)

    # Map to source coordinates
    src_h = H_inv @ pts_h                      # (3, oh*ow)
    src_x = src_h[0] / src_h[2]               # col in source
    src_y = src_h[1] / src_h[2]               # row in source
    coords = jnp.stack([src_y.reshape(oh, ow),
                         src_x.reshape(oh, ow)])    # (2, oh, ow) — row,col order

    if image.ndim == 2:
        return jax.scipy.ndimage.map_coordinates(image, coords, order=1,
                                                  mode="constant", cval=0.0)
    # Colour: map each channel independently
    channels = [
        jax.scipy.ndimage.map_coordinates(image[..., c], coords, order=1,
                                           mode="constant", cval=0.0)
        for c in range(image.shape[2])
    ]
    return jnp.stack(channels, axis=-1)


def build_ortho_grid(model_points: np.ndarray,
                     resolution_m: float = 0.02
                     ) -> tuple:
    """
    Build the output metric grid for the bird-eye-view image.

    Returns
    -------
    x_grid   : 1-D array  world X (local m) column centres
    y_grid   : 1-D array  world Y (local m) row centres  (top = large Y)
    out_shape: (H, W)
    """
    mp = np.array(model_points)
    x_min, x_max = mp[:, 0].min() - 1, mp[:, 0].max() + 1
    y_min, y_max = mp[:, 1].min() - 1, mp[:, 1].max() + 1
    x_grid = np.arange(x_min, x_max, resolution_m)
    y_grid = np.arange(y_max, y_min, -resolution_m)   # top-down
    return x_grid, y_grid, (len(y_grid), len(x_grid))


def orthorectify(frame_bgr: np.ndarray,
                 image_points: list,
                 model_points: list,
                 resolution_m: float = 0.02) -> tuple:
    """
    Orthorectify a video frame using 4 GCPs (ground control points).

    Parameters
    ----------
    frame_bgr     : (H, W, 3) uint8  BGR frame from cv2
    image_points  : list of 4 (col, row) pixel positions in frame
    model_points  : list of 4 (X, Y, Z) local-metre world positions
    resolution_m  : output pixel size in metres

    Returns
    -------
    ortho_rgb  : (H_out, W_out, 3) float32  orthorectified image
    x_grid     : 1-D  world X
    y_grid     : 1-D  world Y  (top-down)
    H_fwd      : (3,3) homography  pixel → world
    """
    img_pts = np.array(image_points, dtype=float)   # (4, 2)  col, row
    wld_pts = np.array(model_points, dtype=float)[:, :2]  # (4, 2)  X, Y

    # H_fwd : pixel (col,row) → world (X,Y)
    H_fwd = compute_homography_dlt(img_pts, wld_pts)

    x_grid, y_grid, out_shape = build_ortho_grid(wld_pts, resolution_m)

    # For the warp we need  H_inv: world grid pixel → source image pixel
    # World pixel coords: col = (X - x_min)/res,  row = (y_max - Y)/res
    x_min, y_max = x_grid[0], y_grid[0]
    # world pixel → world metre
    Tpx2m = np.array([[resolution_m, 0, x_min],
                       [0, -resolution_m, y_max],
                       [0,  0,            1]])
    # world metre → image pixel  (inverse of H_fwd)
    H_inv_total = np.linalg.inv(H_fwd) @ Tpx2m   # world px → img px

    frame_f = jnp.asarray(frame_bgr[..., ::-1].astype(np.float32) / 255.0)
    H_inv_j = jnp.asarray(H_inv_total.astype(np.float32))

    ortho = warp_perspective(frame_f, H_inv_j, out_shape)
    return np.asarray(ortho), x_grid, y_grid, H_fwd


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  STABILISATION — Lucas-Kanade optical flow (JAX)
# ═══════════════════════════════════════════════════════════════════════════════

@jax.jit
def _image_gradients(img: jnp.ndarray) -> tuple:
    """Sobel gradients (Ix, Iy) of a grayscale float image."""
    kx = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32) / 8.0
    ky = kx.T
    Ix = jax.scipy.signal.convolve2d(img, kx, mode="same")
    Iy = jax.scipy.signal.convolve2d(img, ky, mode="same")
    return Ix, Iy


@functools.partial(jax.jit, static_argnums=(3,))
def lk_track_points(img1: jnp.ndarray,
                    img2: jnp.ndarray,
                    pts:  jnp.ndarray,
                    win_half: int = 7) -> jnp.ndarray:
    """
    Lucas-Kanade optical flow for a set of interest points.

    Parameters
    ----------
    img1, img2 : (H, W) float32  grayscale frames
    pts        : (N, 2) float32  (col, row) feature positions
    win_half   : half-width of the integration window

    Returns
    -------
    flow : (N, 2) float32  displacement (du, dv) per point
    """
    Ix, Iy = _image_gradients(img1)
    It = img2 - img1

    def _one_pt(pt):
        col, row = pt[0], pt[1]
        c0 = jnp.clip(jnp.int32(col) - win_half, 0, img1.shape[1] - 1)
        r0 = jnp.clip(jnp.int32(row) - win_half, 0, img1.shape[0] - 1)
        ws = 2 * win_half + 1

        ix_w  = jax.lax.dynamic_slice(Ix, (r0, c0), (ws, ws)).ravel()
        iy_w  = jax.lax.dynamic_slice(Iy, (r0, c0), (ws, ws)).ravel()
        it_w  = jax.lax.dynamic_slice(It, (r0, c0), (ws, ws)).ravel()

        sxx = jnp.dot(ix_w, ix_w)
        sxy = jnp.dot(ix_w, iy_w)
        syy = jnp.dot(iy_w, iy_w)
        sxt = jnp.dot(ix_w, it_w)
        syt = jnp.dot(iy_w, it_w)

        det = sxx * syy - sxy * sxy + 1e-12
        du  = (-syy * sxt + sxy * syt) / det
        dv  = ( sxy * sxt - sxx * syt) / det
        return jnp.array([du, dv])

    return jax.vmap(_one_pt)(pts)


def stabilise_frame(frame_gray: np.ndarray,
                    ref_gray:   np.ndarray,
                    mask:       np.ndarray,
                    max_corners: int = 200) -> np.ndarray:
    """
    Estimate and remove camera motion relative to `ref_gray` using LK flow
    on pixels selected from stable regions (mask).

    Returns stabilised frame (same shape, float32 0-1).
    """
    # Detect Harris corners inside the mask
    import cv2
    mask_u8 = (mask * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        (ref_gray * 255).astype(np.uint8),
        maxCorners=max_corners, qualityLevel=0.01, minDistance=10,
        mask=mask_u8,
    )
    if corners is None or len(corners) < 4:
        return frame_gray

    pts = corners.reshape(-1, 2).astype(np.float32)   # (N, 2) col, row
    ref_j   = jnp.asarray(ref_gray.astype(np.float32))
    frame_j = jnp.asarray(frame_gray.astype(np.float32))
    pts_j   = jnp.asarray(pts)

    flow = np.asarray(lk_track_points(ref_j, frame_j, pts_j, win_half=7))
    pts2 = pts + flow

    # Homography from matched points
    H_stab, inliers = cv2.findHomography(pts, pts2, cv2.RANSAC, 3.0)
    if H_stab is None:
        return frame_gray

    H_stab_inv = np.linalg.inv(H_stab)
    h, w = frame_gray.shape
    H_j = jnp.asarray(H_stab_inv.astype(np.float32))
    frame_j = jnp.asarray(frame_gray.astype(np.float32))
    return np.asarray(warp_perspective(frame_j, H_j, (h, w)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PIV — FFT phase-correlation on interrogation windows (vmap)
# ═══════════════════════════════════════════════════════════════════════════════

@jax.jit
def fft_xcorr(w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
    """
    Phase-only normalised cross-correlation of two square interrogation windows.
    Returns correlation map (fftshifted so peak at centre = zero displacement).
    ws must be statically known (shapes are always concrete in JAX tracing).
    """
    ws = w1.shape[0]
    # Hanning window — shape is concrete at trace time so jnp.hanning works
    h   = jnp.hanning(ws)
    win = jnp.outer(h, h)
    w1c = (w1 - w1.mean()) * win
    w2c = (w2 - w2.mean()) * win
    F1  = jnp.fft.rfft2(w1c)
    F2  = jnp.fft.rfft2(w2c)
    cross = F1 * jnp.conj(F2)
    cross_norm = cross / (jnp.abs(cross) + 1e-8)
    corr = jnp.fft.irfft2(cross_norm, s=(ws, ws))
    return jnp.fft.fftshift(corr)


@jax.jit
def _gaussian_subpixel(corr: jnp.ndarray) -> jnp.ndarray:
    """
    Sub-pixel peak localisation using 3-point Gaussian fit on each axis.
    Returns (dx, dy) displacement (positive = rightward / downward).
    """
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    pk = jnp.argmax(corr)
    py = pk // corr.shape[1]
    px = pk %  corr.shape[1]

    # Clamp to avoid boundary issues
    py = jnp.clip(py, 1, corr.shape[0] - 2)
    px = jnp.clip(px, 1, corr.shape[1] - 2)

    eps = 1e-8
    c0y = jnp.log(jnp.abs(corr[py,   px]) + eps)
    cmy = jnp.log(jnp.abs(corr[py-1, px]) + eps)
    cpy = jnp.log(jnp.abs(corr[py+1, px]) + eps)
    c0x = jnp.log(jnp.abs(corr[py, px  ]) + eps)
    cmx = jnp.log(jnp.abs(corr[py, px-1]) + eps)
    cpx = jnp.log(jnp.abs(corr[py, px+1]) + eps)

    dy_sub = 0.5 * (cmy - cpy) / (cmy - 2*c0y + cpy + eps)
    dx_sub = 0.5 * (cmx - cpx) / (cmx - 2*c0x + cpx + eps)

    dy = (py + dy_sub) - cy
    dx = (px + dx_sub) - cx
    return jnp.array([dx, dy])


def _extract_windows_batch(frame: np.ndarray,
                            y0s: np.ndarray,
                            x0s: np.ndarray,
                            ws: int) -> np.ndarray:
    """Extract a batch of (ws×ws) windows at positions (x0s, y0s)."""
    n = len(y0s)
    batch = np.zeros((n, ws, ws), dtype=np.float32)
    H, W = frame.shape
    for i in range(n):
        r, c = int(y0s[i]), int(x0s[i])
        r = min(r, H - ws);  c = min(c, W - ws)
        batch[i] = frame[r:r+ws, c:c+ws]
    return batch


def piv_frame_pair(frame1:    np.ndarray,
                   frame2:    np.ndarray,
                   win_size:  int   = 32,
                   step:      int   = 16,
                   fps:       float = 30.0,
                   res_m:     float = 0.02,
                   max_disp_px: int = 12) -> dict:
    """
    Compute 2-D velocity field between two orthorectified grayscale frames
    using FFT phase-correlation PIV.

    Parameters
    ----------
    frame1, frame2  : (H, W) float32  orthorectified grayscale frames
    win_size        : interrogation window size (pixels)
    step            : grid step (pixels)
    fps             : frames per second → converts px/frame → m/s
    res_m           : ortho pixel size (m/px)
    max_disp_px     : displacements larger than this are rejected

    Returns
    -------
    dict with keys:
        X, Y    : (N,) world local-metre positions of velocity vectors
        U, V    : (N,) velocity components (m/s)  U=streamwise, V=lateral
        norm    : (N,) speed (m/s)
        n_valid : int
    """
    H, W = frame1.shape
    # Grid of window top-left corners
    rows = np.arange(0, H - win_size, step)
    cols = np.arange(0, W - win_size, step)
    gc, gr = np.meshgrid(cols, rows)
    r0s = gr.ravel();  c0s = gc.ravel()
    n_total = len(r0s)

    # Extract window batches
    batch1 = _extract_windows_batch(frame1, r0s, c0s, win_size)
    batch2 = _extract_windows_batch(frame2, r0s, c0s, win_size)

    # JAX vmap cross-correlation
    batch1_j = jnp.asarray(batch1)
    batch2_j = jnp.asarray(batch2)

    xcorr_batch = jax.vmap(fft_xcorr)(batch1_j, batch2_j)          # (N, ws, ws)
    disps       = jax.vmap(_gaussian_subpixel)(xcorr_batch)         # (N, 2)  dx, dy

    dx = np.asarray(disps[:, 0])   # pixel displacement, positive = rightward
    dy = np.asarray(disps[:, 1])   # positive = downward

    # Convert to velocity (m/s):  1 px/frame × fps × res_m
    U =  dx * fps * res_m          # X-component (m/s)  — positive east
    V = -dy * fps * res_m          # Y-component (m/s)  — positive north (row↓ = Y↑)

    # Window centres in world coords
    # frame column c corresponds to x_grid[c], row r corresponds to y_grid[r]
    cx_px = c0s + win_size // 2
    cy_px = r0s + win_size // 2

    # Filter large / zero displacements
    norm = np.sqrt(U**2 + V**2)
    valid = (np.abs(dx) < max_disp_px) & (np.abs(dy) < max_disp_px) & (norm > 0.01)

    return dict(
        col_px  = cx_px[valid].astype(np.float32),
        row_px  = cy_px[valid].astype(np.float32),
        U       = U[valid].astype(np.float32),
        V       = V[valid].astype(np.float32),
        norm    = norm[valid].astype(np.float32),
        n_valid = int(valid.sum()),
        n_total = n_total,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GAUSSIAN INTERPOLATION  (matches opyflow npInterpolateVTK2D)
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_interpolate(src_xy:   np.ndarray,
                          src_vals: np.ndarray,
                          dst_xy:   np.ndarray,
                          radius:   float = 1.0,
                          sharpness: float = 2.0) -> np.ndarray:
    """
    Gaussian kernel interpolation: each dst point receives a weighted
    average of all src points within `radius`, with weights
    w = exp(-(r/radius)^sharpness).

    Parameters
    ----------
    src_xy   : (N, 2) source positions (world m)
    src_vals : (N, K) source values (e.g. [U, V])
    dst_xy   : (M, 2) destination positions
    radius   : kernel radius (m)
    sharpness: Gaussian sharpness exponent

    Returns
    -------
    (M, K) interpolated values
    """
    src_j = jnp.asarray(src_xy.astype(np.float32))
    val_j = jnp.asarray(src_vals.astype(np.float32))
    dst_j = jnp.asarray(dst_xy.astype(np.float32))

    @jax.jit
    def _interp_one(dpt):
        diff = src_j - dpt[None, :]                     # (N, 2)
        dist = jnp.sqrt((diff**2).sum(axis=1))          # (N,)
        w    = jnp.where(dist < radius,
                         jnp.exp(-(dist / radius) ** sharpness),
                         0.0)                           # (N,)
        w_sum = w.sum() + 1e-12
        return (w[:, None] * val_j).sum(axis=0) / w_sum  # (K,)

    return np.asarray(jax.vmap(_interp_one)(dst_j))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DISCHARGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def sample_transect(terrain_xy: np.ndarray,
                    terrain_z:  np.ndarray,
                    P1: np.ndarray,
                    P2: np.ndarray,
                    n_pts: int = 500) -> tuple:
    """
    Sample MNT elevation along a transect P1→P2 using Gaussian interpolation.

    Parameters
    ----------
    terrain_xy : (M, 2)  terrain point local-metre XY
    terrain_z  : (M,)    terrain elevation (m MSL)
    P1, P2     : (2,)    transect endpoints in local metres
    n_pts      : number of sample points

    Returns
    -------
    pts_xy     : (n_pts, 2)  transect positions
    dist_m     : (n_pts,)    cumulative distance (m)
    z_bed      : (n_pts,)    interpolated bed elevation (m MSL)
    """
    t = np.linspace(0, 1, n_pts)
    pts_xy = P1[None, :] + t[:, None] * (P2 - P1)[None, :]
    dist_m = np.linalg.norm(P2 - P1) * t

    # Use Gaussian interpolation with radius = 3× point spacing
    spacing = np.linalg.norm(P2 - P1) / n_pts
    z_interp = gaussian_interpolate(terrain_xy, terrain_z[:, None],
                                    pts_xy,
                                    radius=max(0.5, 3 * spacing),
                                    sharpness=2.0)
    return pts_xy, dist_m, z_interp[:, 0]


@jax.jit
def compute_discharge(V_norm:  jnp.ndarray,
                      z_bed:   jnp.ndarray,
                      z_water: float,
                      dl:      float,
                      alpha:   float = 0.9) -> jnp.ndarray:
    """
    Q = α · ∫ |V| · h dl   (trapezoidal rule along transect).

    Parameters
    ----------
    V_norm  : (n_pts,)  surface velocity magnitude (m/s)
    z_bed   : (n_pts,)  bed elevation (m MSL)
    z_water : float     water surface elevation (m MSL)
    dl      : float     transect step (m)
    alpha   : float     surface-to-depth-mean velocity coefficient (0.9 typical)

    Returns
    -------
    Q  : scalar (m³/s)
    """
    h = jnp.maximum(z_water - z_bed, 0.0)      # flow depth, non-negative
    integrand = alpha * V_norm * h
    # Only integrate over wet cells
    wet = h > 0.0
    Q = jnp.trapezoid(jnp.where(wet, integrand, 0.0)) * dl
    return Q


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# opyflow GCPs for IMG_1139 (downstream bridge)
GCP_IMAGE_1139 = np.array([(355,429),(1338,350),(99,562),(1673,364)], dtype=float)
GCP_MODEL_1139 = np.array([(30.13,-8.28,0),(32.88,-28.08,0),
                             (20.46,-4.47,0.4),(21.32,-27.14,0.4)], dtype=float)

# opyflow GCPs for IMG_1142 (upstream bridge)
GCP_IMAGE_1142 = np.array([(830,564),(1480,594),(1750,800),(0,616),(369,565)], dtype=float)
GCP_MODEL_1142 = np.array([(-2,-10.5,0),(0,0.,0),(21.2,-5,0),
                             (7.4,-21.7,0),(-2.6,-19.9,0)], dtype=float)

# Discharge transect endpoints (from opyflow code, local metres)
TRANSECT_L = np.array([16.0, -2.0])     # left bank
TRANSECT_R = np.array([11.0, -23.0])    # right bank
ZWATER     = 14.4                        # flood water surface elevation (m MSL)


def run_jax_lspiv(frame_paths:    list,
                  image_points:   np.ndarray = GCP_IMAGE_1139,
                  model_points:   np.ndarray = GCP_MODEL_1139,
                  mnt_xyz_path:   str = "data/brague/MNT.xyz",
                  stabilise:      bool = False,
                  win_size:       int  = 32,
                  step:           int  = 16,
                  fps:            float = 30.0,
                  res_m:          float = 0.02,
                  interp_radius:  float = 1.0,
                  interp_sharp:   float = 2.0,
                  z_water:        float = ZWATER,
                  alpha:          float = 0.9,
                  mnt_stride:     int   = 30,
                  ) -> dict:
    """
    Full JAX LSPIV pipeline.

    Parameters
    ----------
    frame_paths   : list of PNG paths (consecutive pairs processed)
    image_points  : (N, 2) GCP pixel coordinates
    model_points  : (N, 3) GCP world coordinates (local metres)
    mnt_xyz_path  : path to MNT.xyz (tab-separated X Y Z, Lambert-93)
    stabilise     : if True, apply LK stabilisation before orthorectification
    ...

    Returns
    -------
    dict with keys:
        X, Y        : world local-metre positions of all velocity vectors
        U, V, norm  : velocity components and magnitude (m/s)
        ortho_imgs  : list of orthorectified frames (float32)
        x_grid, y_grid
        terrain_xy, terrain_z
        transect_pts, dist_m, z_bed_transect, V_transect, Q
    """
    import cv2

    print(f"  [JAX-LSPIV] Loading {len(frame_paths)} frames …")
    frames_gray = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            raise FileNotFoundError(fp)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames_gray.append(gray)
        print(f"    loaded {Path(fp).name}  {gray.shape}")

    # ── Orthorectification ────────────────────────────────────────────
    print("  [JAX-LSPIV] Orthorectifying …")
    ortho_frames = []
    x_grid = y_grid = None
    H_fwd = None
    for i, (fg, fp) in enumerate(zip(frames_gray, frame_paths)):
        img_bgr = cv2.imread(str(fp))
        ortho, xg, yg, Hf = orthorectify(img_bgr, image_points,
                                           model_points, res_m)
        if i == 0:
            x_grid, y_grid, H_fwd = xg, yg, Hf
        # Grayscale for PIV
        ortho_frames.append(np.dot(ortho[..., :3],
                                    np.array([0.299, 0.587, 0.114])))
        print(f"    ortho {i}: {ortho.shape}  "
              f"X=[{xg[0]:.1f},{xg[-1]:.1f}]  Y=[{yg[-1]:.1f},{yg[0]:.1f}]")

    # ── PIV on consecutive pairs ──────────────────────────────────────
    print("  [JAX-LSPIV] Running PIV (FFT cross-correlation) …")
    all_col, all_row, all_U, all_V, all_norm = [], [], [], [], []

    for i in range(len(ortho_frames) - 1):
        res = piv_frame_pair(ortho_frames[i], ortho_frames[i + 1],
                              win_size=win_size, step=step,
                              fps=fps, res_m=res_m)
        all_col.append(res["col_px"])
        all_row.append(res["row_px"])
        all_U.append(res["U"])
        all_V.append(res["V"])
        all_norm.append(res["norm"])
        print(f"    pair {i}→{i+1}: {res['n_valid']}/{res['n_total']} vectors valid")

    col_px = np.concatenate(all_col)
    row_px = np.concatenate(all_row)
    U_all  = np.concatenate(all_U)
    V_all  = np.concatenate(all_V)
    norm_all = np.concatenate(all_norm)

    # Pixel → world local metres
    X_all = x_grid[0] + col_px * res_m      # x_grid[0] = x_min
    Y_all = y_grid[0] - row_px * res_m      # y_grid[0] = y_max (top-down)

    # ── Load MNT (strided sample) ─────────────────────────────────────
    print(f"  [JAX-LSPIV] Loading MNT.xyz (stride={mnt_stride}) …")
    import subprocess, tempfile
    tmp = tempfile.mktemp(suffix=".xyz")
    subprocess.run(f"awk 'NR % {mnt_stride} == 0' '{mnt_xyz_path}' > {tmp}",
                   shell=True, check=True)
    mnt = np.loadtxt(tmp, dtype=np.float32)
    # Convert Lambert-93 → local metres
    terrain_xy = mnt[:, :2] - np.array([ORIGIN_X, ORIGIN_Y], dtype=np.float32)
    terrain_z  = mnt[:, 2]
    print(f"    {len(terrain_z):,} terrain pts  "
          f"Z=[{terrain_z.min():.2f},{terrain_z.max():.2f}] m")

    # ── Choose transect: opyflow default or auto from coverage ───────
    # The opyflow transect (16,-2)→(11,-23) requires the upstream video
    # (IMG_1142).  If the velocity coverage doesn't reach it, auto-place
    # a cross-channel transect at the mid-Y of actual coverage.
    P1_opyf = np.asarray(TRANSECT_L, dtype=np.float32)
    P2_opyf = np.asarray(TRANSECT_R, dtype=np.float32)
    x_cov_min = X_all.min(); x_cov_max = X_all.max()
    y_cov_min = Y_all.min(); y_cov_max = Y_all.max()
    # Check overlap with opyflow transect
    t_x_min = min(P1_opyf[0], P2_opyf[0]); t_x_max = max(P1_opyf[0], P2_opyf[0])
    overlap = (t_x_min <= x_cov_max) and (t_x_max >= x_cov_min)
    if overlap:
        P1, P2 = P1_opyf, P2_opyf
        print(f"  [JAX-LSPIV] Using opyflow transect L={P1} R={P2}")
    else:
        # Auto transect: horizontal cross-section at mid-Y of coverage
        mid_y = float((y_cov_min + y_cov_max) / 2)
        P1 = np.array([x_cov_min, mid_y], dtype=np.float32)
        P2 = np.array([x_cov_max, mid_y], dtype=np.float32)
        print(f"  [JAX-LSPIV] Opyflow transect outside coverage — "
              f"auto transect at Y={mid_y:.1f} m  "
              f"L={P1.tolist()} R={P2.tolist()}")

    # ── Gaussian interpolation of velocity to transect ───────────────
    print("  [JAX-LSPIV] Interpolating velocity to transect …")
    trans_pts, dist_m, z_bed_t = sample_transect(
        terrain_xy, terrain_z, P1, P2, n_pts=300)

    # Interpolate velocity to transect points
    src_xy  = np.column_stack([X_all, Y_all])
    src_val = np.column_stack([U_all, V_all])
    V_trans = gaussian_interpolate(src_xy, src_val, trans_pts,
                                    radius=interp_radius,
                                    sharpness=interp_sharp)
    norm_trans = np.sqrt(V_trans[:, 0]**2 + V_trans[:, 1]**2)

    # ── Discharge ─────────────────────────────────────────────────────
    print("  [JAX-LSPIV] Computing discharge …")
    dl = float(dist_m[1] - dist_m[0]) if len(dist_m) > 1 else 0.1
    Q = float(compute_discharge(
        jnp.asarray(norm_trans),
        jnp.asarray(z_bed_t),
        z_water, dl, alpha,
    ))
    print(f"  [JAX-LSPIV] Q = {Q:.1f} m³/s  "
          f"(α={alpha}, z_water={z_water} m, transect L={dist_m[-1]:.1f} m)")

    return dict(
        X=X_all, Y=Y_all, U=U_all, V=V_all, norm=norm_all,
        x_grid=x_grid, y_grid=y_grid, H_fwd=H_fwd,
        ortho_frames=ortho_frames,
        terrain_xy=terrain_xy, terrain_z=terrain_z,
        transect_pts=trans_pts, dist_m=dist_m,
        z_bed_transect=z_bed_t,
        V_transect=V_trans, norm_transect=norm_trans,
        Q=Q, z_water=z_water, alpha=alpha,
    )
