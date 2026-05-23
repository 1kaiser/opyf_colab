"""
modules/segment_water.py
=========================
JAX/Flax water segmentation — replaces hand-drawn masks and orthorectified
reference frames with automatic per-frame water detection.

Three modes (in order of accuracy):
─────────────────────────────────────
1. diff_segment   (recommended, no training needed)
   Warp event frame to ortho coordinates via LightGlue homography,
   then difference against the dry pre-event ortho.
   Water = changed pixels (flooded area not present in dry ortho).

2. color_segment  (zero-dep fallback, fast)
   HSV + YCbCr multi-threshold heuristic tuned for river-water in
   both aerial and bridge-level footage.

3. SegNet (learned, highest quality when weights available)
   Lightweight Flax U-Net (~500 K params):
   Encoder: 4× (Conv 3×3 + BN + ReLU) with stride-2 downsampling
   Decoder: 4× bilinear upsample + skip-connection merge + Conv
   Input:  RGB (any size → resized to 512×512)
   Output: 1-channel sigmoid logit → threshold at 0.5

Public API
──────────
    diff_segment(event_frame_path, ortho_path, sp_jit, sp_vars, lg_jit, lg_vars)
        -> (mask uint8, H homography|None)

    color_segment(frame_bgr)
        -> mask uint8  (1=water, 0=land)

    SegNet                  — Flax model class
    load_segnet(weights)    -> (jit_fn, variables)
    infer_segnet(frame_bgr, jit_fn, variables) -> mask uint8

    segment_frame(frame_bgr, method, **kwargs) -> mask uint8
        Unified entry point. method = "diff" | "color" | "segnet"

CLI
───
    python3 modules/segment_water.py --method color  --image frame.jpg --output mask.png
    python3 modules/segment_water.py --method diff   --image frame.jpg --ortho Ortho.tif
        --sp weights/superpoint.msgpack --lg weights/superpoint_lightglue.msgpack
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, serialization

# ── 1. Difference-based segmentation ─────────────────────────────────────────

def diff_segment(
    event_frame_path: str,
    ortho_path: str,
    sp_jit,
    sp_vars,
    lg_jit,
    lg_vars,
    change_thresh: float = 30.0,
    min_area_px: int = 5000,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Detect water pixels by warping the event frame onto the dry ortho
    and computing the per-pixel colour difference.

    Steps
    -----
    1. Match event frame → ortho using SuperPoint + LightGlue
    2. Estimate homography (RANSAC)
    3. Warp event frame to ortho coordinate space
    4. L2 colour diff > change_thresh → changed region
    5. Morphological clean-up → water mask

    Returns
    -------
    mask : (H_ortho, W_ortho) uint8  (255=water, 0=land)
    H    : (3, 3) float64 homography, or None if estimation failed
    """
    from modules.infer_features import match_images, estimate_homography

    frame_bgr = cv2.imread(event_frame_path)
    ortho_bgr = cv2.imread(ortho_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read frame: {event_frame_path}")
    if ortho_bgr is None:
        raise FileNotFoundError(f"Cannot read ortho: {ortho_path}")

    ho, wo = ortho_bgr.shape[:2]

    # Feature matching: event frame → ortho
    mk_frame, mk_ortho = match_images(
        event_frame_path, ortho_path, sp_jit, sp_vars, lg_jit, lg_vars
    )
    print(f"  diff_segment: {len(mk_frame)} feature matches")

    H = estimate_homography(mk_frame, mk_ortho)

    if H is not None:
        warped = cv2.warpPerspective(frame_bgr, H, (wo, ho))
    else:
        # Fallback: resize frame to ortho size
        print("  Homography failed, using resize fallback")
        warped = cv2.resize(frame_bgr, (wo, ho))

    # Colour difference
    diff = np.linalg.norm(
        warped.astype(np.float32) - ortho_bgr.astype(np.float32), axis=2
    )
    mask = (diff > change_thresh).astype(np.uint8) * 255

    # Morphological clean-up: remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove tiny components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            clean[labels == i] = 255
    mask = clean

    return mask, H


# ── 2. Colour-threshold segmentation ─────────────────────────────────────────

def color_segment(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Fast water segmentation via HSV + YCbCr colour thresholds.
    Tuned for river / flood water in both aerial and bridge-view footage.

    Heuristics
    ----------
    • HSV: water tends to be low-saturation, mid-to-low value (darker than sky)
    • YCbCr: water has elevated Cb (blue-red difference) relative to vegetation
    • Combine via union → morphological clean-up

    Returns uint8 mask (255=water, 0=land).
    """
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

    # HSV heuristic: low-S, mid-V range (avoids bright sky and very dark shadows)
    h_mask = (
        (hsv[:, :, 1] < 80)   &   # low saturation
        (hsv[:, :, 2] > 30)   &   # not black
        (hsv[:, :, 2] < 200)       # not white/sky
    ).astype(np.uint8) * 255

    # YCbCr: water has higher Cb than surrounding land/vegetation
    cb_channel = ycrcb[:, :, 2].astype(np.int16)
    y_channel  = ycrcb[:, :, 0].astype(np.int16)
    cb_mask    = (cb_channel > 128).astype(np.uint8) * 255

    # Union
    mask = cv2.bitwise_or(h_mask, cb_mask)

    # Morphological clean-up
    k5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k25  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k25)
    mask = cv2.medianBlur(mask, 9)

    return mask


# ── 3. SegNet — Flax U-Net for water segmentation ────────────────────────────

class _EncBlock(nn.Module):
    """Encoder block: Conv + BN + ReLU, stride-2 downsampling."""
    features: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(self.features, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)


class _DecBlock(nn.Module):
    """Decoder block: bilinear upsample + skip-connection concat + Conv."""
    features: int

    @nn.compact
    def __call__(self, x, skip, train: bool = False):
        # bilinear upsample ×2
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method="bilinear")
        x = jnp.concatenate([x, skip], axis=-1)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)


class SegNet(nn.Module):
    """
    Lightweight U-Net binary segmentation model (~500 K params).

    Input  : (B, 512, 512, 3) float32  (normalised 0–1)
    Output : (B, 512, 512, 1) float32  (sigmoid logit — threshold at 0.5 for mask)
    """
    features: tuple = (16, 32, 64, 128)

    @nn.compact
    def __call__(self, x, train: bool = False):
        skips = []
        for f in self.features:
            s = nn.Conv(f, (3, 3), padding="SAME")(x)
            s = nn.relu(s)
            skips.append(s)
            x = _EncBlock(f)(x, train)

        # bottleneck
        f_bot = self.features[-1] * 2
        x = nn.Conv(f_bot, (3, 3), padding="SAME")(x)
        x = nn.relu(x)

        for f, skip in zip(reversed(self.features), reversed(skips)):
            x = _DecBlock(f)(x, skip, train)

        return nn.sigmoid(nn.Conv(1, (1, 1))(x))


_SEG_INPUT = 512


def load_segnet(weights_path: str):
    """Load SegNet weights from a .msgpack file. Returns (jit_fn, variables)."""
    model = SegNet()
    print(f"  Loading SegNet from {weights_path} …")
    with open(weights_path, "rb") as f:
        variables = serialization.msgpack_restore(f.read())
    jit_fn = jax.jit(lambda v, x: model.apply(v, x, train=False))
    print("  SegNet ready.")
    return jit_fn, variables


def infer_segnet(frame_bgr: np.ndarray, jit_fn, variables) -> np.ndarray:
    """
    Run SegNet on a BGR frame.

    Returns uint8 mask (255=water, 0=land) at original frame resolution.
    """
    oh, ow = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_r  = cv2.resize(rgb, (_SEG_INPUT, _SEG_INPUT)).astype(np.float32) / 255.0
    x      = jnp.array(rgb_r[None])         # (1, H, W, 3)

    logit  = jit_fn(variables, x)           # (1, H, W, 1)
    pred   = (np.array(logit[0, ..., 0]) > 0.5).astype(np.uint8) * 255
    return cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)


# ── Unified entry point ───────────────────────────────────────────────────────

def segment_frame(
    frame_bgr: np.ndarray,
    method: str = "color",
    segnet_jit=None,
    segnet_vars=None,
    event_frame_path: str | None = None,
    ortho_path: str | None = None,
    sp_jit=None, sp_vars=None,
    lg_jit=None, lg_vars=None,
) -> np.ndarray:
    """
    Unified water segmentation entry point.

    method = "color"  — colour threshold (always available)
           = "segnet" — SegNet (requires segnet_jit + segnet_vars)
           = "diff"   — difference method (requires event_frame_path, ortho_path,
                        sp_jit, sp_vars, lg_jit, lg_vars)
    """
    if method == "segnet":
        if segnet_jit is None:
            raise ValueError("segnet_jit/segnet_vars required for method='segnet'")
        return infer_segnet(frame_bgr, segnet_jit, segnet_vars)

    if method == "diff":
        if None in (event_frame_path, ortho_path, sp_jit, sp_vars, lg_jit, lg_vars):
            raise ValueError("diff method requires event_frame_path, ortho_path, and feature models")
        mask, _ = diff_segment(event_frame_path, ortho_path, sp_jit, sp_vars, lg_jit, lg_vars)
        return mask

    return color_segment(frame_bgr)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Water segmentation for flood event frames")
    p.add_argument("--method",  choices=["color", "diff", "segnet"], default="color")
    p.add_argument("--image",   required=True, help="Event frame (BGR image)")
    p.add_argument("--ortho",   help="Dry pre-event ortho (for diff method)")
    p.add_argument("--sp",      help="SuperPoint weights (for diff method)")
    p.add_argument("--lg",      help="LightGlue weights (for diff method)")
    p.add_argument("--weights", help="SegNet weights (for segnet method)")
    p.add_argument("--output",  default="output/water_mask.png")
    args = p.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Cannot read: {args.image}")
        sys.exit(1)

    if args.method == "color":
        mask = color_segment(frame)

    elif args.method == "diff":
        from modules.infer_features import load_feature_models
        sp_jit, sp_vars, lg_jit, lg_vars = load_feature_models(args.sp, args.lg)
        mask, H = diff_segment(args.image, args.ortho, sp_jit, sp_vars, lg_jit, lg_vars)
        if H is not None:
            print(f"  Homography:\n{H}")

    elif args.method == "segnet":
        jit_fn, variables = load_segnet(args.weights)
        mask = infer_segnet(frame, jit_fn, variables)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, mask)
    water_pct = 100 * (mask > 0).sum() / mask.size
    print(f"  Water: {water_pct:.1f}%   Mask → {args.output}")
