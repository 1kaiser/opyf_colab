"""
modules/infer_features.py
==========================
SuperPoint + LightGlue JAX inference — keypoint detection and matching.

Public API
----------
    load_feature_models(sp_weights, lg_weights) -> (sp_jit, sp_vars, lg_jit, lg_vars)
    detect_keypoints(image_path, sp_jit, sp_vars, top_k=1024) -> (kpts, feats)
    match_features(kpts1, feats1, kpts2, feats2, lg_jit, lg_vars, min_score=0.1)
        -> (mkpts1, mkpts2)  matched pixel coordinates in original image space
    match_images(img1, img2, sp_jit, sp_vars, lg_jit, lg_vars) -> (mkpts1, mkpts2)
    estimate_homography(mkpts1, mkpts2) -> H (3×3 ndarray) or None
    draw_matches(img1_path, img2_path, mkpts1, mkpts2) -> BGR canvas ndarray

CLI
---
    python3 modules/infer_features.py --img1 a.jpg --img2 b.jpg
        --sp_weights weights/superpoint.msgpack
        --lg_weights weights/superpoint_lightglue.msgpack
        --output output/matches.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "jax"))
from jax_lightglue.models.superpoint import SuperPoint
from jax_lightglue.models.lightglue import LightGlue

_N_LAYERS = 9


def load_feature_models(sp_weights: str, lg_weights: str):
    """Load SuperPoint and LightGlue weights. Returns (sp_jit, sp_vars, lg_jit, lg_vars)."""
    sp_model = SuperPoint()
    lg_model = LightGlue(n_layers=_N_LAYERS)

    print(f"  Loading SuperPoint from {sp_weights} …")
    with open(sp_weights, "rb") as f:
        sp_vars = serialization.from_bytes(None, f.read())

    print(f"  Loading LightGlue from {lg_weights} …")
    with open(lg_weights, "rb") as f:
        lg_vars = serialization.from_bytes(None, f.read())

    sp_jit = jax.jit(sp_model.apply)
    lg_jit = jax.jit(lg_model.apply)
    return sp_jit, sp_vars, lg_jit, lg_vars


def _load_gray(image_path: str):
    """Load image as float grayscale JAX array (1, H, W, 1), H/W padded to 8×."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w   = img.shape
    nh, nw = (h // 8) * 8, (w // 8) * 8
    img    = cv2.resize(img, (nw, nh))
    return jnp.array(img[None, ..., None] / 255.0), (h, w)


def detect_keypoints(image_path: str, sp_jit, sp_vars, top_k: int = 1024):
    """
    Run SuperPoint on a single image.

    Returns
    -------
    kpts  : (K, 2) int32 array — keypoint pixel coords (x, y) at original resolution
    feats : (K, 256) float32 array — descriptors
    """
    img_jax, (oh, ow) = _load_gray(image_path)
    _, ih, iw, _ = img_jax.shape

    out    = sp_jit(sp_vars, img_jax)
    scores = out["scores"][0]
    descs  = out["descriptors"][0]

    top_idx = jnp.argsort(scores.flatten())[::-1][:top_k]
    y, x    = jnp.unravel_index(top_idx, scores.shape)

    # descriptor coordinates (descriptor map stride = 8)
    dy = jnp.clip((y // 8).astype(jnp.int32), 0, descs.shape[0] - 1)
    dx = jnp.clip((x // 8).astype(jnp.int32), 0, descs.shape[1] - 1)
    feats = descs[dy, dx, :]

    # rescale to original image resolution
    kx = (np.array(x) * ow / iw).astype(np.int32)
    ky = (np.array(y) * oh / ih).astype(np.int32)
    kpts = np.stack([kx, ky], axis=-1)
    return kpts, np.array(feats)


def match_features(kpts1, feats1, kpts2, feats2, lg_jit, lg_vars, min_score: float = 0.1):
    """
    Run LightGlue on two sets of keypoints/features.

    Returns
    -------
    mkpts1, mkpts2 : (M, 2) int32 arrays — matched pixel coords (image 1 / image 2)
    """
    lg_input = {
        "image0": {"keypoints": jnp.array(kpts1)[None], "descriptors": jnp.array(feats1)[None]},
        "image1": {"keypoints": jnp.array(kpts2)[None], "descriptors": jnp.array(feats2)[None]},
    }
    out    = lg_jit(lg_vars, lg_input)
    scores = out["scores"][0, :-1, :-1]

    m0      = jnp.argmax(scores, axis=1)
    m1      = jnp.argmax(scores, axis=0)
    mutual  = jnp.arange(len(m0)) == m1[m0]
    conf    = jnp.exp(jnp.max(scores, axis=1))
    valid   = mutual & (conf > min_score)

    idx0 = np.array(jnp.where(valid)[0])
    idx1 = np.array(m0[jnp.where(valid)[0]])
    return kpts1[idx0], kpts2[idx1]


def match_images(img1_path: str, img2_path: str, sp_jit, sp_vars, lg_jit, lg_vars,
                 top_k: int = 1024, min_score: float = 0.1):
    """Convenience wrapper: detect + match two images. Returns (mkpts1, mkpts2)."""
    k1, f1 = detect_keypoints(img1_path, sp_jit, sp_vars, top_k)
    k2, f2 = detect_keypoints(img2_path, sp_jit, sp_vars, top_k)
    return match_features(k1, f1, k2, f2, lg_jit, lg_vars, min_score)


def estimate_homography(mkpts1, mkpts2, ransac_thresh: float = 4.0):
    """Estimate homography from matched point pairs using RANSAC. Returns 3×3 H or None."""
    if len(mkpts1) < 4:
        return None
    H, mask = cv2.findHomography(
        mkpts1.astype(np.float32), mkpts2.astype(np.float32),
        cv2.RANSAC, ransac_thresh
    )
    return H


def draw_matches(img1_path: str, img2_path: str, mkpts1, mkpts2) -> np.ndarray:
    """Draw matching lines on a side-by-side canvas. Returns BGR ndarray."""
    i1 = cv2.imread(img1_path)
    i2 = cv2.imread(img2_path)
    h1, w1 = i1.shape[:2]
    h2, w2 = i2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]  = i1
    canvas[:h2, w1:]  = i2

    for (x1, y1), (x2, y2) in zip(mkpts1, mkpts2):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (0, 255, 0), 1)
        cv2.circle(canvas, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        cv2.circle(canvas, (int(x2) + w1, int(y2)), 2, (0, 0, 255), -1)
    return canvas


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SuperPoint + LightGlue JAX matching")
    p.add_argument("--img1",       required=True)
    p.add_argument("--img2",       required=True)
    p.add_argument("--sp_weights", default="weights/superpoint.msgpack")
    p.add_argument("--lg_weights", default="weights/superpoint_lightglue.msgpack")
    p.add_argument("--output",     default="output/matches.jpg")
    p.add_argument("--top_k",      type=int, default=1024)
    args = p.parse_args()

    sp_jit, sp_vars, lg_jit, lg_vars = load_feature_models(args.sp_weights, args.lg_weights)
    mk1, mk2 = match_images(args.img1, args.img2, sp_jit, sp_vars, lg_jit, lg_vars, args.top_k)
    print(f"  Found {len(mk1)} matches")

    H = estimate_homography(mk1, mk2)
    if H is not None:
        print(f"  Homography estimated  ({len(mk1)} inliers)")

    canvas = draw_matches(args.img1, args.img2, mk1, mk2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, canvas)
    print(f"  Saved → {args.output}")
