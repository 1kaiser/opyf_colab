"""
modules/infer_video.py
=======================
VGGT JAX inference — video geometry: world points, depth, camera poses.

Public API
----------
    load_vggt(weights_path) -> (jit_fn, variables)
    infer_video(image_paths, jit_fn, variables) -> dict
        keys: world_points, depth, pose_enc

CLI
---
    python3 modules/infer_video.py --image_dir path/to/frames
        --weights weights/vggt_1b.msgpack --output output/vggt
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
from jax_vggt.models.vggt import VGGT

_INPUT_SIZE = 518
_PATCH_SIZE  = 14
_EMBED_DIM   = 1024


def load_vggt(weights_path: str):
    """Load VGGT weights. Returns (jit_fn, variables)."""
    model = VGGT(img_size=_INPUT_SIZE, patch_size=_PATCH_SIZE, embed_dim=_EMBED_DIM)
    print(f"  Loading VGGT from {weights_path} …")
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    jit_fn = jax.jit(model.apply)
    print("  VGGT ready.")
    return jit_fn, variables


def _preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE)) / 255.0
    return rgb.transpose(2, 0, 1)   # (3, H, W)


def infer_video(image_paths: list[str], jit_fn, variables) -> dict:
    """
    Run VGGT on a sequence of frames (minimum 2).

    Returns dict with keys:
        world_points : (N, H, W, 3)  — 3D world coordinates per pixel per frame
        depth        : (N, H, W)     — metric depth per pixel per frame
        pose_enc     : (1, N, D)     — camera pose encodings
    """
    if len(image_paths) < 2:
        raise ValueError("VGGT requires at least 2 frames")

    imgs  = np.stack([_preprocess(p) for p in image_paths])   # (N, 3, H, W)
    x     = jnp.array(imgs)[None]                              # (1, N, 3, H, W)

    preds = jit_fn(variables, x)
    return {
        "world_points": np.array(preds["world_points"][0]),
        "depth":        np.array(preds["depth"][0]),
        "pose_enc":     np.array(preds["pose_enc"]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VGGT JAX video geometry inference")
    p.add_argument("--image_dir", required=True, help="Directory of image frames")
    p.add_argument("--weights",   default="weights/vggt_1b.msgpack")
    p.add_argument("--output",    default="output/vggt")
    p.add_argument("--n_frames",  type=int, default=8, help="Max frames to use")
    args = p.parse_args()

    exts   = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        p for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )[:args.n_frames]

    if len(frames) < 2:
        print("Need at least 2 image frames in --image_dir")
        sys.exit(1)

    jit_fn, variables = load_vggt(args.weights)
    results = infer_video([str(f) for f in frames], jit_fn, variables)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    for key, arr in results.items():
        np.save(os.path.join(args.output, f"{key}.npy"), arr)
    print(f"  Results saved → {args.output}")
    print(f"  world_points: {results['world_points'].shape}")
    print(f"  depth:        {results['depth'].shape}")
