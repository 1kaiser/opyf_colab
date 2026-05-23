"""
modules/infer_depth.py
=======================
Depth Pro JAX inference — single-image metric depth + FOV estimation.

Public API
----------
    load_depth_pro(weights_path) -> (jit_fn, variables)
    infer_depth(image_path, jit_fn, variables) -> (inv_depth ndarray, fov_deg float)
    depth_to_colormap(inv_depth) -> uint8 RGB array (Viridis)

CLI
---
    python3 modules/infer_depth.py --image path.jpg --weights weights/depth_pro.msgpack
                                   --output output/depth_result.png
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
from jax_depth_pro.models.depth_pro import DepthPro

INPUT_SIZE = 1536

_VIT_CONFIG = {
    "img_size": 384, "patch_size": 16, "embed_dim": 1024,
    "depth": 24, "num_heads": 16, "init_values": 1e-5,
}


def load_depth_pro(weights_path: str):
    """Load Depth Pro weights. Returns (jit_fn, variables)."""
    model = DepthPro(vit_config=_VIT_CONFIG)
    print(f"  Loading Depth Pro weights from {weights_path} …")
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    jit_fn = jax.jit(model.apply)
    print("  Depth Pro ready.")
    return jit_fn, variables


def infer_depth(image_path: str, jit_fn, variables):
    """
    Run Depth Pro on a single image.

    Returns
    -------
    inv_depth : np.ndarray  shape (H, W)  — inverse depth map (1/m scale)
    fov_deg   : float       — estimated horizontal FOV in degrees
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_r   = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    x       = (img_r.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    x       = jnp.array(x[None])

    inv_d, fov = jit_fn(variables, x)
    return np.array(inv_d[0, ..., 0]), float(fov[0])


def depth_to_colormap(inv_depth: np.ndarray) -> np.ndarray:
    """Convert inverse depth map to a Viridis colour image (uint8 BGR)."""
    d = 1.0 / np.clip(inv_depth, 1e-5, None)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Depth Pro JAX single-image inference")
    p.add_argument("--image",   required=True)
    p.add_argument("--weights", default="weights/depth_pro.msgpack")
    p.add_argument("--output",  default="output/depth_result.png")
    args = p.parse_args()

    jit_fn, variables = load_depth_pro(args.weights)
    inv_d, fov = infer_depth(args.image, jit_fn, variables)
    print(f"  FOV: {fov:.2f}°   inv_depth shape: {inv_d.shape}")

    viz = depth_to_colormap(inv_d)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, viz)
    np.save(args.output.replace(".png", ".npy").replace(".jpg", ".npy"), inv_d)
    print(f"  Saved → {args.output}")
