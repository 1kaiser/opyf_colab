"""
modules/infer_stereo.py
========================
MASt3R JAX inference — dense stereo: 3D point clouds + feature descriptors.

Public API
----------
    load_mast3r(weights_path) -> (jit_fn, variables)
    infer_stereo(img1_path, img2_path, jit_fn, variables) -> dict
        keys: pts3d1, pts3d2, conf1, conf2, desc1, desc2

CLI
---
    python3 modules/infer_stereo.py --img1 a.jpg --img2 b.jpg
        --weights weights/mast3r_full.msgpack --output output/stereo
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
from jax_mast3r.models.mast3r import FlaxAsymmetricMASt3R

_INPUT_SIZE = 512


def load_mast3r(weights_path: str):
    """Load MASt3R weights. Returns (jit_fn, variables)."""
    model = FlaxAsymmetricMASt3R()
    print(f"  Loading MASt3R from {weights_path} …")
    with open(weights_path, "rb") as f:
        data = serialization.msgpack_restore(f.read())
    variables = {"params": data}
    jit_fn = jax.jit(model.apply)
    print("  MASt3R ready.")
    return jit_fn, variables


def _preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))
    x   = (rgb.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    return jnp.array(x[None])


def infer_stereo(img1_path: str, img2_path: str, jit_fn, variables) -> dict:
    """
    Run MASt3R dense stereo on an image pair.

    Returns dict with keys:
        pts3d1, pts3d2  : (1, H, W, 3) 3D point clouds
        conf1,  conf2   : (1, H, W)    confidence maps
        desc1,  desc2   : (1, H, W, D) feature descriptor maps
    """
    img1 = _preprocess(img1_path)
    img2 = _preprocess(img2_path)
    res1, res2, _ = jit_fn(variables, img1, img2)

    pts3d1, conf1, desc1, _ = res1
    pts3d2, conf2, desc2, _ = res2

    return {
        "pts3d1": np.array(pts3d1),
        "pts3d2": np.array(pts3d2),
        "conf1":  np.array(conf1),
        "conf2":  np.array(conf2),
        "desc1":  np.array(desc1),
        "desc2":  np.array(desc2),
    }


def _conf_viz(conf: np.ndarray) -> np.ndarray:
    c = conf[0]
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    return cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MASt3R JAX dense stereo inference")
    p.add_argument("--img1",    required=True)
    p.add_argument("--img2",    required=True)
    p.add_argument("--weights", default="weights/mast3r_full.msgpack")
    p.add_argument("--output",  default="output/stereo")
    args = p.parse_args()

    jit_fn, variables = load_mast3r(args.weights)
    results = infer_stereo(args.img1, args.img2, jit_fn, variables)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    for key in ["pts3d1", "pts3d2", "desc1", "desc2"]:
        np.save(os.path.join(args.output, f"{key}.npy"), results[key])
    cv2.imwrite(os.path.join(args.output, "conf1.jpg"), _conf_viz(results["conf1"]))
    cv2.imwrite(os.path.join(args.output, "conf2.jpg"), _conf_viz(results["conf2"]))
    print(f"  Results saved → {args.output}")
    print(f"  pts3d1 shape: {results['pts3d1'].shape}")
