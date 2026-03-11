import jax
import jax.numpy as jnp
from flax import serialization
import cv2
import numpy as np
import os
import sys
import argparse

# Add models/jax to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../models/jax'))
from jax_lightglue.models.superpoint import SuperPoint
from jax_lightglue.models.lightglue import LightGlue

def infer_lightglue(image1_path, image2_path, sp_weights, lg_weights, output_path):
    sp_model = SuperPoint()
    lg_model = LightGlue(n_layers=9)
    
    print(f"Loading weights from {sp_weights} and {lg_weights}...")
    with open(sp_weights, "rb") as f:
        sp_vars = serialization.from_bytes(None, f.read())
    with open(lg_weights, "rb") as f:
        lg_vars = serialization.from_bytes(None, f.read())
        
    def load_img(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        nh, nw = (h // 8) * 8, (w // 8) * 8
        img = cv2.resize(img, (nw, nh))
        return jnp.array(img[None, ..., None] / 255.0), (h, w)

    img1, size1 = load_img(image1_path)
    img2, size2 = load_img(image2_path)
    
    print("Running SuperPoint...")
    jit_sp = jax.jit(sp_model.apply)
    out1 = jit_sp(sp_vars, img1)
    out2 = jit_sp(sp_vars, img2)
    
    def get_kpts(sp_out, k=1024):
        scores = sp_out['scores'][0]
        desc = sp_out['descriptors'][0]
        indices = jnp.argsort(scores.flatten())[::-1][:k]
        y, x = jnp.unravel_index(indices, scores.shape)
        kpts = jnp.stack([x, y], axis=-1)
        iy = jnp.clip((y / 8).astype(jnp.int32), 0, desc.shape[0]-1)
        ix = jnp.clip((x / 8).astype(jnp.int32), 0, desc.shape[1]-1)
        feat = desc[iy, ix, :]
        return kpts, feat

    kpts1, feat1 = get_kpts(out1)
    kpts2, feat2 = get_kpts(out2)
    
    print("Running LightGlue...")
    jit_lg = jax.jit(lg_model.apply)
    lg_input = {
        "image0": {"keypoints": kpts1[None], "descriptors": feat1[None]},
        "image1": {"keypoints": kpts2[None], "descriptors": feat2[None]}
    }
    lg_out = jit_lg(lg_vars, lg_input)
    
    scores = lg_out['scores'][0, :-1, :-1]
    m0 = jnp.argmax(scores, axis=1)
    m1 = jnp.argmax(scores, axis=0)
    mutual = (jnp.arange(len(m0)) == m1[m0])
    valid = mutual & (jnp.exp(jnp.max(scores, axis=1)) > 0.1)
    idx0, idx1 = jnp.where(valid)[0], m0[jnp.where(valid)[0]]
    
    print(f"Found {len(idx0)} matches.")
    
    img1_rgb = cv2.imread(image1_path)
    img2_rgb = cv2.imread(image2_path)
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    kpts1_orig = np.array(kpts1[idx0]) * np.array([w1 / img1.shape[2], h1 / img1.shape[1]])
    kpts2_orig = np.array(kpts2[idx1]) * np.array([w2 / img2.shape[2], h2 / img2.shape[1]])
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1], canvas[:h2, w1:] = img1_rgb, img2_rgb
    for p1, p2 in zip(kpts1_orig, kpts2_orig):
        p1 = tuple(p1.astype(int))
        p2 = tuple((p2 + np.array([w1, 0])).astype(int))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.circle(canvas, p1, 2, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 2, (0, 0, 255), -1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Match visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperPoint + LightGlue JAX Inference")
    parser.add_argument("--img1", type=str, required=True, help="Path to image 1")
    parser.add_argument("--img2", type=str, required=True, help="Path to image 2")
    parser.add_argument("--sp_weights", type=str, default="weights/superpoint.msgpack", help="Path to SuperPoint weights")
    parser.add_argument("--lg_weights", type=str, default="weights/superpoint_lightglue.msgpack", help="Path to LightGlue weights")
    parser.add_argument("--output", type=str, default="output/match_result.jpg", help="Path to output visualization")
    args = parser.parse_args()
    infer_lightglue(args.img1, args.img2, args.sp_weights, args.lg_weights, args.output)
