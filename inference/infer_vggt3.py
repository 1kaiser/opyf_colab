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
from jax_vggt.models.vggt3 import VGGT3

def infer_vggt3(image_dir, weights_path, output_dir):
    model = VGGT3(img_size=518, patch_size=14, embed_dim=1024)
    
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
        
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])[:2]
    if len(img_files) < 2:
        print("Need at least 2 images for VGG-T³ inference.")
        return
        
    imgs = []
    for f in img_files:
        img = cv2.imread(os.path.join(image_dir, f))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_vggt = cv2.resize(img_rgb, (518, 518)) / 255.0
        imgs.append(img_vggt.transpose(2, 0, 1))
    
    input_imgs = jnp.array(imgs)[None, ...] 
    
    print("Running VGG-T³ inference (TTT steps included)...")
    jit_apply = jax.jit(model.apply, static_argnames=['train'])
    preds = jit_apply(variables, input_imgs, train=False)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "world_points.npy"), np.array(preds['world_points'][0]))
    np.save(os.path.join(output_dir, "depth.npy"), np.array(preds['depth'][0]))
    np.save(os.path.join(output_dir, "pose_enc.npy"), np.array(preds['pose_enc']))
    print(f"VGG-T³ results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG-T³ JAX Inference")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--weights", type=str, default="/home/kaiser/gemini_project2/weights/vggt_1b.msgpack", help="Path to weights")
    parser.add_argument("--output", type=str, default="output/vggt3_results", help="Output directory")
    args = parser.parse_args()
    
    infer_vggt3(args.image_dir, args.weights, args.output)
