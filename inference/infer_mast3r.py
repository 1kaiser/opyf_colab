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
from jax_mast3r.models.mast3r import FlaxAsymmetricMASt3R

def infer_mast3r(image1_path, image2_path, weights_path, output_dir):
    # Initialize model with correct params based on weights
    model = FlaxAsymmetricMASt3R()
    
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, "rb") as f:
        data = serialization.msgpack_restore(f.read())
    
    # Wrap in 'params' for Flax
    variables = {'params': data}
        
    def load_img(path):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MAST3R expects 512x512
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_input = (img_resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        return jnp.array(img_input[None, ...])

    img1 = load_img(image1_path)
    img2 = load_img(image2_path)
    
    print("Running inference...")
    jit_apply = jax.jit(model.apply)
    # MAST3R call: res1, res2, debug = model(img1, img2)
    res1, res2, _ = jit_apply(variables, img1, img2)
    
    pts3d1, conf1, desc1, desc_conf1 = res1
    pts3d2, conf2, desc2, desc_conf2 = res2
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "pts3d1.npy"), np.array(pts3d1))
    np.save(os.path.join(output_dir, "pts3d2.npy"), np.array(pts3d2))
    np.save(os.path.join(output_dir, "desc1.npy"), np.array(desc1))
    np.save(os.path.join(output_dir, "desc2.npy"), np.array(desc2))
    
    # Optional: Save visualization of confidence
    conf1_np = np.array(conf1[0])
    conf1_viz = (conf1_np - conf1_np.min()) / (conf1_np.max() - conf1_np.min())
    conf1_viz = (conf1_viz * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "conf1.jpg"), cv2.applyColorMap(conf1_viz, cv2.COLORMAP_JET))
    
    print(f"MAST3R results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAST3R JAX Inference")
    parser.add_argument("--img1", type=str, required=True, help="Path to image 1")
    parser.add_argument("--img2", type=str, required=True, help="Path to image 2")
    parser.add_argument("--weights", type=str, default="/home/kaiser/gemini_project2/weights/mast3r_full.msgpack", help="Path to weights")
    parser.add_argument("--output", type=str, default="output/mast3r_results", help="Output directory")
    args = parser.parse_args()
    
    infer_mast3r(args.img1, args.img2, args.weights, args.output)
