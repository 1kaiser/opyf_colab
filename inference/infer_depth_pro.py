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
from jax_depth_pro.models.depth_pro import DepthPro

def infer_depth_pro(image_path, weights_path, output_path):
    vit_config = {
        'img_size': 384, 'patch_size': 16, 'embed_dim': 1024, 
        'depth': 24, 'num_heads': 16, 'init_values': 1e-5
    }
    model = DepthPro(vit_config=vit_config)
    
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
        
    print(f"Loading image from: {os.path.abspath(image_path)}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not read image at {os.path.abspath(image_path)}. Check if the file exists.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_size = 1536
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    img_input = (img_resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    img_input = jnp.array(img_input[None, ...]) 
    
    print("Running inference...")
    jit_apply = jax.jit(model.apply)
    inv_depth, fov = jit_apply(variables, img_input)
    
    depth = 1.0 / jnp.clip(inv_depth[0, ..., 0], min=1e-5)
    fov_deg = fov[0]
    print(f"Estimated FOV: {fov_deg:.2f} degrees")
    
    depth_np = np.array(depth)
    depth_rescaled = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    depth_viz = (depth_rescaled * 255).astype(np.uint8)
    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_VIRIDIS)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, depth_viz)
    print(f"Depth visualization saved to {output_path}")
    np.save(output_path.replace('.jpg', '.npy').replace('.png', '.npy'), depth_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Pro JAX Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/depth_pro.msgpack", help="Path to weights")
    parser.add_argument("--output", type=str, default="output/depth_pro_result.jpg", help="Path to output visualization")
    args = parser.parse_args()
    infer_depth_pro(args.image, args.weights, args.output)
