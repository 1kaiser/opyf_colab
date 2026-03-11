import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
from tqdm import tqdm
from models.jax.jax_vggt.models.vggt import VGGT

jax.config.update("jax_default_matmul_precision", "highest")

def run_vggt_jax():
    # 1. Setup Model
    model = VGGT(
        img_size=518, 
        patch_size=14, 
        embed_dim=1024
    )
    
    # 2. Load Weights
    weights_path = "weights/vggt/vggt_1b.msgpack"
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Run convert_weights.py first.")
        return
        
    with open(weights_path, "rb") as f:
        variables = serialization.from_bytes(None, f.read())
        
    # 3. Load Images
    image_dir = "data/pinecone_subset"
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    img_files = img_files[:2] # Just two frames for test
    
    images_list = []
    for f in img_files:
        img = cv2.imread(os.path.join(image_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (518, 518))
        img = img / 255.0 # VGGT expects [0, 1]
        images_list.append(img)
        
    images = jnp.array(images_list)[None, ...] # [B, S, 3, H, W]
    # Wait, VGGT expects [B, S, 3, H, W] or [S, 3, H, W]
    # My images_list is [S, H, W, 3]
    images = jnp.array(images_list).transpose(0, 3, 1, 2)[None, ...]
    
    # 4. Run Inference
    print("Running VGGT JAX inference...")
    jit_apply = jax.jit(model.apply)
    preds = jit_apply(variables, images)
    
    # 5. Inspect Results
    print("Predictions keys:", preds.keys())
    if "depth" in preds:
        print("Depth shape:", preds["depth"].shape)
    if "pose_enc" in preds:
        print("Pose shape:", preds["pose_enc"].shape)
        
    # Save a depth map visualization
    if "depth" in preds:
        depth = np.array(preds["depth"][0, 0, ..., 0])
        # Normalize for vis
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite("output/vggt_depth_test.png", cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS))
        print("Saved depth map to output/vggt_depth_test.png")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    run_vggt_jax()
