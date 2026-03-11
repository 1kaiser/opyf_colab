
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_depth_map(pts, filename):
    """Saves a depth map visualization from a points array."""
    # Squeeze the batch dimension if it exists
    if pts.ndim == 4:
        pts = pts.squeeze(0)
        
    depth = pts[..., 2]  # Extract Z channel
    
    # Normalize depth for visualization
    d_min = depth.min()
    d_max = depth.max()
    if d_max > d_min:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth)
        
    # Convert to colormapped image
    cmap = plt.get_cmap('viridis')
    depth_colored = (cmap(depth_norm) * 255).astype(np.uint8)
    
    Image.fromarray(depth_colored).save(filename)
    print(f"Saved depth map visualization to {filename}")

def save_diff_map(pts1, pts2, filename):
    """Saves a visualization of the difference between two depth maps."""
    if pts1.ndim == 4:
        pts1 = pts1.squeeze(0)
    if pts2.ndim == 4:
        pts2 = pts2.squeeze(0)
        
    depth1 = pts1[..., 2]
    depth2 = pts2[..., 2]
    
    diff = np.abs(depth1 - depth2)
    
    d_min = diff.min()
    d_max = diff.max()
    if d_max > d_min:
        diff_norm = (diff - d_min) / (d_max - d_min)
    else:
        diff_norm = np.zeros_like(diff)
        
    cmap = plt.get_cmap('inferno')
    diff_colored = (cmap(diff_norm) * 255).astype(np.uint8)
    
    Image.fromarray(diff_colored).save(filename)
    print(f"Saved difference map visualization to {filename}")


def compare(name, pt_path, jax_path):
    pt = np.load(pt_path)
    jax = np.load(jax_path)
    
    print(f"Comparing {name}:")
    print(f"  PyTorch shape: {pt.shape}")
    print(f"  JAX shape: {jax.shape}")
    
    if pt.shape != jax.shape:
        print(f"  ERROR: Shapes mismatch!")
        return

    mse = np.mean((pt - jax)**2)
    max_diff = np.max(np.abs(pt - jax))
    
    # Calculate PSNR
    # For data where range isn't explicitly 0-255 or 0-1, we use the peak-to-peak of the reference
    data_range = np.max(pt) - np.min(pt)
    if mse > 0:
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    print(f"  MSE: {mse:.2e}")
    print(f"  Max Diff: {max_diff:.2e}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    flat_pt = pt.reshape(-1, pt.shape[-1])
    flat_jax = jax.reshape(-1, jax.shape[-1])
    print(f"  PyTorch[:5]:\n{flat_pt[:5]}")
    print(f"  JAX[:5]:\n{flat_jax[:5]}")
    print(f"  PyTorch[-5:]:\n{flat_pt[-5:]}")
    print(f"  JAX[-5:]:\n{flat_jax[-5:]}")
    
    if max_diff < 1e-4:
        print(f"  SUCCESS: {name} matches well!")
    else:
        print(f"  WARNING: {name} has significant differences.")

def main():
    # Create output directory for visualizations
    vis_dir = "output/visual_comparison"
    os.makedirs(vis_dir, exist_ok=True)

    # --- Numerical Comparison ---
    print("--- Running Numerical Comparison ---")
    compare("img1 (Input Side 1)", "output/pytorch_inference/img1.npy", "output/jax_inference/img1.npy")
    compare("img2 (Input Side 2)", "output/pytorch_inference/img2.npy", "output/jax_inference/img2.npy")
    compare("feat1 (Encoder Side 1)", "output/pytorch_inference/feat1.npy", "output/jax_inference/feat1.npy")
    compare("feat2 (Encoder Side 2)", "output/pytorch_inference/feat2.npy", "output/jax_inference/feat2.npy")

    for i in range(12):
        pt_path1 = f"output/pytorch_inference/decoder_blocks/dec1_blk_{i}.npy"
        jax_path1 = f"output/jax_inference/decoder_blocks/dec1_blk_{i}.npy"
        if i == 11:
            # Special case for Layer 11: PyTorch is unnormalized, JAX is normalized
            pt = np.load(pt_path1)
            jax_val = np.load(jax_path1)
            # ROUGH manual norm
            mean = pt.mean(axis=-1, keepdims=True)
            var = pt.var(axis=-1, keepdims=True)
            pt_norm = (pt - mean) / np.sqrt(var + 1e-5)
            # Just print first element to see if they align roughly
            print(f"Decoder Block 1, Layer 11 (MANUAL NORM ROUGH):")
            print(f"  PyTorch norm[:1,0, :5]: {pt_norm[0,0, :5]}")
            print(f"  JAX[:1,0, :5]: {jax_val[0,0, :5]}")
        else:
            compare(f"Decoder Block 1, Layer {i}", pt_path1, jax_path1)
        
        pt_path2 = f"output/pytorch_inference/decoder_blocks/dec2_blk_{i}.npy"
        jax_path2 = f"output/jax_inference/decoder_blocks/dec2_blk_{i}.npy"
        if i < 11:
            compare(f"Decoder Block 2, Layer {i}", pt_path2, jax_path2)

    compare("dpt_act0_1 (DPT Side 1 Act 0)", "output/pytorch_inference/dpt_act0_1.npy", "output/jax_inference/dpt_act0_1.npy")
    compare("dpt_act0_2 (DPT Side 2 Act 0)", "output/pytorch_inference/dpt_act0_2.npy", "output/jax_inference/dpt_act0_2.npy")
    
    pt_pts1 = np.load("output/pytorch_inference/pts1.npy")
    jax_pts1 = np.load("output/jax_inference/pts1.npy")
    compare("pts1 (Final Pts Side 1)", "output/pytorch_inference/pts1.npy", "output/jax_inference/pts1.npy")
    
    pt_pts2 = np.load("output/pytorch_inference/pts2.npy")
    jax_pts2 = np.load("output/jax_inference/pts2.npy")
    compare("pts2 (Final Pts Side 2)", "output/pytorch_inference/pts2.npy", "output/jax_inference/pts2.npy")

    compare("desc1 (Final Desc Side 1)", "output/pytorch_inference/desc1.npy", "output/jax_inference/desc1.npy")
    compare("desc2 (Final Desc Side 2)", "output/pytorch_inference/desc2.npy", "output/jax_inference/desc2.npy")
    print("-" * 20)

    # --- Visual Comparison ---
    print("\n--- Running Visual Comparison ---")
    save_depth_map(pt_pts1, os.path.join(vis_dir, "depth_pt1.png"))
    save_depth_map(jax_pts1, os.path.join(vis_dir, "depth_jax1.png"))
    save_diff_map(pt_pts1, jax_pts1, os.path.join(vis_dir, "depth_diff1.png"))
    
    save_depth_map(pt_pts2, os.path.join(vis_dir, "depth_pt2.png"))
    save_depth_map(jax_pts2, os.path.join(vis_dir, "depth_jax2.png"))
    save_diff_map(pt_pts2, jax_pts2, os.path.join(vis_dir, "depth_diff2.png"))


if __name__ == "__main__":
    main()
