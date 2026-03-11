import numpy as np
import os

def compare(name, pt, jax):
    # PyTorch: [B, C, H, W]
    # JAX: [B, H, W, C]
    if pt.shape != jax.shape:
        # try to transpose jax
        if jax.ndim == 4:
            jax = jax.transpose(0, 3, 1, 2)
            
    if pt.shape != jax.shape:
        print(f"{name}: SHAPE MISMATCH! PT: {pt.shape}, JAX: {jax.shape}")
        return

    mse = np.mean((pt - jax)**2)
    max_diff = np.max(np.abs(pt - jax))
    print(f"{name}: MSE: {mse:.2e}, Max Diff: {max_diff:.2e}")

def compare_depth_pro():
    # Encodings
    for i in range(5):
        pt_enc = np.load(f"output/depth_pro_parity/pt_enc_{i}.npy")
        jax_enc = np.load(f"output/depth_pro_parity/jax_enc_{i}.npy")
        compare(f"Encoding {i}", pt_enc, jax_enc)

    # x_down
    pt_xd = np.load("output/depth_pro_parity/pt_x_down.npy")
    jax_xd = np.load("output/depth_pro_parity/jax_x_down.npy")
    compare("x_down", pt_xd, jax_xd)

    # Lowres Features
    pt_low = np.load("output/depth_pro_parity/pt_lowres_features.npy")
    jax_low = np.load("output/depth_pro_parity/jax_lowres_features.npy")
    compare("Lowres Features", pt_low, jax_low)

    # FOV Granular
    pt_f = np.load("output/depth_pro_parity/pt_fov_feat.npy")
    jax_f = np.load("output/depth_pro_parity/jax_fov_feat.npy")
    compare("FOV Feat", pt_f, jax_f)
    
    pt_l = np.load("output/depth_pro_parity/pt_fov_low.npy")
    jax_l = np.load("output/depth_pro_parity/jax_fov_low.npy")
    compare("FOV Low", pt_l, jax_l)

    # Canonical Inv Depth
    pt = np.load("output/depth_pro_parity/pt_canonical_inv_depth.npy")
    jax_val = np.load("output/depth_pro_parity/jax_canonical_inv_depth.npy")
    compare("Canonical Inv Depth", pt, jax_val)
    
    # FOV Deg
    pt_fov = np.load("output/depth_pro_parity/pt_fov_deg.npy")
    jax_fov = np.load("output/depth_pro_parity/jax_fov_deg.npy")
    compare("FOV Deg", pt_fov, jax_fov)

if __name__ == "__main__":
    compare_depth_pro()
