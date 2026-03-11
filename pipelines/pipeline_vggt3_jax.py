import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
from tqdm import tqdm
import open3d as o3d
from typing import Optional

# Local imports
from models.jax.jax_vggt.models.vggt3 import VGGT3
from models.jax.jax_vggt.utils.pose_utils import pose_encoding_to_extri_intri
from models.jax.jax_reconstruction.utils.geometry import apply_transform

jax.config.update("jax_default_matmul_precision", "default")

"""
VGG-T³: Scalable 3D Reconstruction Pipeline (JAX)
This pipeline uses Linearized Global Attention ($O(n)$) to handle thousands of views.

Citation:
@article{sun2026vggt3,
  title={VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale},
  author={Sun, Aljoša and others},
  journal={arXiv preprint arXiv:2602.23361},
  year={2026}
}
"""

class VGGT3ReconstructionPipeline:
    def __init__(self, vggt_weights: str):
        # 1. Initialize Model
        self.model = VGGT3(img_size=518, patch_size=14, embed_dim=1024)
        
        # 2. Load Weights
        print(f"Loading VGG-T³ weights from {vggt_weights}...")
        with open(vggt_weights, "rb") as f:
            self.variables = serialization.from_bytes(None, f.read())
            
        # 3. JIT compile (Static argnames 'train' for boolean control)
        print("Compiling JAX JIT function (this may take 2-3 minutes for 1B model)...")
        self.jit_apply = jax.jit(self.model.apply, static_argnames=['train'])

    def run(self, image_folder: str, output_path: str = "output/vggt3_reconstruction", window_size: int = 2, stride: int = 1):
        os.makedirs(output_path, exist_ok=True)
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
        
        # Note: VGGT3 is designed for large image collections.
        # Here we use a sliding window for incremental point cloud generation, 
        # but the model itself could process the entire sequence if memory permits.
        
        all_points = []
        all_colors = []
        frame_poses = {0: jnp.eye(4)}
        
        print(f"Processing {len(img_files)} images with VGG-T³ Linear Attention...")
        
        for start_idx in tqdm(range(0, len(img_files) - window_size + 1, stride)):
            end_idx = start_idx + window_size
            window_files = img_files[start_idx:end_idx]
            
            # Load and preprocess
            imgs = []
            for f in window_files:
                img = cv2.imread(os.path.join(image_folder, f))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_vggt = cv2.resize(img_rgb, (518, 518)) / 255.0
                imgs.append(img_vggt.transpose(2, 0, 1))
            
            input_imgs = jnp.array(imgs)[None, ...] # [1, S, 3, 518, 518]
            
            # TTT inference (train=False triggers 2 steps of optimization)
            preds = self.jit_apply(self.variables, input_imgs, train=False)
            
            # Decode poses
            extrinsics, intrinsics = pose_encoding_to_extri_intri(preds['pose_enc_list'][-1], (518, 518))
            
            T_c2l = []
            for s in range(window_size):
                ext = extrinsics[0, s]
                T = jnp.eye(4).at[:3, :].set(ext)
                T_c2l.append(jnp.linalg.inv(T))
            
            T_anchor = frame_poses[start_idx]
            for s in range(1, window_size):
                global_idx = start_idx + s
                T_frame_to_anchor = jnp.linalg.inv(T_c2l[0]) @ T_c2l[s]
                frame_poses[global_idx] = T_anchor @ T_frame_to_anchor

            # Point generation
            target_indices = range(window_size) if start_idx == 0 else [window_size - 1]
            depths = preds['depth'][0][..., 0]
            confs = preds['depth_conf'][0] if 'depth_conf' in preds else None
            
            for s in target_indices:
                h_step, w_step = 4, 4
                y_grid, x_grid = jnp.mgrid[0:518:h_step, 0:518:w_step]
                
                z = depths[s, y_grid, x_grid].flatten()
                mask = z > 0
                if confs is not None:
                    c = confs[s, y_grid, x_grid].flatten()
                    mask &= (c > 0.5)
                
                if not jnp.any(mask): continue
                
                z_masked = z[mask]
                pts_2d = jnp.stack([x_grid.flatten()[mask], y_grid.flatten()[mask]], axis=-1)
                
                K = intrinsics[0, s]
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                
                x_3d = (pts_2d[:, 0] - cx) * z_masked / fx
                y_3d = (pts_2d[:, 1] - cy) * z_masked / fy
                p_3d = jnp.stack([x_3d, y_3d, z_masked], axis=-1)
                
                T_global = frame_poses[start_idx + s]
                p_global = apply_transform(p_3d, T_global[:3, :3], T_global[:3, 3])
                
                img_518 = (imgs[s].transpose(1, 2, 0) * 255).astype(np.uint8)
                colors = img_518[y_grid.flatten()[mask], x_grid.flatten()[mask], :].reshape(-1, 3) / 255.0
                
                all_points.append(p_global)
                all_colors.append(colors)

        if not all_points:
            print("No points generated!")
            return
            
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        o3d.io.write_point_cloud(os.path.join(output_path, "vggt3_reconstruction.ply"), pcd)
        print(f"\nVGG-T³ Reconstruction saved to {output_path}/vggt3_reconstruction.ply")
        print(f"Total points: {len(pcd.points)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VGG-T³ Scalable 3D Reconstruction")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--weights", type=str, default="weights/vggt_1b.msgpack", help="Path to weights")
    parser.add_argument("--output", type=str, default="output/vggt3", help="Output directory")
    args = parser.parse_args()

    pipeline = VGGT3ReconstructionPipeline(args.weights)
    pipeline.run(args.image_dir, output_path=args.output)
