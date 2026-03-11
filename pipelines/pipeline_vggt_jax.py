import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
from tqdm import tqdm
import open3d as o3d

# Local imports
from models.jax.jax_vggt.models.vggt import VGGT
from models.jax.jax_vggt.utils.pose_utils import pose_encoding_to_extri_intri
from models.jax.jax_reconstruction.utils.geometry import apply_transform

jax.config.update("jax_default_matmul_precision", "default")

class VGGTReconstructionPipeline:
    def __init__(self, vggt_weights):
        # 1. Initialize Model
        self.model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
        
        # 2. Load Weights
        with open(vggt_weights, "rb") as f:
            self.variables = serialization.from_bytes(None, f.read())
            
        # 3. JIT compile
        self.jit_apply = jax.jit(self.model.apply)

    def run(self, image_folder, output_path="output/vggt_reconstruction", window_size=2, stride=1):
        os.makedirs(output_path, exist_ok=True)
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        
        # Limit for test
        img_files = img_files[:5]
        
        all_points = []
        all_colors = []
        
        # We'll store the absolute pose of each frame
        frame_poses = {0: jnp.eye(4)}
        
        print(f"Processing {len(img_files)} images with window_size={window_size}...")
        
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
            preds = self.jit_apply(self.variables, input_imgs)
            
            # Decode poses
            extrinsics, intrinsics = pose_encoding_to_extri_intri(preds['pose_enc_list'][-1], (518, 518))
            
            # T_cam_to_local_origin (local_origin is the first frame of the window)
            T_c2l = []
            for s in range(window_size):
                ext = extrinsics[0, s]
                T = jnp.eye(4).at[:3, :].set(ext)
                T_c2l.append(jnp.linalg.inv(T))
            
            # The first frame of this window (start_idx) should match its already computed global pose
            T_anchor = frame_poses[start_idx]
            
            # Update subsequent frames in this window
            for s in range(1, window_size):
                global_idx = start_idx + s
                T_frame_to_anchor = jnp.linalg.inv(T_c2l[0]) @ T_c2l[s]
                T_global = T_anchor @ T_frame_to_anchor
                frame_poses[global_idx] = T_global

            # Generate points for the NEWEST frame in this window to avoid redundant points
            target_indices = range(window_size) if start_idx == 0 else [window_size - 1]
            
            depths = preds['depth'][0]
            if depths.ndim == 4:
                depths = depths[..., 0]
            
            confs = preds['depth_conf'][0] if 'depth_conf' in preds else None
            if confs is not None and confs.ndim == 4:
                confs = confs[..., 0]
            
            for s in target_indices:
                h_step, w_step = 4, 4
                y_grid, x_grid = jnp.mgrid[0:518:h_step, 0:518:w_step]
                
                # Sample depth and confidence
                z = depths[s, y_grid, x_grid].flatten()
                if confs is not None:
                    c = confs[s, y_grid, x_grid].flatten()
                    mask = c > 0.5 
                else:
                    mask = z > 0
                
                if not jnp.any(mask): continue
                
                z = z[mask]
                pts_2d = jnp.stack([x_grid.flatten()[mask], y_grid.flatten()[mask]], axis=-1)
                
                K = intrinsics[0, s]
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                
                x_3d = (pts_2d[:, 0] - cx) * z / fx
                y_3d = (pts_2d[:, 1] - cy) * z / fy
                p_3d = jnp.stack([x_3d, y_3d, z], axis=-1)
                
                # Transform to global
                T_global = frame_poses[start_idx + s]
                p_global = apply_transform(p_3d, T_global[:3, :3], T_global[:3, 3])
                
                # Color
                img_518 = (imgs[s].transpose(1, 2, 0) * 255).astype(np.uint8)
                colors = img_518[y_grid.flatten()[mask], x_grid.flatten()[mask], :].reshape(-1, 3) / 255.0
                
                all_points.append(p_global)
                all_colors.append(colors)

        # Fusion and Save
        if not all_points:
            print("No points generated!")
            return
            
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))
        
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        o3d.io.write_point_cloud(os.path.join(output_path, "vggt_reconstruction.ply"), pcd)
        print(f"\nVGGT Reconstruction saved to {output_path}/vggt_reconstruction.ply")
        print(f"Total points: {len(pcd.points)}")

if __name__ == "__main__":
    pipeline = VGGTReconstructionPipeline("weights/vggt/vggt_1b.msgpack")
    pipeline.run("data/pinecone_subset")
