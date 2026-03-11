import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import open3d as o3d

# Local imports
from models.jax.jax_depth_pro.models.depth_pro import DepthPro
from models.jax.jax_lightglue.models.superpoint import SuperPoint
from models.jax.jax_lightglue.models.lightglue import LightGlue
from models.jax.jax_reconstruction.utils.geometry import lift_points, kabsch_alignment, apply_transform

jax.config.update("jax_default_matmul_precision", "default")

class ReconstructionPipeline:
    def __init__(self, depth_pro_weights, sp_weights, lg_weights):
        print(f"Loading weights from {depth_pro_weights}, {sp_weights}, {lg_weights}...")
        # Initialize Models
        self.depth_model = DepthPro(vit_config={
            'img_size': 384, 'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'init_values': 1e-5
        })
        self.sp_model = SuperPoint()
        self.lg_model = LightGlue(n_layers=9)
        
        self.variables = {}
        print("Loading Depth Pro weights...")
        with open(depth_pro_weights, "rb") as f:
            self.variables['depth'] = serialization.from_bytes(None, f.read())
        print("Loading SuperPoint weights...")
        with open(sp_weights, "rb") as f:
            self.variables['sp'] = serialization.from_bytes(None, f.read())
        print("Loading LightGlue weights...")
        with open(lg_weights, "rb") as f:
            self.variables['lg'] = serialization.from_bytes(None, f.read())
            
        print("Compiling JIT functions (this may take a few minutes)...")
        self.jit_depth = jax.jit(self.depth_model.apply)
        self.jit_sp = jax.jit(self.sp_model.apply)
        self.jit_lg = jax.jit(self.lg_model.apply)
        print("Pipeline initialized.")

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_size = 1536
        input_depth = cv2.resize(img_rgb, (target_size, target_size))
        input_depth = (input_depth.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        input_depth = jnp.array(input_depth[None, ...])
        inv_depth, fov = self.jit_depth(self.variables['depth'], input_depth)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_resized = cv2.resize(img_gray, (target_size, target_size))
        input_sp = jnp.array(img_gray_resized[None, ..., None] / 255.0)
        sp_out = self.jit_sp(self.variables['sp'], input_sp)
        return {
            'img_rgb': img_rgb,
            'inv_depth': inv_depth[0, ..., 0], 
            'fov': fov[0],
            'sp_scores': sp_out['scores'][0],
            'sp_desc': sp_out['descriptors'][0]
        }

    def get_concentric_zones(self, scores, desc, num_zones=3, k=3072):
        H, W = scores.shape
        cy, cx = H / 2.0, W / 2.0
        max_r = math.sqrt(cy**2 + cx**2)
        overlap = 0.20
        w = 1.0 / (num_zones - (num_zones - 1) * overlap)
        zones = []
        indices = jnp.argsort(scores.flatten())[::-1][:k*2]
        y, x = jnp.unravel_index(indices, scores.shape)
        dist = jnp.sqrt((y - cy)**2 + (x - cx)**2) / max_r
        for i in range(num_zones):
            r_start = i * w * (1 - overlap)
            r_end = r_start + w
            mask = (dist >= r_start) & (dist <= r_end)
            z_y = y[mask][:k//num_zones]
            z_x = x[mask][:k//num_zones]
            z_kpts = jnp.stack([z_x, z_y], axis=-1)
            iy = jnp.clip((z_y / 8).astype(jnp.int32), 0, desc.shape[0]-1)
            ix = jnp.clip((z_x / 8).astype(jnp.int32), 0, desc.shape[1]-1)
            zones.append((z_kpts, desc[iy, ix, :], (r_start, r_end)))
        return zones

    def run(self, image_folder, output_path="output/reconstruction", max_kpts=3072, num_zones=3, radial_clip=0.70):
        os.makedirs(output_path, exist_ok=True)
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        img_files = img_files[:10]
        all_points, all_colors = [], []
        T_globals = {img_files[0]: [jnp.eye(4) for _ in range(num_zones)]}
        prev_data = self.process_image(os.path.join(image_folder, img_files[0]))
        prev_data['zones'] = self.get_concentric_zones(prev_data['sp_scores'], prev_data['sp_desc'], num_zones=num_zones, k=max_kpts)
        
        for i in tqdm(range(1, len(img_files)), desc=f"Reconstructing ({num_zones} Zones, Clip={radial_clip})"):
            img_name = img_files[i]
            curr_data = self.process_image(os.path.join(image_folder, img_name))
            curr_data['zones'] = self.get_concentric_zones(curr_data['sp_scores'], curr_data['sp_desc'], num_zones=num_zones, k=max_kpts)
            current_frame_poses = []
            print(f"\nFrame {i} Zonal Registration:")
            for z_idx in range(num_zones):
                kpts0, desc0, _ = prev_data['zones'][z_idx]
                kpts1, desc1, _ = curr_data['zones'][z_idx]
                if len(kpts0) < 10 or len(kpts1) < 10:
                    current_frame_poses.append(T_globals[img_files[i-1]][z_idx])
                    continue
                lg_out = self.jit_lg(self.variables['lg'], {"image0": {"keypoints": kpts0[None], "descriptors": desc0[None]}, "image1": {"keypoints": kpts1[None], "descriptors": desc1[None]}})
                scores = lg_out['scores'][0, :-1, :-1]
                m0 = jnp.argmax(scores, axis=1)
                m1 = jnp.argmax(scores, axis=0)
                mutual = (jnp.arange(len(m0)) == m1[m0])
                valid = mutual & (jnp.exp(jnp.max(scores, axis=1)) > 0.1)
                idx0, idx1 = jnp.where(valid)[0], m0[jnp.where(valid)[0]]
                if len(idx0) > 8:
                    p0_3d = lift_points(kpts0[idx0], prev_data['inv_depth'], prev_data['fov'])
                    p1_3d = lift_points(kpts1[idx1], curr_data['inv_depth'], curr_data['fov'])
                    R, t = kabsch_alignment(p0_3d, p1_3d)
                    T_z_glob = T_globals[img_files[i-1]][z_idx] @ jnp.linalg.inv(jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(t))
                    rmse = jnp.sqrt(jnp.mean(jnp.sum((apply_transform(p0_3d, R, t) - p1_3d)**2, axis=1)))
                    print(f"  Zone {z_idx}: RMSE={rmse:.6f}, Matches={len(idx0)}")
                else:
                    T_z_glob = T_globals[img_files[i-1]][z_idx]
                current_frame_poses.append(T_z_glob)
            T_globals[img_name] = current_frame_poses
            h_step, w_step = 8, 8
            y_grid, x_grid = jnp.mgrid[0:1536:h_step, 0:1536:w_step]
            cy, cx = 1536/2.0, 1536/2.0
            max_r = math.sqrt(cy**2 + cx**2)
            dist_grid = jnp.sqrt((y_grid - cy)**2 + (x_grid - cx)**2) / max_r
            rgb_sub = cv2.resize(curr_data['img_rgb'], (1536, 1536))[::h_step, ::w_step, :]
            for z_idx in range(num_zones):
                r_start, r_end = curr_data['zones'][z_idx][2]
                actual_r_end = min(r_end, radial_clip)
                if r_start >= actual_r_end: continue
                z_mask = (dist_grid >= r_start) & (dist_grid <= actual_r_end)
                z_kpts = jnp.stack([x_grid[z_mask], y_grid[z_mask]], axis=-1)
                if len(z_kpts) == 0: continue
                p_global = apply_transform(lift_points(z_kpts, curr_data['inv_depth'], curr_data['fov']), current_frame_poses[z_idx][:3, :3], current_frame_poses[z_idx][:3, 3])
                all_points.append(p_global)
                all_colors.append(rgb_sub[z_mask.astype(bool)].reshape(-1, 3) / 255.0)
            prev_data = curr_data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
        o3d.io.write_point_cloud(os.path.join(output_path, f"point_cloud_clip{radial_clip}.ply"), pcd)
        print("Generating mesh...")
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        o3d.io.write_triangle_mesh(os.path.join(output_path, f"mesh_clip{radial_clip}.glb"), mesh)
        print(f"Reconstruction (Clip={radial_clip}) saved to {output_path}")
        return T_globals

if __name__ == "__main__":
    pipeline = ReconstructionPipeline("weights/depth_pro.msgpack", "weights/superpoint.msgpack", "weights/superpoint_lightglue.msgpack")
    pipeline.run("data/pinecone_subset", radial_clip=0.70)
