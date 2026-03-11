import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
from pipelines.pipelines.pipeline_jax import ReconstructionPipeline
from models.jax.jax_reconstruction.utils.geometry import lift_points, kabsch_alignment, apply_transform

class SegmentationAnalyzer(ReconstructionPipeline):
    def analyze_edge_distortion(self, image_folder, num_tiles=4, overlap=0.40):
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        data0 = self.process_image(os.path.join(image_folder, img_files[0]))
        data1 = self.process_image(os.path.join(image_folder, img_files[1]))
        H, W = 1536, 1536
        tile_size = int(H / (num_tiles - (num_tiles - 1) * overlap))
        stride = int(tile_size * (1 - overlap))
        tile_poses = {}
        print(f"Computing poses for {num_tiles}x{num_tiles} segments...")
        for r in range(num_tiles):
            for c in range(num_tiles):
                y0, x0 = r * stride, c * stride
                y1, x1 = y0 + tile_size, x0 + tile_size
                def get_tile_data(scores, desc):
                    flat_idx = jnp.argsort(scores.flatten())[::-1][:5000]
                    sy, sx = jnp.unravel_index(flat_idx, scores.shape)
                    mask = (sy >= y0) & (sy < y1) & (sx >= x0) & (sx < x1)
                    kpts = jnp.stack([sx[mask], sy[mask]], axis=-1)[:512]
                    iy = jnp.clip((sy[mask][:512] / 8).astype(jnp.int32), 0, desc.shape[0]-1)
                    ix = jnp.clip((sx[mask][:512] / 8).astype(jnp.int32), 0, desc.shape[1]-1)
                    return kpts, desc[iy, ix, :]
                k0, d0 = get_tile_data(data0['sp_scores'], data0['sp_desc'])
                k1, d1 = get_tile_data(data1['sp_scores'], data1['sp_desc'])
                if len(k0) < 10 or len(k1) < 10: continue
                lg_out = self.jit_lg(self.variables['lg'], {"image0": {"keypoints": k0[None], "descriptors": d0[None]}, "image1": {"keypoints": k1[None], "descriptors": d1[None]}})
                scores = lg_out['scores'][0, :-1, :-1]
                m0 = jnp.argmax(scores, axis=1)
                m1 = jnp.argmax(scores, axis=0)
                mutual = (jnp.arange(len(m0)) == m1[m0])
                valid = mutual & (jnp.exp(jnp.max(scores, axis=1)) > 0.1)
                idx0, idx1 = jnp.where(valid)[0], m0[jnp.where(valid)[0]]
                if len(idx0) > 8:
                    p0_3d = lift_points(k0[idx0], data0['inv_depth'], data0['fov'])
                    p1_3d = lift_points(k1[idx1], data1['inv_depth'], data1['fov'])
                    R, t = kabsch_alignment(p0_3d, p1_3d)
                    tile_poses[(r, c)] = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(t)
        distortions = []
        print("Analyzing edge discontinuities...")
        for r in range(num_tiles):
            for c in range(num_tiles - 1):
                if (r, c) in tile_poses and (r, c+1) in tile_poses:
                    edge_x = int(c * stride + tile_size - (tile_size * overlap / 2))
                    y_start, y_end = r * stride, r * stride + tile_size
                    edge_y = jnp.linspace(y_start, y_end, 10)
                    edge_kpts = jnp.stack([jnp.full_like(edge_y, edge_x), edge_y], axis=-1)
                    p_3d = lift_points(edge_kpts, data0['inv_depth'], data0['fov'])
                    T_left, T_right = tile_poses[(r, c)], tile_poses[(r, c+1)]
                    p_left = apply_transform(p_3d, T_left[:3, :3], T_left[:3, 3])
                    p_right = apply_transform(p_3d, T_right[:3, :3], T_right[:3, 3])
                    dist = jnp.linalg.norm(p_left - p_right, axis=1)
                    distortions.append({'type': 'vertical', 'pos': (r, c, c+1), 'dist': float(jnp.mean(dist))})
        for r in range(num_tiles - 1):
            for c in range(num_tiles):
                if (r, c) in tile_poses and (r+1, c) in tile_poses:
                    edge_y = int(r * stride + tile_size - (tile_size * overlap / 2))
                    x_start, x_end = c * stride, c * stride + tile_size
                    edge_x = jnp.linspace(x_start, x_end, 10)
                    edge_kpts = jnp.stack([edge_x, jnp.full_like(edge_x, edge_y)], axis=-1)
                    p_3d = lift_points(edge_kpts, data0['inv_depth'], data0['fov'])
                    T_top, T_bottom = tile_poses[(r, c)], tile_poses[(r+1, c)]
                    p_top = apply_transform(p_3d, T_top[:3, :3], T_top[:3, 3])
                    p_bottom = apply_transform(p_3d, T_bottom[:3, :3], T_bottom[:3, 3])
                    dist = jnp.linalg.norm(p_top - p_bottom, axis=1)
                    distortions.append({'type': 'horizontal', 'pos': (r, r+1, c), 'dist': float(jnp.mean(dist))})
        print("\n==================================================")
        print("SEGMENT EDGE DISTORTION ANALYSIS (GAP LENGTH)")
        print("==================================================")
        for d in sorted(distortions, key=lambda x: x['dist'], reverse=True):
            print(f"  {d['type'].capitalize()} Edge {d['pos']}: Gap = {d['dist']:.6f} units")

if __name__ == "__main__":
    analyzer = SegmentationAnalyzer(
        "weights/depth_pro.msgpack",
        "weights/superpoint.msgpack",
        "weights/superpoint_lightglue.msgpack"
    )
    analyzer.analyze_edge_distortion("data/pinecone_subset", num_tiles=4, overlap=0.40)
