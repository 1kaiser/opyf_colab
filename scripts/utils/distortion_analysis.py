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

class DistortionAnalyzer(ReconstructionPipeline):
    def analyze_distortion(self, image_folder, num_tiles=4, overlap=0.40):
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        img0_path = os.path.join(image_folder, img_files[0])
        img1_path = os.path.join(image_folder, img_files[1])
        data0 = self.process_image(img0_path)
        data1 = self.process_image(img1_path)
        H, W = 1536, 1536
        tile_size = int(H / (num_tiles - (num_tiles - 1) * overlap))
        stride = int(tile_size * (1 - overlap))
        results = []
        cy, cx = H/2.0, W/2.0
        max_r = math.sqrt(cy**2 + cx**2)
        for row in range(num_tiles):
            for col in range(num_tiles):
                y0, x0 = row * stride, col * stride
                y1, x1 = y0 + tile_size, x0 + tile_size
                ty, tx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
                radial_dist = math.sqrt((ty - cy)**2 + (tx - cx)**2) / max_r
                def get_tile_kpts(scores, desc):
                    flat_idx = jnp.argsort(scores.flatten())[::-1][:5000]
                    sy, sx = jnp.unravel_index(flat_idx, scores.shape)
                    mask = (sy >= y0) & (sy < y1) & (sx >= x0) & (sx < x1)
                    kpts = jnp.stack([sx[mask], sy[mask]], axis=-1)[:512]
                    iy = jnp.clip((sy[mask][:512] / 8).astype(jnp.int32), 0, desc.shape[0]-1)
                    ix = jnp.clip((sx[mask][:512] / 8).astype(jnp.int32), 0, desc.shape[1]-1)
                    sampled_desc = desc[iy, ix, :]
                    return kpts, sampled_desc
                k0, d0 = get_tile_kpts(data0['sp_scores'], data0['sp_desc'])
                k1, d1 = get_tile_kpts(data1['sp_scores'], data1['sp_desc'])
                if len(k0) < 10 or len(k1) < 10:
                    results.append({'row': row, 'col': col, 'rmse': None, 'r': radial_dist})
                    continue
                lg_input = {
                    "image0": {"keypoints": k0[None], "descriptors": d0[None]},
                    "image1": {"keypoints": k1[None], "descriptors": d1[None]}
                }
                lg_out = self.jit_lg(self.variables['lg'], lg_input)
                scores = lg_out['scores'][0, :-1, :-1]
                m0 = jnp.argmax(scores, axis=1)
                m1 = jnp.argmax(scores, axis=0)
                mutual = (jnp.arange(len(m0)) == m1[m0])
                valid = mutual & (jnp.exp(jnp.max(scores, axis=1)) > 0.1)
                idx0 = jnp.where(valid)[0]
                idx1 = m0[idx0]
                if len(idx0) > 8:
                    p0_3d = lift_points(k0[idx0], data0['inv_depth'], data0['fov'])
                    p1_3d = lift_points(k1[idx1], data1['inv_depth'], data1['fov'])
                    R, t = kabsch_alignment(p0_3d, p1_3d)
                    p0_trans = apply_transform(p0_3d, R, t)
                    rmse = jnp.sqrt(jnp.mean(jnp.sum((p0_trans - p1_3d)**2, axis=1)))
                    results.append({'row': row, 'col': col, 'rmse': float(rmse), 'r': float(radial_dist), 'matches': int(len(idx0))})
                else:
                    results.append({'row': row, 'col': col, 'rmse': None, 'r': radial_dist})
        print("\n" + "="*40)
        print("TILE DISTORTION MAP (RMSE)")
        print("="*40)
        grid = np.zeros((num_tiles, num_tiles))
        for res in results:
            grid[res['row'], res['col']] = res['rmse'] if res['rmse'] is not None else -1
        print(grid)
        print("\nRadial Correlation (Distance from Center vs RMSE):")
        sorted_res = sorted([r for r in results if r['rmse'] is not None], key=lambda x: x['r'])
        for r in sorted_res:
            print(f"  Dist: {r['r']:.3f} | RMSE: {r['rmse']:.6f} | Matches: {r['matches']}")

if __name__ == "__main__":
    analyzer = DistortionAnalyzer(
        "weights/depth_pro.msgpack",
        "weights/superpoint.msgpack",
        "weights/superpoint_lightglue.msgpack"
    )
    analyzer.analyze_distortion("data/pinecone_subset", num_tiles=4, overlap=0.40)
