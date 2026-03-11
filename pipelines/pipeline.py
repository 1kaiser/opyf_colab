import os
import sys
import torch
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import pylas

# Add repos to path
sys.path.append(os.path.abspath("depth_pro_repo/src"))
sys.path.append(os.path.abspath("LightGlue"))

import depth_pro
from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT
from lightglue import LightGlue, SuperPoint
from lightglue.utils import read_image, numpy_image_to_torch, rbd, resize_image

def get_depth_model(device):
    checkpoint_path = "depth_pro_repo/checkpoints/depth_pro.pt"
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = checkpoint_path
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    return model, transform

def get_matching_models(device):
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    return extractor, matcher

@torch.no_grad()
def run_depth_pro(model, transform, image_path, device):
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(device).unsqueeze(0)
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()
    focallength_px = prediction["focallength_px"]
    if torch.is_tensor(focallength_px):
        focallength_px = focallength_px.detach().cpu().item()
    return depth, focallength_px

def lift_to_3d(image_path, depth, focal_px):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = depth.shape
    
    # Create meshgrid of coordinates
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)
    
    # Back-project to 3D
    # Using center of image as principal point
    cx, cy = w / 2, h / 2
    z = depth
    x_3d = (xv - cx) * z / focal_px
    y_3d = (yv - cy) * z / focal_px
    
    points_3d = np.stack([x_3d, y_3d, z], axis=-1).reshape(-1, 3)
    colors = img.reshape(-1, 3) / 255.0
    
    return points_3d, colors

def estimate_rigid_transform(pts_src, pts_dst):
    # pts_src, pts_dst: (N, 3)
    # Returns R, t such that pts_dst = R @ pts_src + t
    
    centroid_src = np.mean(pts_src, axis=0)
    centroid_dst = np.mean(pts_dst, axis=0)
    
    pts_src_centered = pts_src - centroid_src
    pts_dst_centered = pts_dst - centroid_dst
    
    H = pts_src_centered.T @ pts_dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_dst - R @ centroid_src
    return R, t

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    image_dir = "data/pinecone/images"
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".JPG")])
    image_files = image_files[:3] # Subset for testing
    
    depth_model, depth_transform = get_depth_model(device)
    extractor, matcher = get_matching_models(device)
    
    global_points = []
    global_colors = []
    
    # Current transformation (identity)
    current_R = np.eye(3)
    current_t = np.zeros(3)
    
    prev_data = None
    
    for i, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
        # 1. Depth Estimation
        depth, focal_px = run_depth_pro(depth_model, depth_transform, img_path, device)
        pts_3d, colors = lift_to_3d(img_path, depth, focal_px)
        
        # Load image for matching (SuperPoint expects grayscale)
        image_np = read_image(img_path, grayscale=True)
        # Resize for matching to save memory
        image_np_resized, _ = resize_image(image_np, 1024)
        image_tensor = numpy_image_to_torch(image_np_resized).to(device).unsqueeze(0)
        feats = extractor({'image': image_tensor})
        
        if prev_data is not None:
            # 2. Feature Matching
            matches01 = matcher({'image0': prev_data['feats'], 'image1': feats})
            feats0, feats1, matches01 = [rbd(x) for x in [prev_data['feats'], feats, matches01]]
            
            kpts0 = feats0['keypoints'][matches01['matches'][..., 0]].cpu().numpy()
            kpts1 = feats1['keypoints'][matches01['matches'][..., 1]].cpu().numpy()
            
            if len(kpts0) > 8:
                # 3. Lift matching keypoints to 3D
                h, w = depth.shape
                # Coordinates for prev image (src)
                kpts0_int = kpts0.astype(int)
                kpts0_int[:, 0] = np.clip(kpts0_int[:, 0], 0, w - 1)
                kpts0_int[:, 1] = np.clip(kpts0_int[:, 1], 0, h - 1)
                
                # Coordinates for curr image (dst)
                kpts1_int = kpts1.astype(int)
                kpts1_int[:, 0] = np.clip(kpts1_int[:, 0], 0, w - 1)
                kpts1_int[:, 1] = np.clip(kpts1_int[:, 1], 0, h - 1)
                
                # Previous image's depth and focal
                prev_depth = prev_data['depth']
                prev_focal = prev_data['focal']
                
                # Principal points (assuming center)
                cx, cy = w / 2, h / 2
                
                # 3D points in prev frame coords
                z0 = prev_depth[kpts0_int[:, 1], kpts0_int[:, 0]]
                x0 = (kpts0[:, 0] - cx) * z0 / prev_focal
                y0 = (kpts0[:, 1] - cy) * z0 / prev_focal
                pts3d_0 = np.stack([x0, y0, z0], axis=-1)
                
                # 3D points in curr frame coords
                z1 = depth[kpts1_int[:, 1], kpts1_int[:, 0]]
                x1 = (kpts1[:, 0] - cx) * z1 / focal_px
                y1 = (kpts1[:, 1] - cy) * z1 / focal_px
                pts3d_1 = np.stack([x1, y1, z1], axis=-1)
                
                # Estimate transform from curr to prev
                # pts3d_0 = R @ pts3d_1 + t
                R, t = estimate_rigid_transform(pts3d_1, pts3d_0)
                
                # Update global pose
                current_t = current_R @ t + current_t
                current_R = current_R @ R
            else:
                print(f"Warning: Not enough matches between {i-1} and {i}")

        # Accumulate transformed points
        # Downsample for efficiency
        downsample_factor = 4
        pts_3d_ds = pts_3d.reshape(depth.shape[0], depth.shape[1], 3)[::downsample_factor, ::downsample_factor].reshape(-1, 3)
        colors_ds = colors.reshape(depth.shape[0], depth.shape[1], 3)[::downsample_factor, ::downsample_factor].reshape(-1, 3)
        
        pts_3d_global = (pts_3d_ds @ current_R.T) + current_t
        global_points.append(pts_3d_global)
        global_colors.append(colors_ds)
        
        prev_data = {
            'feats': feats,
            'depth': depth,
            'focal': focal_px,
            'path': img_path
        }

    # 4. Fusion
    full_points = np.concatenate(global_points, axis=0)
    full_colors = np.concatenate(global_colors, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_points)
    pcd.colors = o3d.utility.Vector3dVector(full_colors)
    
    # Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Save Point Cloud
    print("Saving point cloud...")
    o3d.io.write_point_cloud("point_cloud.glb", pcd)
    
    # LAS Export
    las = pylas.create()
    las.x = full_points[:, 0]
    las.y = full_points[:, 1]
    las.z = full_points[:, 2]
    las.red = (full_colors[:, 0] * 65535).astype(np.uint16)
    las.green = (full_colors[:, 1] * 65535).astype(np.uint16)
    las.blue = (full_colors[:, 2] * 65535).astype(np.uint16)
    las.write("point_cloud.las")
    
    # Mesh Generation
    print("Generating mesh...")
    pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.io.write_triangle_mesh("mesh.glb", mesh)
    
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
