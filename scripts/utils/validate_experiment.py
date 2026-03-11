import pycolmap
import numpy as np
import os

def load_colmap_poses(path):
    recon = pycolmap.Reconstruction(path)
    poses = {}
    for image_id, image in recon.images.items():
        T_cam_to_world_3x4 = image.cam_from_world().inverse().matrix()
        T = np.eye(4)
        T[:3, :] = T_cam_to_world_3x4
        poses[image.name] = T
    return poses

def align_trajectories(P, Q):
    mu_P = np.mean(P, axis=0)
    mu_Q = np.mean(Q, axis=0)
    P_c = P - mu_P
    Q_c = Q - mu_Q
    s = np.linalg.norm(Q_c) / np.linalg.norm(P_c)
    P_c *= s
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = mu_Q - R @ (s * mu_P)
    return R, t, s

def validate_all():
    colmap_path = "output/colmap_run/0"
    colmap_poses = load_colmap_poses(colmap_path)
    results = {}
    for nz in [2, 3, 4, 5]:
        jax_poses_path = f"output/reconstruction/poses_{nz}zones.npy"
        if not os.path.exists(jax_poses_path): continue
        jax_poses_dict = np.load(jax_poses_path, allow_pickle=True).item()
        common_images = sorted([img for img in jax_poses_dict if img in colmap_poses])
        if not common_images: continue
        jax_centers, colmap_centers = [], []
        mid_zone = nz // 2
        for img in common_images:
            jax_centers.append(jax_poses_dict[img][mid_zone][:3, 3])
            colmap_centers.append(colmap_poses[img][:3, 3])
        jax_centers = np.array(jax_centers)
        colmap_centers = np.array(colmap_centers)
        R, t, s = align_trajectories(jax_centers, colmap_centers)
        jax_centers_aligned = (R @ (s * jax_centers).T).T + t
        rmse = np.sqrt(np.mean(np.linalg.norm(jax_centers_aligned - colmap_centers, axis=1)**2))
        results[nz] = rmse
        print(f"Zones: {nz}, Trajectory RMSE: {rmse:.6f}")
    print("\n==============================")
    print("FINAL EXPERIMENT RESULTS")
    print("==============================")
    for nz, rmse in results.items():
        print(f"Zones {nz}: RMSE = {rmse:.6f}")

if __name__ == "__main__":
    validate_all()
