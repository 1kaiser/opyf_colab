import pycolmap
import numpy as np
import os

def load_colmap_poses(path):
    recon = pycolmap.Reconstruction(path)
    poses = {}
    for image_id, image in recon.images.items():
        # cam_from_world() returns a Rigid3d
        # world_from_cam() returns the inverse Rigid3d
        T_cam_to_world_3x4 = image.cam_from_world().inverse().matrix()
        T = np.eye(4)
        T[:3, :] = T_cam_to_world_3x4
        poses[image.name] = T
    return poses

def align_trajectories(P, Q):
    """Aligns trajectory P to Q using Procrustes (rotation, translation, scale).
    P, Q: [N, 3] camera centers
    """
    mu_P = np.mean(P, axis=0)
    mu_Q = np.mean(Q, axis=0)
    
    P_c = P - mu_P
    Q_c = Q - mu_Q
    
    # Scale
    s = np.linalg.norm(Q_c) / np.linalg.norm(P_c)
    P_c *= s
    
    # Rotation
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = mu_Q - R @ (s * mu_P)
    return R, t, s

def validate():
    colmap_path = "output/colmap_run/0"
    jax_poses_path = "output/reconstruction/poses.npy"
    
    if not os.path.exists(jax_poses_path):
        print("JAX poses not found. Run pipeline first.")
        return
        
    colmap_poses = load_colmap_poses(colmap_path)
    # npy was saved as a dict
    jax_poses_dict = np.load(jax_poses_path, allow_pickle=True).item()
    
    common_images = sorted([img for img in jax_poses_dict if img in colmap_poses])
    
    if not common_images:
        print("No common images found between JAX and COLMAP.")
        return
        
    print(f"Validating trajectory over {len(common_images)} frames.")
    
    jax_centers = []
    colmap_centers = []
    
    for img in common_images:
        # Camera center is the translation vector of T_cam_to_world
        jax_centers.append(jax_poses_dict[img][:3, 3])
        colmap_centers.append(colmap_poses[img][:3, 3])
        
    jax_centers = np.array(jax_centers)
    colmap_centers = np.array(colmap_centers)
    
    # Align
    R, t, s = align_trajectories(jax_centers, colmap_centers)
    jax_centers_aligned = (R @ (s * jax_centers).T).T + t
    
    errors = np.linalg.norm(jax_centers_aligned - colmap_centers, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"Trajectory Alignment Results:")
    print(f"  RMSE: {rmse:.6f} (COLMAP units)")
    print(f"  Scale Factor (JAX/COLMAP): {1.0/s:.6f}")
    
    for i, img in enumerate(common_images):
        print(f"  {img}: Error={errors[i]:.6f}")

if __name__ == "__main__":
    validate()
