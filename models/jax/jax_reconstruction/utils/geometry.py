import jax
import jax.numpy as jnp
from typing import Tuple

def fov_to_intrinsics(fov_deg: float, width: int, height: int) -> jnp.ndarray:
    """Computes camera intrinsic matrix from FOV and image dimensions."""
    fov_rad = jnp.deg2rad(fov_deg)
    f = 0.5 * width / jnp.tan(0.5 * fov_rad)
    cx = width / 2.0
    cy = height / 2.0
    return jnp.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])

def lift_points(kpts: jnp.ndarray, depth_map: jnp.ndarray, fov_deg: float) -> jnp.ndarray:
    """Lifts 2D keypoints to 3D points using depth and FOV.
    kpts: [N, 2] (x, y)
    depth_map: [H, W]
    """
    H, W = depth_map.shape
    fov_rad = jnp.deg2rad(fov_deg)
    f = 0.5 * W / jnp.tan(0.5 * fov_rad)
    cx = W / 2.0
    cy = H / 2.0
    
    # Fast sampling using rounded integer coordinates
    # map_coordinates is extremely slow during JIT for many points
    ix = jnp.clip(kpts[:, 0].astype(jnp.int32), 0, W - 1)
    iy = jnp.clip(kpts[:, 1].astype(jnp.int32), 0, H - 1)
    depths = depth_map[iy, ix]
    
    # Lift to 3D
    z = depths
    x = (kpts[:, 0] - cx) * z / f
    y = (kpts[:, 1] - cy) * z / f
    
    return jnp.stack([x, y, z], axis=-1)

def kabsch_alignment(P: jnp.ndarray, Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes rigid transform (R, t) such that Q = R*P + t
    P, Q: [N, 3]
    """
    # 1. Centers of mass
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)
    
    # 2. Shift points to origin
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 3. Covariance matrix
    H = P_centered.T @ Q_centered
    
    # 4. SVD
    U, S, Vt = jnp.linalg.svd(H)
    
    # 5. Correct rotation to avoid reflections
    d = jnp.linalg.det(Vt.T @ U.T)
    V_adj = Vt.T.at[:, 2].multiply(jnp.where(d < 0, -1.0, 1.0))
    
    R = V_adj @ U.T
    
    # 6. Translation
    t = centroid_Q - R @ centroid_P
    
    return R, t

def umeyama_alignment(P: jnp.ndarray, Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Computes similarity transform (s, R, t) such that Q = s*R*P + t
    P, Q: [N, 3]
    """
    n, m = P.shape
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Variance of P
    var_P = jnp.mean(jnp.sum(P_centered**2, axis=1))
    
    # Covariance
    H = P_centered.T @ Q_centered / n
    U, S, Vt = jnp.linalg.svd(H)
    
    # Correct reflection
    d = jnp.linalg.det(U) * jnp.linalg.det(Vt.T)
    S_adj = jnp.eye(m).at[m-1, m-1].set(jnp.where(d < 0, -1.0, 1.0))
    
    R = Vt.T @ S_adj @ U.T
    
    # Scale
    s = (1.0 / var_P) * jnp.trace(jnp.diag(S) @ S_adj)
    
    # Translation
    t = centroid_Q - s * (R @ centroid_P)
    
    return R, t, s

def apply_transform(P: jnp.ndarray, R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """P: [N, 3], R: [3, 3], t: [3]"""
    return (R @ P.T).T + t
