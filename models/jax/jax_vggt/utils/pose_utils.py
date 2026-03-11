import jax
import jax.numpy as jnp

def quat_to_mat(quat):
    """Converts a quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    # Note: VGGT uses (qw, qx, qy, qz) order based on PyTorch3D or similar
    # Let's verify common conventions
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    row0 = jnp.stack([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)
    row1 = jnp.stack([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)
    row2 = jnp.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)
    
    return jnp.stack([row0, row1, row2], axis=-2)

def pose_encoding_to_extri_intri(pose_encoding, image_size_hw):
    """JAX implementation of VGGT pose decoder."""
    # pose_encoding: [B, S, 9]
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    # Normalize quaternion
    quat = quat / jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    
    R = quat_to_mat(quat)
    # extrinsics: [B, S, 3, 4]
    extrinsics = jnp.concatenate([R, T[..., None]], axis=-1)
    
    H, W = image_size_hw
    fy = (H / 2.0) / jnp.tan(fov_h / 2.0)
    fx = (W / 2.0) / jnp.tan(fov_w / 2.0)
    
    B, S = pose_encoding.shape[:2]
    intrinsics = jnp.zeros((B, S, 3, 3))
    
    # Use index_update equivalents
    intrinsics = intrinsics.at[..., 0, 0].set(fx)
    intrinsics = intrinsics.at[..., 1, 1].set(fy)
    intrinsics = intrinsics.at[..., 0, 2].set(W / 2.0)
    intrinsics = intrinsics.at[..., 1, 2].set(H / 2.0)
    intrinsics = intrinsics.at[..., 2, 2].set(1.0)
    
    return extrinsics, intrinsics
