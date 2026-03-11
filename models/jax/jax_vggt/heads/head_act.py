import jax
import jax.numpy as jnp

def inverse_log_transform(y):
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))

def base_pose_act(pose_enc, act_type="linear"):
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return jnp.exp(pose_enc)
    elif act_type == "relu":
        return jax.nn.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")

def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]
    
    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)
    
    return jnp.concatenate([T, quat, fl], axis=-1)

def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    # out: [B, H, W, C]
    xyz = out[..., :-1]
    conf = out[..., -1]
    
    if activation == "norm_exp":
        d = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        d = jnp.maximum(d, 1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * jnp.expm1(d)
    elif activation == "norm":
        pts3d = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    elif activation == "exp":
        pts3d = jnp.exp(xyz)
    elif activation == "relu":
        pts3d = jax.nn.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy = xyz[..., :2]
        z = xyz[..., 2:3]
        z = inverse_log_transform(z)
        pts3d = jnp.concatenate([xy * z, z], axis=-1)
    elif activation == "sigmoid":
        pts3d = jax.nn.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")
        
    if conf_activation == "expp1":
        conf_out = 1 + jnp.exp(conf)
    elif conf_activation == "expp0":
        conf_out = jnp.exp(conf)
    elif conf_activation == "sigmoid":
        conf_out = jax.nn.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")
        
    return pts3d, conf_out
