import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Optional, Any
from ..layers.block import Block
from ..layers.mlp import Mlp
from .head_act import activate_pose

class CameraHead(nn.Module):
    dim_in: int = 2048
    trunk_depth: int = 4
    num_heads: int = 16
    mlp_ratio: int = 4
    init_values: float = 0.01
    target_dim: int = 9
    trans_act: str = "linear"
    quat_act: str = "linear"
    fl_act: str = "relu"

    @nn.compact
    def __call__(self, aggregated_tokens_list: List[jnp.ndarray], num_iterations: int = 4) -> List[jnp.ndarray]:
        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        
        token_norm = nn.LayerNorm(epsilon=1e-6, name='token_norm')
        pose_tokens = token_norm(pose_tokens)
        
        trunk = [
            Block(dim=self.dim_in, num_heads=self.num_heads, mlp_ratio=float(self.mlp_ratio), init_values=self.init_values, name=f'trunk_{i}')
            for i in range(self.trunk_depth)
        ]
        
        trunk_norm = nn.LayerNorm(epsilon=1e-6, name='trunk_norm')
        adaln_norm = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6, name='adaln_norm')
        
        empty_pose_tokens = self.param('empty_pose_tokens', nn.initializers.zeros, (1, 1, self.target_dim))
        embed_pose = nn.Dense(self.dim_in, name='embed_pose')
        
        poseLN_mod_dense = nn.Dense(3 * self.dim_in, name='poseLN_modulation_dense')
        pose_branch = Mlp(in_features=self.dim_in, hidden_features=self.dim_in // 2, out_features=self.target_dim, name='pose_branch')
        
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = embed_pose(jnp.broadcast_to(empty_pose_tokens, (B, S, self.target_dim)))
            else:
                module_input = embed_pose(pred_pose_enc)
                
            mod = poseLN_mod_dense(jax.nn.silu(module_input))
            shift, scale, gate = jnp.split(mod, 3, axis=-1)
            
            h = adaln_norm(pose_tokens)
            h = h * (1 + scale) + shift
            h = gate * h
            
            h = h + pose_tokens
            
            for blk in trunk:
                h = blk(h)
                
            delta = pose_branch(trunk_norm(h))
            
            if pred_pose_enc is None:
                pred_pose_enc = delta
            else:
                pred_pose_enc = pred_pose_enc + delta
                
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)
            
        return pred_pose_enc_list
