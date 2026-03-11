import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Optional, Any, Union

# Relative imports following the jax_vggt convention
from .aggregator import Aggregator
from ..layers.block import Block
from ..layers.attention import Attention
from ..layers.vision_transformer import DinoVisionTransformer
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ..layers.mlp import Mlp
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead

# Muon optimizer import
from ..utils.muon import muon_update

"""
VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale
arXiv:2602.23361 (2026)

Citation:
@article{sun2026vggt3,
  title={VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale},
  author={Sun, Aljoša and others},
  journal={arXiv preprint arXiv:2602.23361},
  year={2026}
}
"""

class NormLayer(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (self.dim,))
        bias = self.param('bias', nn.initializers.zeros, (self.dim,))
        return x * scale + bias

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.initializers.constant(self.init_values), (self.dim,))
        return x * gamma

class TTTAttention(nn.Module):
    """
    Linearized Global Attention using Test-Time Training (TTT).
    Replaces quadratic softmax attention with an MLP optimized at test-time.
    """
    dim: int
    num_heads: int = 16
    lr: float = 0.1
    ns_iters: int = 5
    ttt_steps: int = 2
    qkv_bias: bool = True
    proj_bias: bool = True
    tokens_per_frame: int = 1374
    patches_per_frame: int = 1369

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        B, SN, C = x.shape
        S = SN // self.tokens_per_frame
        head_dim = self.dim // self.num_heads
        hidden_dim = 4 * self.dim
        
        # 1. QKV Projections
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # 2. QK L2-Normalization (Crucial for TTT convergence)
        q = q.reshape(B, SN, self.num_heads, head_dim)
        k = k.reshape(B, SN, self.num_heads, head_dim)
        
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        q = NormLayer(dim=head_dim, name='q_norm')(q)
        
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        k = NormLayer(dim=head_dim, name='k_norm')(k)
        
        q_normed = q.reshape(B, SN, self.dim)
        k_normed = k.reshape(B, SN, self.dim)

        # 3. ShortConv2D on Values (Non-linear Spatial Mixing)
        v_per_frame = v.reshape(B, S, self.tokens_per_frame, self.dim)
        num_special = self.tokens_per_frame - self.patches_per_frame
        v_special = v_per_frame[:, :, :num_special, :]
        v_patches = v_per_frame[:, :, num_special:, :]
        
        v_grid = v_patches.reshape(B * S, 37, 37, self.dim)
        
        # Check if short_conv weights exist (added during post-training linearization)
        if 'short_conv' in self.scope.variables().get('params', {}):
            v_mixed = nn.Conv(
                self.dim, (3, 3), padding='SAME', feature_group_count=self.dim, 
                use_bias=False, name='short_conv'
            )(v_grid)
        else:
            v_mixed = v_grid
            
        v_target_patches = v_mixed.reshape(B, S, self.patches_per_frame, self.dim)
        v_target = jnp.concatenate([v_special, v_target_patches], axis=2).reshape(B, SN, self.dim)
        
        k_ttt = k_normed.reshape(-1, self.dim)
        v_ttt = v_target.reshape(-1, self.dim)
        
        # 4. TTT Update (Optimizing Fast Weights theta)
        def ttt_step(params, k_b, v_b):
            def loss_fn(p):
                # SwiGLU MLP: (Swish(x @ W1) * (x @ V1)) @ W2
                gate = k_b @ p['gate']
                val = k_b @ p['val']
                h = jax.nn.swish(gate) * val
                pred = h @ p['proj']
                return jnp.mean((pred - v_b)**2)
            
            grads = jax.grad(loss_fn)(params)
            return {
                'gate': muon_update(params['gate'], grads['gate'], lr=self.lr, ns_iters=self.ns_iters),
                'val': muon_update(params['val'], grads['val'], lr=self.lr, ns_iters=self.ns_iters),
                'proj': muon_update(params['proj'], grads['proj'], lr=self.lr, ns_iters=self.ns_iters)
            }

        init_fn = nn.initializers.lecun_normal()
        params_dict = self.scope.variables().get('params', {})
        
        if 'ttt_base_gate' in params_dict:
            theta = {
                'gate': self.param('ttt_base_gate', init_fn, (self.dim, hidden_dim)),
                'val': self.param('ttt_base_val', init_fn, (self.dim, hidden_dim)),
                'proj': self.param('ttt_base_proj', init_fn, (hidden_dim, self.dim))
            }
        else:
            # Cold start for inference if not in pre-trained weights
            key = jax.random.PRNGKey(0) 
            theta = {
                'gate': init_fn(key, (self.dim, hidden_dim)),
                'val': init_fn(key, (self.dim, hidden_dim)),
                'proj': init_fn(key, (hidden_dim, self.dim))
            }

        steps = 1 if train else self.ttt_steps
        for _ in range(steps):
            theta = ttt_step(theta, k_ttt, v_ttt)
            
        # 5. Apply optimized MLP to queries
        q_ttt = q_normed.reshape(-1, self.dim)
        gate_q = q_ttt @ theta['gate']
        val_q = q_ttt @ theta['val']
        h_q = jax.nn.swish(gate_q) * val_q
        o = h_q @ theta['proj']
        
        o = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(o)
        return o.reshape(B, SN, self.dim)

class TTTBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    init_values: Optional[float] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        h = nn.LayerNorm(epsilon=1e-6, name='norm1')(x)
        h = TTTAttention(dim=self.dim, num_heads=self.num_heads, name='attn')(h, train=train)
        
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls1')(h)
        x = x + h
        
        h = nn.LayerNorm(epsilon=1e-6, name='norm2')(x)
        h = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            name='mlp'
        )(h)
        
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls2')(h)
        x = x + h
        return x

class VGGT3Aggregator(nn.Module):
    """
    Interleaves standard per-frame blocks with TTT-based view-global blocks.
    """
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    init_values: float = 0.01

    @nn.compact
    def __call__(self, images: jnp.ndarray, train: bool = False) -> Tuple[List[jnp.ndarray], int]:
        B, S, C_in, H, W = images.shape
        
        # Normalize
        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 3, 1, 1))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 3, 1, 1))
        images = (images - mean) / std
        
        # Patch Embedding (DinoV2)
        images_flat = images.reshape((B * S, C_in, H, W))
        patch_embed_layer = DinoVisionTransformer(
            img_size=self.img_size, patch_size=self.patch_size,
            embed_dim=self.embed_dim, depth=24,
            num_register_tokens=self.num_register_tokens,
            init_values=1.0,
            name='patch_embed'
        )
        patch_out = patch_embed_layer(images_flat)
        patch_tokens = patch_out["x_norm_patchtokens"]
        
        # Aggregator Tokens
        camera_token_param = self.param('camera_token', nn.initializers.normal(stddev=1e-6), (1, 2, 1, self.embed_dim))
        register_token_param = self.param('register_token', nn.initializers.normal(stddev=1e-6), (1, 2, 4, self.embed_dim))
        
        def slice_expand_and_flatten(token_tensor, B, S):
            query = jnp.broadcast_to(token_tensor[:, 0:1, ...], (B, 1) + token_tensor.shape[2:])
            others = jnp.broadcast_to(token_tensor[:, 1:, ...], (B, S - 1) + token_tensor.shape[2:])
            combined = jnp.concatenate([query, others], axis=1)
            return combined.reshape((B * S,) + token_tensor.shape[2:])

        camera_token = slice_expand_and_flatten(camera_token_param, B, S)
        register_token = slice_expand_and_flatten(register_token_param, B, S)
        
        tokens = jnp.concatenate([camera_token, register_token, patch_tokens], axis=1)
        
        rope = RotaryPositionEmbedding2D(frequency=100.0, name='rope')
        patch_start_idx = 1 + self.num_register_tokens
        gh, gw = H // self.patch_size, W // self.patch_size
        pos = PositionGetter()(B * S, gh, gw)
        pos = pos + 1
        pos_special = jnp.zeros((B * S, patch_start_idx, 2), dtype=pos.dtype)
        pos = jnp.concatenate([pos_special, pos], axis=1)
        
        output_list = []
        curr_tokens = tokens
        
        for i in range(self.depth):
            # Frame Block (Standard)
            curr_tokens = Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qk_norm=True, rope=rope, init_values=self.init_values, name=f'frame_blocks_{i}'
            )(curr_tokens, pos=pos)
            frame_intermediates = curr_tokens.reshape((B, S, -1, self.embed_dim))
            
            # Global Block (TTT - Linear Complexity)
            tokens_global = curr_tokens.reshape((B, S * curr_tokens.shape[1], self.embed_dim))
            tokens_global = TTTBlock(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                init_values=self.init_values, name=f'global_blocks_{i}'
            )(tokens_global, train=train)
            
            curr_tokens = tokens_global.reshape((B * S, -1, self.embed_dim))
            global_intermediates = curr_tokens.reshape((B, S, -1, self.embed_dim))
            
            concat_inter = jnp.concatenate([frame_intermediates, global_intermediates], axis=-1)
            output_list.append(concat_inter)
            
        return output_list, patch_start_idx

class VGGT3(nn.Module):
    """
    The complete VGG-T³ model for scalable 3D reconstruction.
    """
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    enable_camera: bool = True
    enable_point: bool = True
    enable_depth: bool = True

    @nn.compact
    def __call__(self, images: jnp.ndarray, train: bool = False) -> dict:
        # images: [B, S, 3, H, W]
        aggregated_tokens_list, patch_start_idx = VGGT3Aggregator(
            img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim, name='aggregator'
        )(images, train=train)
        
        predictions = {}
        if self.enable_camera:
            predictions["pose_enc_list"] = CameraHead(dim_in=2 * self.embed_dim, name='camera_head')(aggregated_tokens_list)
            predictions["pose_enc"] = predictions["pose_enc_list"][-1]
            
        if self.enable_depth:
            depth, depth_conf = DPTHead(dim_in=2 * self.embed_dim, output_dim=2, name='depth_head')(aggregated_tokens_list, images, patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
            
        if self.enable_point:
            pts3d, pts3d_conf = DPTHead(dim_in=2 * self.embed_dim, output_dim=4, name='point_head')(aggregated_tokens_list, images, patch_start_idx)
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf
            
        return predictions
