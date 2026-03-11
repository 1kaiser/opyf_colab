import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, List, Union
import math

from ..layers.vision_transformer import DinoVisionTransformer
from ..layers.patch_embed import PatchEmbed
from ..layers.block import Block
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter

class Aggregator(nn.Module):
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    patch_embed_type: str = "dinov2_vitl14_reg"
    aa_order: List[str] = ("frame", "global")
    aa_block_size: int = 1
    qk_norm: bool = True
    rope_freq: int = 100
    init_values: float = 0.01

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> Tuple[List[jnp.ndarray], int]:
        # images: [B, S, 3, H, W]
        B, S, C_in, H, W = images.shape
        
        # 1. Normalize images (ResNet mean/std)
        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 3, 1, 1))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 3, 1, 1))
        images = (images - mean) / std
        
        # 2. Patch Embedding
        # Flatten B and S for patch embedding
        images_flat = images.reshape((B * S, C_in, H, W))
        
        if "conv" in self.patch_embed_type:
            patch_embed_layer = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, 
                in_chans=3, embed_dim=self.embed_dim, name='patch_embed'
            )
            patch_tokens = patch_embed_layer(images_flat)
        else:
            # ViT as patch embed
            patch_embed_layer = DinoVisionTransformer(
                img_size=self.img_size, patch_size=self.patch_size,
                embed_dim=self.embed_dim, depth=24, # Hardcoded for Large
                num_register_tokens=self.num_register_tokens,
                init_values=1.0, # DinoV2 default
                name='patch_embed'
            )
            patch_out = patch_embed_layer(images_flat)
            patch_tokens = patch_out["x_norm_patchtokens"]
            
        _, P_patch, _ = patch_tokens.shape
        
        # 3. Special Tokens
        # camera_token: [1, 2, 1, D]
        camera_token_param = self.param('camera_token', nn.initializers.normal(stddev=1e-6), (1, 2, 1, self.embed_dim))
        # register_token: [1, 2, num_reg, D]
        register_token_param = self.param('register_token', nn.initializers.normal(stddev=1e-6), (1, 2, self.num_register_tokens, self.embed_dim))
        
        def slice_expand_and_flatten(token_tensor, B, S):
            # query [1, 1, X, D] -> [B, 1, X, D]
            query = jnp.broadcast_to(token_tensor[:, 0:1, ...], (B, 1) + token_tensor.shape[2:])
            # others [1, 1, X, D] -> [B, S-1, X, D]
            others = jnp.broadcast_to(token_tensor[:, 1:, ...], (B, S - 1) + token_tensor.shape[2:])
            combined = jnp.concatenate([query, others], axis=1) # [B, S, X, D]
            return combined.reshape((B * S,) + token_tensor.shape[2:])

        camera_token = slice_expand_and_flatten(camera_token_param, B, S)
        register_token = slice_expand_and_flatten(register_token_param, B, S)
        
        tokens = jnp.concatenate([camera_token, register_token, patch_tokens], axis=1) # [B*S, 1 + num_reg + P_patch, D]
        
        # 4. RoPE
        rope = None
        if self.rope_freq > 0:
            rope = RotaryPositionEmbedding2D(frequency=float(self.rope_freq), name='rope')
            
        patch_start_idx = 1 + self.num_register_tokens
        
        # Position indices
        gh, gw = H // self.patch_size, W // self.patch_size
        pos = PositionGetter()(B * S, gh, gw) # [B*S, gh*gw, 2]
        
        # special tokens get pos 0, others get pos+1
        pos = pos + 1
        pos_special = jnp.zeros((B * S, patch_start_idx, 2), dtype=pos.dtype)
        pos = jnp.concatenate([pos_special, pos], axis=1)
        
        # 5. Alternating Attention Blocks
        _, P_total, C = tokens.shape
        
        frame_blocks = [
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qk_norm=self.qk_norm, rope=rope, init_values=self.init_values, name=f'frame_blocks_{i}'
            ) for i in range(self.depth)
        ]
        
        global_blocks = [
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qk_norm=self.qk_norm, rope=rope, init_values=self.init_values, name=f'global_blocks_{i}'
            ) for i in range(self.depth)
        ]
        
        output_list = []
        
        curr_tokens = tokens
        
        for b_idx in range(self.depth // self.aa_block_size):
            # Frame Attention
            for i in range(self.aa_block_size):
                idx = b_idx * self.aa_block_size + i
                # Frame attention is over tokens of each frame independently
                # curr_tokens is already [B*S, P, D]
                curr_tokens = frame_blocks[idx](curr_tokens, pos=pos)
                frame_intermediates = curr_tokens.reshape((B, S, P_total, C))
                
            # Global Attention
            for i in range(self.aa_block_size):
                idx = b_idx * self.aa_block_size + i
                # Reshape for global attention: [B, S*P, D]
                tokens_global = curr_tokens.reshape((B, S * P_total, C))
                # RoPE positions also need to be expanded?
                # In PyTorch: pos = pos.view(B, S, P, 2).view(B, S * P, 2)
                pos_global = pos.reshape((B, S * P_total, 2))
                
                tokens_global = global_blocks[idx](tokens_global, pos=pos_global)
                
                # Reshape back
                curr_tokens = tokens_global.reshape((B * S, P_total, C))
                global_intermediates = curr_tokens.reshape((B, S, P_total, C))
                
            # Combine intermediates
            # In PyTorch: concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
            # Since aa_block_size is usually 1, we just take the last ones
            concat_inter = jnp.concatenate([frame_intermediates, global_intermediates], axis=-1)
            output_list.append(concat_inter)
            
        return output_list, patch_start_idx
