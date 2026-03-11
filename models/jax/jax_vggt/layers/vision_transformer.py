import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Union, List
from .patch_embed import PatchEmbed
from .block import Block

class DinoVisionTransformer(nn.Module):
    img_size: int = 518
    patch_size: int = 14
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    init_values: Optional[float] = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> dict:
        B, C, H, W = x.shape
        
        x = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )(x)
        
        # cls_token
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_token = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        
        # Concat cls_token first
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # pos_embed (added to cls + patches)
        num_patches = (self.img_size // self.patch_size) ** 2
        pos_embed = self.param('pos_embed', nn.initializers.zeros, (1, num_patches + 1, self.embed_dim))
        x = x + pos_embed
        
        # register_tokens (added AFTER pos_embed)
        if self.num_register_tokens > 0:
            register_tokens = self.param('register_tokens', nn.initializers.zeros, (1, self.num_register_tokens, self.embed_dim))
            register_tokens = jnp.broadcast_to(register_tokens, (B, self.num_register_tokens, self.embed_dim))
            # PyTorch: x = torch.cat((x[:, :1], self.register_tokens.expand(...), x[:, 1:]), dim=1)
            x = jnp.concatenate([x[:, :1], register_tokens, x[:, 1:]], axis=1)
        
        # Blocks
        for i in range(self.depth):
            x = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                init_values=self.init_values,
                name=f'blocks_{i}'
            )(x)
            
        x_norm = nn.LayerNorm(epsilon=1e-6, name='norm')(x)
        
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x
        }
