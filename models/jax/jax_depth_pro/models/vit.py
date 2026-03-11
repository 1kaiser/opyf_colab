import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional, Any

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.initializers.constant(self.init_values), (self.dim,))
        return x * gamma

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = nn.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        return x

class MLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_dim, name='fc2')(x)
        return x

class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    proj_bias: bool = True
    init_values: Optional[float] = 1e-5

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(epsilon=1e-6, name='norm1')(x) # DinoV2 uses 1e-6? Let's check
        h = Attention(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, proj_bias=self.proj_bias, name='attn')(h)
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls1')(h)
        x = x + h
        
        h = nn.LayerNorm(epsilon=1e-6, name='norm2')(x)
        h = MLP(hidden_dim=int(self.dim * self.mlp_ratio), out_dim=self.dim, name='mlp')(h)
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls2')(h)
        x = x + h
        return x

class PatchEmbed(nn.Module):
    patch_size: int = 16
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        # x shape: [B, C, H, W]
        x = nn.Conv(self.embed_dim, kernel_size=(self.patch_size, self.patch_size), 
                    strides=(self.patch_size, self.patch_size), padding='VALID', name='proj')(x.transpose(0, 2, 3, 1))
        # x shape: [B, H_p, W_p, C]
        B, H_p, W_p, C = x.shape
        x = x.reshape(B, -1, C)
        return x

class ViT(nn.Module):
    img_size: int = 384
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    proj_bias: bool = True
    init_values: Optional[float] = 1e-5

    @nn.compact
    def __call__(self, x):
        B, C, H, W = x.shape
        x = PatchEmbed(patch_size=self.patch_size, embed_dim=self.embed_dim, name='patch_embed')(x)
        
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_token = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        num_patches = (self.img_size // self.patch_size) ** 2
        pos_embed = self.param('pos_embed', nn.initializers.zeros, (1, num_patches + 1, self.embed_dim))
        x = x + pos_embed
        
        # We'll collect hook outputs
        hooks = []
        for i in range(self.depth):
            x = Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, 
                      qkv_bias=self.qkv_bias, proj_bias=self.proj_bias, init_values=self.init_values, name=f'blocks_{i}')(x)
            hooks.append(x)
            
        x = nn.LayerNorm(epsilon=1e-6, name='norm')(x)
        return x, hooks
