import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional, Any

class RoPE2D(nn.Module):
    freq: float = 100.0
    F0: float = 1.0

    def get_cos_sin(self, D, max_pos, dtype):
        inv_freq = self.F0 / (self.freq ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D))
        t = jnp.arange(max_pos, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq).astype(dtype)
        freqs = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(freqs) # (MaxPos, D)
        sin = jnp.sin(freqs)
        return cos, sin

    def rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        # tokens: (B, H, N, D/2)
        # pos1d: (B, N)
        # cos, sin: (MaxPos, D/2)
        cos_emb = cos[pos1d][:, None, :, :]
        sin_emb = sin[pos1d][:, None, :, :]
        return (tokens * cos_emb) + (self.rotate_half(tokens) * sin_emb)

    def __call__(self, tokens, positions):
        # tokens: (B, H, N, D)
        # positions: (B, N, 2)
        B, H, N, D = tokens.shape
        D_half = D // 2
        
        # Use a fixed max_pos for JIT compatibility
        # 64 is enough for 1024x1024 image with 16x16 patches
        max_pos = 64 
        cos, sin = self.get_cos_sin(D_half, max_pos, tokens.dtype)
        
        y_tokens, x_tokens = jnp.split(tokens, 2, axis=-1)
        y_tokens = self.apply_rope1d(y_tokens, positions[:, :, 0], cos, sin)
        x_tokens = self.apply_rope1d(x_tokens, positions[:, :, 1], cos, sin)
        
        return jnp.concatenate((y_tokens, x_tokens), axis=-1)

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, pos: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos is not None:
            rope = RoPE2D()
            q = rope(q, pos)
            k = rope(k, pos)

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn = jnp.where(mask, -jnp.inf, attn)

        attn = nn.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        return x

class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    drop: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_dim, name='fc2')(x)
        return x

class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    proj_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.

    @nn.compact
    def __call__(self, x, pos: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
        h = nn.LayerNorm(epsilon=1e-5, name='norm1')(x)
        h = Attention(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')(h, pos=pos, mask=mask)
        x = x + h
        
        h = nn.LayerNorm(epsilon=1e-5, name='norm2')(x)
        h = MLP(hidden_dim=int(self.dim * self.mlp_ratio), out_dim=self.dim, drop=self.drop, name='mlp')(h)
        x = x + h
        return x

class PatchEmbed(nn.Module):
    patch_size: Tuple[int, int] = (16, 16)
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        B, C, H, W = x.shape
        x = nn.Conv(self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, name='proj', padding='VALID')(x.transpose(0, 2, 3, 1))
        B, H_p, W_p, C = x.shape
        x = x.reshape(B, -1, C)
        
        # Generate 2D positions for each patch
        grid_h = jnp.arange(H_p)
        grid_w = jnp.arange(W_p)
        grid = jnp.stack(jnp.meshgrid(grid_h, grid_w, indexing='ij'), axis=-1) # (H_p, W_p, 2)
        pos = grid.reshape(-1, 2)
        pos = jnp.tile(pos[None, :, :], (B, 1, 1))
        
        return x, pos
