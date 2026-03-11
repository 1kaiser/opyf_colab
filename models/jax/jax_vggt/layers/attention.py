import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    qk_norm: bool = False
    rope: Optional[Any] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.qk_norm:
            q = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6, name='q_norm')(q)
            k = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6, name='k_norm')(k)
            
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        return x
