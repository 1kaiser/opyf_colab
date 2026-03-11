import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any
from .attention import Attention
from .mlp import Mlp
from .layer_scale import LayerScale

class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    init_values: Optional[float] = None
    qk_norm: bool = False
    rope: Optional[Any] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Note: In JAX we don't implement drop_path here for inference weights conversion
        # but we could if needed for training.
        
        h = nn.LayerNorm(epsilon=1e-6, name='norm1')(x)
        h = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            qk_norm=self.qk_norm,
            rope=self.rope,
            name='attn'
        )(h, pos=pos)
        
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls1')(h)
        x = x + h
        
        h = nn.LayerNorm(epsilon=1e-6, name='norm2')(x)
        h = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            bias=self.ffn_bias,
            name='mlp'
        )(h)
        
        if self.init_values is not None:
            h = LayerScale(dim=self.dim, init_values=self.init_values, name='ls2')(h)
        x = x + h
        return x
