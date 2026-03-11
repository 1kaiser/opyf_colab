import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

class Mlp(nn.Module):
    in_features: int
    hidden_features: int
    out_features: Optional[int] = None
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_features = self.out_features if self.out_features is not None else self.in_features
        x = nn.Dense(self.hidden_features, use_bias=self.bias, name='fc1')(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Dense(out_features, use_bias=self.bias, name='fc2')(x)
        return x

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.initializers.constant(self.init_values), (self.dim,))
        return x * gamma
