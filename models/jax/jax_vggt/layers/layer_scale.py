import jax
import jax.numpy as jnp
import flax.linen as nn

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.initializers.constant(self.init_values), (self.dim,))
        return x * gamma
