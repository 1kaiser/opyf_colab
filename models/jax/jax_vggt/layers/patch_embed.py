import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Union, Callable

class PatchEmbed(nn.Module):
    img_size: Union[int, Tuple[int, int]] = 224
    patch_size: Union[int, Tuple[int, int]] = 16
    in_chans: int = 3
    embed_dim: int = 768
    flatten_embedding: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, C, H, W]
        # JAX Conv expects NHWC
        x = x.transpose(0, 2, 3, 1)
        
        patch_size = self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='VALID',
            name='proj'
        )(x)
        
        B, H, W, C = x.shape
        if self.flatten_embedding:
            x = x.reshape(B, -1, C)
        return x
