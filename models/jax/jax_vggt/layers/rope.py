import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict

class PositionGetter:
    def __call__(self, batch_size: int, height: int, width: int) -> jnp.ndarray:
        y_coords = jnp.arange(height)
        x_coords = jnp.arange(width)
        # JAX meshgrid for positions
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        positions = jnp.stack([yy, xx], axis=-1).reshape(-1, 2)
        return jnp.broadcast_to(positions[None, :, :], (batch_size, height * width, 2))

class RotaryPositionEmbedding2D(nn.Module):
    frequency: float = 100.0
    scaling_factor: float = 1.0

    def _compute_frequency_components(self, dim: int, max_pos: int, dtype):
        exponents = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
        inv_freq = 1.0 / (self.frequency ** exponents)
        positions = jnp.arange(max_pos, dtype=jnp.float32)
        angles = jnp.einsum("i,j->ij", positions, inv_freq).astype(dtype)
        # angles: [max_pos, dim//2]
        angles = jnp.concatenate([angles, angles], axis=-1)
        # angles: [max_pos, dim]
        return jnp.cos(angles), jnp.sin(angles)

    def _rotate_features(self, x: jnp.ndarray) -> jnp.ndarray:
        feature_dim = x.shape[-1]
        x1 = x[..., :feature_dim // 2]
        x2 = x[..., feature_dim // 2:]
        return jnp.concatenate([-x2, x1], axis=-1)

    def _apply_1d_rope(self, tokens: jnp.ndarray, positions: jnp.ndarray, cos_comp: jnp.ndarray, sin_comp: jnp.ndarray) -> jnp.ndarray:
        # tokens: [B, H, N, D/2]
        # positions: [B, N]
        # cos_comp, sin_comp: [max_pos, D/2]
        
        # In JAX we can use indexing
        cos = cos_comp[positions][:, None, :, :] # [B, 1, N, D/2]
        sin = sin_comp[positions][:, None, :, :] # [B, 1, N, D/2]
        
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def __call__(self, tokens: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        # tokens: [B, H, N, D]
        # positions: [B, N, 2]
        
        feature_dim = tokens.shape[-1] // 2
        max_pos = jnp.max(positions).astype(jnp.int32) + 1
        
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, 1024, tokens.dtype) # Use large enough fixed max_pos
        
        # Explicitly get the components for actual positions to avoid large constant in graph
        # Wait, in JIT we prefer fixed shapes. 1024 is safe for 518x518 images (37x37 patches).
        
        vertical_features = tokens[..., :feature_dim]
        horizontal_features = tokens[..., feature_dim:]
        
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0].astype(jnp.int32), cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1].astype(jnp.int32), cos_comp, sin_comp)
        
        return jnp.concatenate([vertical_features, horizontal_features], axis=-1)
