import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, List

def simple_nms(scores: jnp.ndarray, nms_radius: int) -> jnp.ndarray:
    """Fast Non-maximum suppression to remove nearby points"""
    if nms_radius == 0:
        return scores
    
    # In PyTorch: max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
    # JAX pool takes window_shape and strides. padding can be explicit.
    ksize = nms_radius * 2 + 1
    
    def max_pool(x):
        return nn.max_pool(x, window_shape=(ksize, ksize), strides=(1, 1), padding=((nms_radius, nms_radius), (nms_radius, nms_radius)))

    max_mask = (scores == max_pool(scores))
    
    # PyTorch implementation does this twice for some reason
    for _ in range(2):
        supp_mask = max_pool(max_mask.astype(jnp.float32)) > 0
        supp_scores = jnp.where(supp_mask, 0.0, scores)
        new_max_mask = (supp_scores == max_pool(supp_scores))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
        
    return jnp.where(max_mask, scores, 0.0)

class SuperPoint(nn.Module):
    descriptor_dim: int = 256
    nms_radius: int = 4
    max_num_keypoints: Optional[int] = None
    detection_threshold: float = 0.0005
    remove_borders: int = 4

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> dict:
        # image: [B, H, W, 1] (Grayscale)
        
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        # Helper for Conv blocks
        def conv_block(x, out_dims, name):
            x = nn.Conv(out_dims, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name=f'{name}a')(x)
            x = jax.nn.relu(x)
            x = nn.Conv(out_dims, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name=f'{name}b')(x)
            x = jax.nn.relu(x)
            return x

        # Shared Encoder
        x = conv_block(image, c1, 'conv1')
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = conv_block(x, c2, 'conv2')
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = conv_block(x, c3, 'conv3')
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(c4, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='conv4a')(x)
        x = jax.nn.relu(x)
        x = nn.Conv(c4, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='conv4b')(x)
        x = jax.nn.relu(x)
        
        # Detector Head
        cPa = nn.Conv(c5, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='convPa')(x)
        cPa = jax.nn.relu(cPa)
        scores = nn.Conv(65, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='convPb')(cPa)
        
        # Softmax over channels, remove dustbin (last channel)
        scores = jax.nn.softmax(scores, axis=-1)[:, :, :, :-1]
        
        # Reshape to dense scores
        B, H, W, _ = scores.shape
        # PyTorch: scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        # JAX NHWC: [B, H, W, 64]
        scores = scores.reshape((B, H, W, 8, 8))
        scores = scores.transpose((0, 1, 3, 2, 4))
        scores = scores.reshape((B, H * 8, W * 8))
        
        # NMS
        scores = simple_nms(scores[:, :, :, None], self.nms_radius).squeeze(-1)
        
        # Discard borders
        if self.remove_borders > 0:
            pad = self.remove_borders
            # Use slice assignment equivalent in JAX
            # We can't do scores[:, :pad] = -1 directly
            mask = jnp.ones_like(scores)
            # Create a border mask
            # row mask
            mask = mask.at[:, :pad, :].set(0)
            mask = mask.at[:, -pad:, :].set(0)
            # col mask
            mask = mask.at[:, :, :pad].set(0)
            mask = mask.at[:, :, -pad:].set(0)
            scores = jnp.where(mask > 0, scores, -1.0)

        # Descriptor Head
        cDa = nn.Conv(c5, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='convDa')(x)
        cDa = jax.nn.relu(cDa)
        descriptors = nn.Conv(self.descriptor_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='convDb')(cDa)
        
        # L2 Normalize
        descriptors = descriptors / jnp.maximum(jnp.linalg.norm(descriptors, axis=-1, keepdims=True), 1e-12)
        
        return {
            "scores": scores,
            "descriptors": descriptors # [B, H/8, W/8, D]
        }
