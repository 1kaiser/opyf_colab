import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from typing import Tuple, Optional, Any, List

class ResidualBlock(nn.Module):
    num_features: int
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x):
        # resnet block from Depth Pro
        def block(y):
            y = jax.nn.relu(y)
            y = nn.Conv(self.num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=not self.batch_norm)(y)
            if self.batch_norm:
                y = nn.BatchNorm(use_scale=True, use_bias=True)(y)
            return y
        
        h = block(x)
        h = block(h)
        return x + h

class FeatureFusionBlock2d(nn.Module):
    num_features: int
    deconv: bool = False
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x0, x1=None):
        x = x0
        if x1 is not None:
            res = ResidualBlock(num_features=self.num_features, batch_norm=self.batch_norm, name='resnet1')(x1)
            x = x + res
            
        x = ResidualBlock(num_features=self.num_features, batch_norm=self.batch_norm, name='resnet2')(x)
        
        if self.deconv:
            x = nn.ConvTranspose(self.num_features, kernel_size=(2, 2), strides=(2, 2), padding='VALID', name='deconv', transpose_kernel=True, use_bias=False)(x)
            
        x = nn.Conv(self.num_features, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='out_conv')(x)
        return x

class MultiresConvDecoder(nn.Module):
    dims_encoder: List[int]
    dim_decoder: int

    @nn.compact
    def __call__(self, encodings: List[jnp.ndarray]):
        num_encoders = len(self.dims_encoder)
        projected = []
        for i in range(num_encoders):
            if i == 0 and self.dims_encoder[0] != self.dim_decoder:
                p = nn.Conv(self.dim_decoder, kernel_size=(1, 1), use_bias=False, name=f'convs_{i}')(encodings[i])
            elif i == 0:
                p = encodings[i] 
            else:
                p = nn.Conv(self.dim_decoder, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, name=f'convs_{i}')(encodings[i])
            projected.append(p)
            
        features = projected[-1]
        lowres_features = features
        features = FeatureFusionBlock2d(num_features=self.dim_decoder, deconv=(num_encoders-1 != 0), name=f'fusions_{num_encoders-1}')(features)
        
        for i in range(num_encoders - 2, -1, -1):
            features = FeatureFusionBlock2d(num_features=self.dim_decoder, deconv=(i != 0), name=f'fusions_{i}')(features, projected[i])
            
        return features, lowres_features

class FOVNetwork(nn.Module):
    num_features: int
    use_fov_encoder: bool = False
    vit_config: Optional[dict] = None

    @nn.compact
    def __call__(self, x, lowres_feature):
        if self.use_fov_encoder:
            B, C, H, W = x.shape
            x_down = jax.image.resize(x, (B, C, H//4, W//4), method='bilinear', antialias=False)
            from .vit import ViT
            encoder = ViT(**self.vit_config, name='encoder_0')
            x_feat, _ = encoder(x_down)
            x_feat = x_feat[:, 1:, :] 
            grid = int(math.sqrt(x_feat.shape[1]))
            x_feat = x_feat.reshape((B, grid, grid, x_feat.shape[-1]))
            x_feat = nn.Dense(self.num_features // 2, name='encoder_1')(x_feat)
            h_low = nn.Conv(self.num_features // 2, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), name='downsample_0')(lowres_feature)
            h_low = jax.nn.relu(h_low)
            h = x_feat + h_low
        else:
            h = lowres_feature

        h = nn.Conv(self.num_features // 4, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), name='head_0')(h)
        h = jax.nn.relu(h)
        h = nn.Conv(self.num_features // 8, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), name='head_2')(h)
        h = jax.nn.relu(h)
        h = nn.Conv(1, kernel_size=(6, 6), strides=(1, 1), padding='VALID', name='head_4')(h)
        
        return h.squeeze((1, 2, 3))
