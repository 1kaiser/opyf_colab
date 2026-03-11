import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, List
from .encoder import DepthProEncoder
from .decoder import MultiresConvDecoder, FOVNetwork

class DepthPro(nn.Module):
    vit_config: dict
    decoder_features: int = 256
    last_dims: Tuple[int, int] = (32, 1)

    @nn.compact
    def __call__(self, x):
        # x shape: [B, C, H, W]
        # 1. Encoder
        encoder = DepthProEncoder(vit_config=self.vit_config, decoder_features=self.decoder_features, name='encoder')
        encodings = encoder(x)
        
        # 2. Decoder
        # dims_encoder for decoder: [decoder_features, 256, 512, 1024, 1024]
        decoder = MultiresConvDecoder(dims_encoder=[self.decoder_features, 256, 512, 1024, 1024], dim_decoder=self.decoder_features, name='decoder')
        features, lowres_features = decoder(encodings)
        
        # 3. Head
        # head.0: nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1)
        h = nn.Conv(self.decoder_features // 2, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='head_0')(features)
        
        # head.1: nn.ConvTranspose2d(dim_decoder // 2, dim_decoder // 2, kernel_size=2, stride=2)
        h = nn.ConvTranspose(self.decoder_features // 2, kernel_size=(2, 2), strides=(2, 2), padding='VALID', name='head_1', transpose_kernel=True, use_bias=True)(h)
        
        # head.2: nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1)
        h = nn.Conv(self.last_dims[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='head_2')(h)
        h = jax.nn.relu(h) # head[3]
        
        # head.4: nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0)
        h = nn.Conv(self.last_dims[1], kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='head_4')(h)
        canonical_inverse_depth = jax.nn.relu(h) # head[5]
        
        # 4. FOV
        fov_net = FOVNetwork(num_features=self.decoder_features, use_fov_encoder=True, vit_config=self.vit_config, name='fov')
        fov_deg = fov_net(x, lowres_features)
        
        return canonical_inverse_depth, fov_deg
