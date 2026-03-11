import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Optional, Union, Any
from .head_act import activate_head

class ResidualConvUnit(nn.Module):
    features: int
    activation: Any = jax.nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.activation(x)
        h = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='conv1')(h)
        h = self.activation(h)
        h = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='conv2')(h)
        return x + h

class FeatureFusionBlock(nn.Module):
    features: int
    activation: Any = jax.nn.relu
    deconv: bool = False
    expand: bool = False
    align_corners: bool = True
    has_residual: bool = True

    @nn.compact
    def __call__(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = ResidualConvUnit(self.features, activation=self.activation, name='resConfUnit1')(xs[1])
            output = output + res
            
        output = ResidualConvUnit(self.features, activation=self.activation, name='resConfUnit2')(output)
        
        if size is None:
            target_shape = (output.shape[0], output.shape[1] * 2, output.shape[2] * 2, output.shape[3])
        else:
            target_shape = (output.shape[0], size[0], size[1], output.shape[3])
            
        output = jax.image.resize(output, target_shape, method='bilinear', antialias=False)
        
        out_features = self.features // 2 if self.expand else self.features
        output = nn.Conv(out_features, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='out_conv')(output)
        return output

class DPTHead(nn.Module):
    dim_in: int
    patch_size: int = 14
    output_dim: int = 4
    activation: str = "inv_log"
    conf_activation: str = "expp1"
    features: int = 256
    out_channels: List[int] = (256, 512, 1024, 1024)
    intermediate_layer_idx: List[int] = (4, 11, 17, 23)
    pos_embed: bool = True
    feature_only: bool = False
    down_ratio: int = 1

    @nn.compact
    def __call__(self, aggregated_tokens_list: List[jnp.ndarray], images: jnp.ndarray, patch_start_idx: int) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        out = []
        norm = nn.LayerNorm(epsilon=1e-6, name='norm')
        
        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = x.reshape((B * S, -1, x.shape[-1]))
            x = norm(x)
            x = x.reshape((B * S, patch_h, patch_w, x.shape[-1]))
            x = nn.Conv(self.out_channels[i], kernel_size=(1, 1), strides=(1, 1), padding='VALID', name=f'projects_{i}')(x)
            
            if i == 0:
                x = nn.ConvTranspose(self.out_channels[0], kernel_size=(4, 4), strides=(4, 4), padding='VALID', name='resize_layers_0', transpose_kernel=True)(x)
            elif i == 1:
                x = nn.ConvTranspose(self.out_channels[1], kernel_size=(2, 2), strides=(2, 2), padding='VALID', name='resize_layers_1', transpose_kernel=True)(x)
            elif i == 2:
                pass
            elif i == 3:
                x = nn.Conv(self.out_channels[3], kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), name='resize_layers_3')(x)
            
            out.append(x)

        # Fusion
        layer1_rn = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), use_bias=False, name='scratch_layer1_rn')(out[0])
        layer2_rn = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), use_bias=False, name='scratch_layer2_rn')(out[1])
        layer3_rn = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), use_bias=False, name='scratch_layer3_rn')(out[2])
        layer4_rn = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), use_bias=False, name='scratch_layer4_rn')(out[3])
        
        res = FeatureFusionBlock(self.features, has_residual=False, name='scratch_refinenet4')(layer4_rn, size=layer3_rn.shape[1:3])
        res = FeatureFusionBlock(self.features, name='scratch_refinenet3')(res, layer3_rn, size=layer2_rn.shape[1:3])
        res = FeatureFusionBlock(self.features, name='scratch_refinenet2')(res, layer2_rn, size=layer1_rn.shape[1:3])
        res = FeatureFusionBlock(self.features, name='scratch_refinenet1')(res, layer1_rn)
        
        # Final upsample to match image resolution
        res = jax.image.resize(res, (res.shape[0], H // self.down_ratio, W // self.down_ratio, res.shape[3]), method='bilinear', antialias=False)

        res = nn.Conv(self.features // 2 if not self.feature_only else self.features, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='scratch_output_conv1')(res)
        
        if self.feature_only:
            return res.reshape((B, S) + res.shape[1:])
            
        res = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name='scratch_output_conv2_0')(res)
        res = jax.nn.relu(res)
        res = nn.Conv(self.output_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='scratch_output_conv2_2')(res)
        
        preds, conf = activate_head(res, activation=self.activation, conf_activation=self.conf_activation)
        
        preds = preds.reshape((B, S) + preds.shape[1:])
        conf = conf.reshape((B, S) + conf.shape[1:])
        return preds, conf
