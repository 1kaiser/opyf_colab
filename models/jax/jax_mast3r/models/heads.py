import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional, Any, List
import jax

def pixel_shuffle(x, factor):
    # x: (B, H, W, C_rr)
    # PyTorch pixel_shuffle expects (B, C * r^2, H, W) -> (B, C, H*r, W*r)
    # The elements for each channel's r^2 block are contiguous.
    B, H, W, Crr = x.shape
    c = Crr // (factor**2)
    # 1. Reshape to (B, H, W, c, factor, factor)
    x = x.reshape(B, H, W, c, factor, factor)
    # 2. Transpose to (B, H, factor, W, factor, c)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    # 3. Reshape to (B, H*factor, W*factor, c)
    x = x.reshape(B, H * factor, W * factor, c)
    return x

def reg_dense_depth(xyz, mode):
    # xyz: (B, H, W, 3)
    mode_name, vmin, vmax = mode
    if mode_name == 'exp':
        d = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        xyz_unit = xyz / jnp.clip(d, min=1e-8)
        return xyz_unit * jnp.expm1(d)
    return xyz # simplified for now

def reg_dense_conf(x, mode):
    # x: (B, H, W)
    mode_name, vmin, vmax = mode
    if mode_name == 'exp':
        return vmin + jnp.exp(x)
    return x

class ResidualConvUnit(nn.Module):
    features: int
    use_bn: bool = False

    @nn.compact
    def __call__(self, x):
        h = nn.relu(x)
        h = nn.Conv(self.features, kernel_size=(3, 3), name='conv1', use_bias=not self.use_bn)(h)
        if self.use_bn:
            h = nn.BatchNorm(use_running_average=True, name='bn1')(h)
        h = nn.relu(h)
        h = nn.Conv(self.features, kernel_size=(3, 3), name='conv2', use_bias=not self.use_bn)(h)
        if self.use_bn:
            h = nn.BatchNorm(use_running_average=True, name='bn2')(h)
        return x + h

def upsample_bilinear(x, factor=2, align_corners=True):
    B, H, W, C = x.shape
    if align_corners:
        # PyTorch align_corners=True mapping
        new_H, new_W = H * factor, W * factor
        # For factor=2, this maps 0->0, H-1 -> 2H-1? No.
        # Actually PyTorch factor=2 with align_corners=True on size N gives size 2N.
        # Wait, if input is 32, output is 64.
        return jax.image.resize(x, (B, H * factor, W * factor, C), method='bilinear', antialias=False)
    else:
        return jax.image.resize(x, (B, H * factor, W * factor, C), method='bilinear', antialias=False)

def resize_bilinear_align_corners(x, shape):
    """Resizes an image using bilinear interpolation with align_corners=True, 
    matching PyTorch's F.interpolate behavior.
    x: (B, H, W, C)
    shape: (B, H_new, W_new, C)
    """
    B, H, W, C = x.shape
    _, H_new, W_new, _ = shape
    
    if H == H_new and W == W_new:
        return x

    # Generate coordinates for output grid
    # For align_corners=True, we map [0, H_new-1] to [0, H-1]
    h_coords = jnp.linspace(0, H - 1, H_new)
    w_coords = jnp.linspace(0, W - 1, W_new)
    
    # Create meshgrid
    grid_h, grid_w = jnp.meshgrid(h_coords, w_coords, indexing='ij')
    coords = jnp.stack([grid_h, grid_w], axis=0)
    
    # Reshape x to (H, W, B*C) to map all feature maps in parallel
    x_reshaped = x.transpose(1, 2, 0, 3).reshape(H, W, -1)
    
    # map_coordinates order=1 is bilinear. mode='nearest' handles boundaries.
    mapped = jax.vmap(lambda feat: jax.scipy.ndimage.map_coordinates(feat, coords, order=1, mode='nearest'), 
                      in_axes=-1, out_axes=-1)(x_reshaped)
    
    return mapped.reshape(H_new, W_new, B, C).transpose(2, 0, 1, 3)

class FeatureFusionBlock(nn.Module):
    features: int
    use_bn: bool = False
    align_corners: bool = True

    @nn.compact
    def __call__(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = ResidualConvUnit(self.features, use_bn=self.use_bn, name='resConfUnit1')(xs[1])
            output = output + res
        
        output = ResidualConvUnit(self.features, use_bn=self.use_bn, name='resConfUnit2')(output)
        
        B, H, W, C = output.shape
        # Use our custom align_corners=True resize
        output = resize_bilinear_align_corners(output, (B, H * 2, W * 2, C))
        
        output = nn.Conv(self.features, kernel_size=(1, 1), name='out_conv')(output)
        return output

class LocalFeatureHead(nn.Module):
    in_features: int
    hidden_features: int
    out_features: int
    patch_size: int = 16

    @nn.compact
    def __call__(self, x, image_size: Tuple[int, int]):
        # x: (B, N, C)
        B, N, _ = x.shape
        H, W = image_size
        
        x = nn.Dense(self.hidden_features, name='fc1')(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_features, name='fc2')(x)
        
        # Reshape to spatial grid of patches
        # rearrange(x, 'b (nh nw) (p1 p2 c) -> b nh nw (p1 p2 c)')
        nh, nw = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, nh, nw, -1)
        
        # Pixel Shuffle to target resolution
        x = pixel_shuffle(x, self.patch_size)
        return x

class DPTHead(nn.Module):
    num_channels: int
    feature_dim: int = 256
    last_dim: int = 128
    layer_dims: Tuple[int, ...] = (96, 192, 384, 768)

    @nn.compact
    def __call__(self, layers: List[jnp.ndarray], image_size: Tuple[int, int]):
        B, N, _ = layers[0].shape
        H, W = image_size
        nh = H // 16
        nw = W // 16
        
        # 1. Reshape to spatial
        layers = [l.reshape(B, nh, nw, -1) for l in layers]
        
        # 2. Act postprocess
        l1_0 = nn.Conv(self.layer_dims[0], kernel_size=(1, 1), name='act_postprocess_0_0')(layers[0])
        l1 = nn.ConvTranspose(self.layer_dims[0], kernel_size=(4, 4), strides=(4, 4), padding='VALID', 
                              transpose_kernel=True, name='act_postprocess_0_1')(l1_0)
        
        l2_0 = nn.Conv(self.layer_dims[1], kernel_size=(1, 1), name='act_postprocess_1_0')(layers[1])
        l2 = nn.ConvTranspose(self.layer_dims[1], kernel_size=(2, 2), strides=(2, 2), padding='VALID', 
                              transpose_kernel=True, name='act_postprocess_1_1')(l2_0)
        
        l3 = nn.Conv(self.layer_dims[2], kernel_size=(1, 1), name='act_postprocess_2_0')(layers[2])
        
        l4_0 = nn.Conv(self.layer_dims[3], kernel_size=(1, 1), name='act_postprocess_3_0')(layers[3])
        # Use explicit padding=1 to match PyTorch's Conv2d(padding=1) for stride=2
        l4 = nn.Conv(self.layer_dims[3], kernel_size=(3, 3), strides=(2, 2), padding=1, name='act_postprocess_3_1')(l4_0)
        
        layers_rn = [
            nn.Conv(self.feature_dim, kernel_size=(3, 3), use_bias=False, name=f'scratch_layer_rn_{i}')(l)
            for i, l in enumerate([l1, l2, l3, l4])
        ]
        
        # 4. RefineNet blocks
        path_4 = FeatureFusionBlock(self.feature_dim, name='scratch_refinenet4')(layers_rn[3])
        path_3 = FeatureFusionBlock(self.feature_dim, name='scratch_refinenet3')(path_4, layers_rn[2])
        path_2 = FeatureFusionBlock(self.feature_dim, name='scratch_refinenet2')(path_3, layers_rn[1])
        path_1 = FeatureFusionBlock(self.feature_dim, name='scratch_refinenet1')(path_2, layers_rn[0])
        
        # 5. Output head
        x = nn.Conv(self.feature_dim // 2, kernel_size=(3, 3), name='head_0')(path_1)
        B, H_f, W_f, C_f = x.shape
        x = resize_bilinear_align_corners(x, (B, H_f * 2, W_f * 2, C_f))
        x = nn.Conv(self.last_dim, kernel_size=(3, 3), name='head_2')(x)
        x = nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(1, 1), name='head_4')(x)
        
        return x
