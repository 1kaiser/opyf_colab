import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from typing import Tuple, Optional, Any, List
from .vit import ViT

class DepthProEncoder(nn.Module):
    vit_config: dict
    decoder_features: int = 256
    hook_block_ids: List[int] = (5, 11) # For ViT-L/16

    @nn.compact
    def __call__(self, x):
        # x shape: [B, C, H, W]
        B = x.shape[0]
        
        # 1. Create Pyramid
        # Depth Pro original design expects x0=1536, x1=768, x2=384
        # At 768 input: x0=768, x1=384, x2=192
        x0 = x
        x1 = jax.image.resize(x, (B, x.shape[1], x.shape[2]//2, x.shape[3]//2), method='bilinear', antialias=False)
        x2 = jax.image.resize(x, (B, x.shape[1], x.shape[2]//4, x.shape[3]//4), method='bilinear', antialias=False)
        
        # We MUST ensure the lowest level patches are 384x384 for concatenation
        # If x2 is 192x192, we upscale it to 384
        if x2.shape[-1] != 384:
            x2_in = jax.image.resize(x2, (B, x2.shape[1], 384, 384), method='bilinear', antialias=False)
        else:
            x2_in = x2
            
        # 2. Split into patches (384x384)
        def split(img, overlap_ratio):
            patch_size = 384
            if img.shape[-1] < patch_size:
                # Upscale if too small
                img = jax.image.resize(img, (img.shape[0], img.shape[1], patch_size, patch_size), method='bilinear', antialias=False)
            
            stride = int(patch_size * (1 - overlap_ratio))
            patches = []
            steps = (img.shape[-1] - patch_size) // stride + 1
            for j in range(steps):
                for i in range(steps):
                    p = img[:, :, j*stride:j*stride+patch_size, i*stride:i*stride+patch_size]
                    patches.append(p)
            return jnp.concatenate(patches, axis=0)

        x0_patches = split(x0, 0.25) 
        x1_patches = split(x1, 0.5)  
        x2_patches_batch = split(x2_in, 0.0) # 1x1
        
        # Total patches
        x_pyramid_patches = jnp.concatenate([x0_patches, x1_patches, x2_patches_batch], axis=0)
        
        # 3. Batched ViT
        patch_encoder = ViT(**self.vit_config, name='patch_encoder')
        x_out, hooks = patch_encoder(x_pyramid_patches)
        
        # x_out: [B*35, 1+N_patches, D]
        # We need to reshape features to 2D
        def reshape_feature(feat):
            # feat: [B_total, 1+N_patches, D]
            # remove cls token
            feat = feat[:, 1:, :]
            grid_size = int(math.sqrt(feat.shape[1]))
            # [B_total, grid, grid, D]
            feat = feat.reshape((feat.shape[0], grid_size, grid_size, feat.shape[-1]))
            return feat

        x_pyramid_features = reshape_feature(x_out)
        hook0 = reshape_feature(hooks[self.hook_block_ids[0]])
        hook1 = reshape_feature(hooks[self.hook_block_ids[1]])
        
        num_x0 = x0_patches.shape[0]
        num_x1 = x1_patches.shape[0]
        num_x2 = x2_patches_batch.shape[0]
        
        # 4. Merge
        def merge(feat, num_p, padding):
            steps = int(math.sqrt(num_p // B))
            gh, gw, D = feat.shape[1], feat.shape[2], feat.shape[3]
            
            rows = []
            for j in range(steps):
                cols = []
                for i in range(steps):
                    idx = j * steps + i
                    p = feat[B*idx : B*(idx+1)]
                    
                    if j != 0: p = p[:, padding:, :, :]
                    if i != 0: p = p[:, :, padding:, :]
                    if j != steps - 1: p = p[:, :-padding, :, :]
                    if i != steps - 1: p = p[:, :, :-padding, :]
                    cols.append(p)
                rows.append(jnp.concatenate(cols, axis=2))
            return jnp.concatenate(rows, axis=1)

        x_latent0_features = merge(hook0[:num_x0], num_x0, 3)
        x_latent1_features = merge(hook1[:num_x0], num_x0, 3)
        
        x0_enc, x1_enc, x2_enc = jnp.split(x_pyramid_features, [num_x0, num_x0 + num_x1], axis=0)
        
        x0_features = merge(x0_enc, num_x0, 3)
        x1_features = merge(x1_enc, num_x1, 6)
        x2_features = x2_enc # 1x1
        
        # Global Image Encoder - use small x2 directly (already 384x384 resized or original)
        image_encoder = ViT(**self.vit_config, name='image_encoder')
        x_global_out, _ = image_encoder(x2_in)
        x_global_features = reshape_feature(x_global_out)
        
        # 5. Projection and Upsampling
        def project_upsample(feat, dim_in, dim_out, layers, name, dim_int=None):
            if dim_int is None: dim_int = dim_out
            # manual sequential
            h = nn.Conv(dim_int, kernel_size=(1, 1), use_bias=False, name=f'{name}_0')(feat)
            for i in range(layers):
                h = nn.ConvTranspose(dim_out, kernel_size=(2, 2), strides=(2, 2), padding='VALID', name=f'{name}_{i+1}', transpose_kernel=True, use_bias=False)(h)
            return h

        # Default dims: [256, 512, 1024, 1024]
        # Latent0 -> decoder_features (256)
        x_latent0_features = project_upsample(x_latent0_features, 1024, self.decoder_features, 3, 'upsample_latent0', dim_int=256)
        x_latent1_features = project_upsample(x_latent1_features, 1024, 256, 2, 'upsample_latent1')
        
        x0_features = project_upsample(x0_features, 1024, 512, 1, 'upsample0')
        x1_features = project_upsample(x1_features, 1024, 1024, 1, 'upsample1')
        x2_features = project_upsample(x2_features, 1024, 1024, 1, 'upsample2')
        
        # Lowres fusion
        x_global_features = nn.ConvTranspose(1024, kernel_size=(2, 2), strides=(2, 2), padding='VALID', name='upsample_lowres', transpose_kernel=True, use_bias=True)(x_global_features)
        x_global_features = nn.Conv(1024, kernel_size=(1, 1), use_bias=True, name='fuse_lowres')(jnp.concatenate([x2_features, x_global_features], axis=-1))
        
        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features
        ]
