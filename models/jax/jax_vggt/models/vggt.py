import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Optional, Any, Union

from .aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead

class VGGT(nn.Module):
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    enable_camera: bool = True
    enable_point: bool = True
    enable_depth: bool = True

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> dict:
        # images: [B, S, 3, H, W]
        aggregated_tokens_list, patch_start_idx = Aggregator(
            img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim, name='aggregator'
        )(images)
        
        predictions = {}
        
        if self.enable_camera:
            pose_enc_list = CameraHead(dim_in=2 * self.embed_dim, name='camera_head')(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
            
        if self.enable_depth:
            depth, depth_conf = DPTHead(
                dim_in=2 * self.embed_dim, output_dim=2, name='depth_head'
            )(aggregated_tokens_list, images, patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
            
        if self.enable_point:
            pts3d, pts3d_conf = DPTHead(
                dim_in=2 * self.embed_dim, output_dim=4, name='point_head'
            )(aggregated_tokens_list, images, patch_start_idx)
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf
            
        return predictions
