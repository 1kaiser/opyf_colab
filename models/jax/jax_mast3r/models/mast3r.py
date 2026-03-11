
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional, Any, List
from .vit import Attention, MLP, EncoderBlock, PatchEmbed, RoPE2D
from .heads import LocalFeatureHead, DPTHead, reg_dense_depth, reg_dense_conf

class CrossAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, context, pos_x: Optional[jnp.ndarray] = None, pos_ctx: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        # Query from x
        q = nn.Dense(self.dim, use_bias=self.qkv_bias, name='projq')(x)
        q = q.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Key and Value from context
        k = nn.Dense(self.dim, use_bias=self.qkv_bias, name='projk')(context)
        k = k.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = nn.Dense(self.dim, use_bias=self.qkv_bias, name='projv')(context)
        v = v.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        if pos_x is not None and pos_ctx is not None:
            rope = RoPE2D()
            q = rope(q, pos_x)
            k = rope(k, pos_ctx)

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn = jnp.where(mask, -jnp.inf, attn)

        attn = nn.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        return x

class DecoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    proj_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.

    @nn.compact
    def __call__(self, x, context, pos_x: Optional[jnp.ndarray] = None, pos_ctx: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
        # Self-attention
        h = nn.LayerNorm(epsilon=1e-5, name='norm1')(x)
        h = Attention(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')(h, pos=pos_x, mask=mask)
        x = x + h
        
        # Context normalization
        context_norm = nn.LayerNorm(epsilon=1e-5, name='norm_y')(context)
        
        # Cross-attention
        h = nn.LayerNorm(epsilon=1e-5, name='norm2')(x)
        h = CrossAttention(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias, attn_drop=self.attn_drop, proj_drop=self.drop, name='cross_attn')(h, context_norm, pos_x=pos_x, pos_ctx=pos_ctx)
        x = x + h
        
        # MLP
        h = nn.LayerNorm(epsilon=1e-5, name='norm3')(x)
        h = MLP(hidden_dim=int(self.dim * self.mlp_ratio), out_dim=self.dim, drop=self.drop, name='mlp')(h)
        x = x + h
        return x, context

class MASt3RHead(nn.Module):
    num_channels: int
    feature_dim: int
    last_dim: int
    enc_embed_dim: int
    dec_embed_dim: int
    local_feat_dim: int = 24
    patch_size: int = 16
    two_confs: bool = False
    depth_mode: Tuple[Any, ...] = ('exp', -float('inf'), float('inf'))
    conf_mode: Tuple[Any, ...] = ('exp', 1, float('inf'))
    hooks: Tuple[int, ...] = (0, 6, 9, 12)

    def setup(self):
        self.dpt = DPTHead(num_channels=self.num_channels, feature_dim=self.feature_dim, last_dim=self.last_dim)
        
        out_features = (self.local_feat_dim + self.two_confs) * self.patch_size**2
        self.head_local_features = LocalFeatureHead(in_features=self.enc_embed_dim + self.dec_embed_dim, 
                                                    hidden_features=7168, 
                                                    out_features=out_features,
                                                    patch_size=self.patch_size)

    def __call__(self, all_tokens, feat, image_size):
        # 1. Select tokens for DPT using hooks
        dpt_tokens = [all_tokens[h] for h in self.hooks]
        
        # 2. DPT for 3D points and confidence
        out = self.dpt(dpt_tokens, image_size=image_size) # (B, H, W, 4)
        
        pts3d = reg_dense_depth(out[..., 0:3], mode=self.depth_mode)
        conf = reg_dense_conf(out[..., 3], mode=self.conf_mode)
        
        # 3. Local features head
        local_feat_out = self.head_local_features(jnp.concatenate([feat, all_tokens[-1]], axis=-1), image_size=image_size)
        
        desc = local_feat_out[..., :self.local_feat_dim]
        desc = desc / jnp.clip(jnp.linalg.norm(desc, axis=-1, keepdims=True), min=1e-8)
        
        if self.two_confs:
            desc_conf = reg_dense_conf(local_feat_out[..., self.local_feat_dim], mode=self.conf_mode)
            return pts3d, conf, desc, desc_conf
        
        return pts3d, conf, desc

class FlaxAsymmetricMASt3R(nn.Module):
    enc_depth: int = 24
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    dec_embed_dim: int = 768
    enc_num_heads: int = 16
    dec_num_heads: int = 12
    patch_size: int = 16
    has_conf: bool = True
    two_confs: bool = True
    
    def setup(self):
        # 1. Encoder (Siamese)
        self.patch_embed = PatchEmbed(patch_size=(self.patch_size, self.patch_size), embed_dim=self.enc_embed_dim, name='patch_embed')
        self.enc_blocks = [
            EncoderBlock(dim=self.enc_embed_dim, num_heads=self.enc_num_heads, qkv_bias=True, name=f'enc_blocks.{i}')
            for i in range(self.enc_depth)
        ]
        self.enc_norm = nn.LayerNorm(epsilon=1e-5, name='enc_norm')
        
        # 2. Decoder
        self.decoder_embed = nn.Dense(self.dec_embed_dim, name='decoder_embed')
        self.dec_blocks = [
            DecoderBlock(dim=self.dec_embed_dim, num_heads=self.dec_num_heads, qkv_bias=True, name=f'dec_blocks.{i}')
            for i in range(self.dec_depth)
        ]
        self.dec_blocks2 = [
            DecoderBlock(dim=self.dec_embed_dim, num_heads=self.dec_num_heads, qkv_bias=True, name=f'dec_blocks2.{i}')
            for i in range(self.dec_depth)
        ]
        self.dec_norm = nn.LayerNorm(epsilon=1e-5, name='dec_norm')
        
        # 3. Heads
        num_channels = 3 + self.has_conf
        self.downstream_head1 = MASt3RHead(num_channels=num_channels, feature_dim=256, last_dim=128, 
                                          enc_embed_dim=self.enc_embed_dim, dec_embed_dim=self.dec_embed_dim,
                                          patch_size=self.patch_size, two_confs=self.two_confs,
                                          name='downstream_head1')
        self.downstream_head2 = MASt3RHead(num_channels=num_channels, feature_dim=256, last_dim=128, 
                                          enc_embed_dim=self.enc_embed_dim, dec_embed_dim=self.dec_embed_dim,
                                          patch_size=self.patch_size, two_confs=self.two_confs,
                                          name='downstream_head2')

    def __call__(self, img1, img2):
        # 1. Encode images
        def encode(img):
            x, pos = self.patch_embed(img)
            for blk in self.enc_blocks:
                x = blk(x, pos=pos)
            return self.enc_norm(x), pos
        
        feat1, pos1 = encode(img1)
        feat2, pos2 = encode(img2)
        
        # 2. Project to decoder dimension
        q1 = self.decoder_embed(feat1)
        q2 = self.decoder_embed(feat2)
        
        # 3. Dual side decoding with token capture (13 tokens total, excluding projection)
        dec1_all = [feat1]
        dec2_all = [feat2]
        
        for i in range(self.dec_depth):
            q1_prev, q2_prev = q1, q2
            q1, _ = self.dec_blocks[i](q1_prev, q2_prev, pos_x=pos1, pos_ctx=pos2)
            q2, _ = self.dec_blocks2[i](q2_prev, q1_prev, pos_x=pos2, pos_ctx=pos1)
            dec1_all.append(q1)
            dec2_all.append(q2)
            
        q1_final = self.dec_norm(q1)
        q2_final = self.dec_norm(q2)
        
        # Update last token with normalized version
        dec1_all[-1] = q1_final
        dec2_all[-1] = q2_final
        
        # 4. Heads
        H, W = img1.shape[-2:]
        res1 = self.downstream_head1(dec1_all, feat1, image_size=(H, W))
        res2 = self.downstream_head2(dec2_all, feat2, image_size=(H, W))
        
        # Return everything needed for debugging
        # The decoder lists (decX_all) contain the raw encoder output at index 0, 
        # followed by the 12 decoder block outputs.
        return (res1, res2, (feat1, feat2, dec1_all[1:], dec2_all[1:]))
