import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional, Any, Iterable

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, mask: Optional[jnp.ndarray] = None):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn = jnp.where(mask, -jnp.inf, attn)

        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop, deterministic=True)(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        x = nn.Dropout(self.proj_drop, deterministic=True)(x)
        return x

class CrossAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, context, mask: Optional[jnp.ndarray] = None):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        # Query from x
        q = nn.Dense(self.dim, use_bias=self.qkv_bias, name='projq')(x)
        q = q.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Key and Value from context
        kv = nn.Dense(self.dim * 2, use_bias=self.qkv_bias, name='projkv')(context)
        kv = kv.reshape(B, -1, 2, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn = jnp.where(mask, -jnp.inf, attn)

        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop, deterministic=True)(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.proj_bias, name='proj')(x)
        x = nn.Dropout(self.proj_drop, deterministic=True)(x)
        return x


class DecoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    proj_bias: bool = True
    ffn_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.

    @nn.compact
    def __call__(self, x, context, mask: Optional[jnp.ndarray] = None):
        # Self-attention
        h = nn.LayerNorm(epsilon=1e-6, name='norm1')(x)
        h = Attention(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')(h, mask=mask)
        x = x + h
        
        # Cross-attention
        h = nn.LayerNorm(epsilon=1e-6, name='norm2')(x)
        h = CrossAttention(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias, attn_drop=self.attn_drop, proj_drop=self.drop, name='cross_attn')(h, context)
        x = x + h
        
        # MLP
        h = nn.LayerNorm(epsilon=1e-6, name='norm3')(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        h = nn.Sequential([
            nn.Dense(mlp_hidden_dim, use_bias=self.ffn_bias, name='fc1'),
            nn.gelu,
            nn.Dropout(self.drop, deterministic=True),
            nn.Dense(self.dim, use_bias=self.ffn_bias, name='fc2'),
            nn.Dropout(self.drop, deterministic=True)
        ])(h)
        x = x + h
        return x

class PatchEmbed(nn.Module):
    img_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (16, 16)
    in_chans: int = 3
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        B, C, H, W = x.shape
        x = nn.Conv(self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, name='proj')(x)
        x = x.transpose(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        return x

class VisionTransformer(nn.Module):
    img_size: Any = (224, 224)
    patch_size: Any = (16, 16)
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    
    def setup(self):
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, 
                       in_chans=self.in_chans, embed_dim=self.embed_dim)
        
        self.blocks = [
            EncoderBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, 
                  qkv_bias=self.qkv_bias, name=f'blocks.{i}')
            for i in range(self.depth)
        ]
        self.norm = nn.LayerNorm(epsilon=1e-6)

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]
        if x.ndim == 4: # Image input
            x = self.patch_embed(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x

class FlaxAsymmetricMASt3R(nn.Module):
    """
    Flax model for MASt3R. This will be built out further.
    """
    enc_depth: int = 24
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    dec_embed_dim: int = 768
    enc_num_heads: int = 16
    dec_num_heads: int = 12
    patch_size: int = 16
    
    def setup(self):
        # Encoder (ViT-L)
        self.encoder = VisionTransformer(
            patch_size=(self.patch_size, self.patch_size),
            embed_dim=self.enc_embed_dim,
            depth=self.enc_depth,
            num_heads=self.enc_num_heads,
            name='encoder'
        )
        
        # Decoder blocks
        self.decoder_blocks = [
            DecoderBlock(dim=self.dec_embed_dim, num_heads=self.dec_num_heads,
                         mlp_ratio=4., qkv_bias=True, name=f'dec_blocks.{i}')
            for i in range(self.dec_depth)
        ]
        self.decoder_norm = nn.LayerNorm(epsilon=1e-6, name='dec_norm')
        
        # Initial query token for decoder (similar to class token but for decoding)
        self.query_token = self.param('query_token', nn.initializers.zeros, (1, 1, self.dec_embed_dim))

    @nn.compact
    def __call__(self, view1, view2):
        # Dummy input for shape inference
        if view1 is None:
            view1 = jnp.ones((1, 3, 512, 512))
            
        # 1. Encode images
        feat1 = self.encoder(view1) # (B, N, D_enc)
        
        # Initialize decoder query with query_token
        query = jnp.tile(self.query_token, (feat1.shape[0], feat1.shape[1], 1)) # (B, N, D_dec) - shape needs to match feat1

        # 2. Decode features using cross-attention
        for i, block in enumerate(self.decoder_blocks):
            query = block(query, feat1)
            
        query = self.decoder_norm(query)
        
        # TODO: Add heads for pts3d, conf, desc
        
        return query

print("Updated Flax model skeleton with DecoderBlock and CrossAttention.")