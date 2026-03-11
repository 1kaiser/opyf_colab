import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Callable, List

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    # x shape: [..., D]
    # Unflatten last dim to [..., D//2, 2]
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    # Stack (-x2, x1) and flatten
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape[:-2] + (-1,))

def apply_rotary_emb(freqs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    # freqs shape: [2, 1, N, D]
    # t shape: [B, H, N, D]
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class LearnableFourierPositionalEncoding(nn.Module):
    M: int
    dim: int
    F_dim: Optional[int] = None
    gamma: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: [B, N, M]
        f_dim = self.F_dim if self.F_dim is not None else self.dim
        wr = nn.Dense(f_dim // 2, use_bias=False, name="Wr")
        projected = wr(x)
        cosines = jnp.cos(projected)
        sines = jnp.sin(projected)
        # emb shape: [2, B, N, f_dim // 2]
        emb = jnp.stack([cosines, sines], axis=0)
        # Add a dimension for heads later? In PyTorch: .unsqueeze(-3)
        # PyTorch: [2, 1, N, f_dim // 2] if B=1
        emb = jnp.expand_dims(emb, axis=-3)
        # repeat_interleave(2, dim=-1)
        emb = jnp.repeat(emb, 2, axis=-1)
        return emb

class TokenConfidence(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, desc: jnp.ndarray) -> jnp.ndarray:
        # desc shape: [B, N, D]
        # In PyTorch: nn.Sequential(nn.Dense(dim, 1), nn.Sigmoid())
        token = nn.Dense(1, name="token")
        return jax.nn.sigmoid(token(desc)).squeeze(-1)

class Attention(nn.Module):
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # q, k, v shape: [B, H, N, D_head]
        # mask shape: [B, H, N, N] or similar
        d_k = q.shape[-1]
        scale = 1.0 / jnp.sqrt(d_k)
        
        # [B, H, N_q, N_k]
        logits = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * scale
        
        if mask is not None:
            # mask is True for valid positions
            logits = jnp.where(mask, logits, -1e9)
            
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.matmul(attn, v)

class SelfBlock(nn.Module):
    embed_dim: int
    num_heads: int
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, encoding: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: [B, N, D]
        # encoding: [2, B, N, D_head]
        b, n, d = x.shape
        head_dim = self.embed_dim // self.num_heads
        
        wqkv = nn.Dense(3 * self.embed_dim, use_bias=self.bias, name="Wqkv")
        qkv = wqkv(x) # [B, N, 3*D]
        
        # Reshape and transpose to [B, H, N, D_head, 3]
        qkv = qkv.reshape((b, n, self.num_heads, head_dim, 3)).transpose((0, 2, 1, 3, 4))
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        
        q = apply_rotary_emb(encoding, q)
        k = apply_rotary_emb(encoding, k)
        
        context = Attention()(q, k, v, mask=mask) # [B, H, N, D_head]
        
        # Flatten back to [B, N, D]
        message = context.transpose((0, 2, 1, 3)).reshape((b, n, self.embed_dim))
        
        out_proj = nn.Dense(self.embed_dim, use_bias=self.bias, name="out_proj")
        message = out_proj(message)
        
        # FFN
        ffn_0 = nn.Dense(2 * self.embed_dim, name="ffn_0")
        ffn_1 = nn.LayerNorm(epsilon=1e-5, name="ffn_1")
        ffn_3 = nn.Dense(self.embed_dim, name="ffn_3")
        
        res = ffn_0(jnp.concatenate([x, message], axis=-1))
        res = ffn_1(res)
        res = jax.nn.gelu(res, approximate=False)
        res = ffn_3(res)
        
        return x + res

class CrossBlock(nn.Module):
    embed_dim: int
    num_heads: int
    bias: bool = True

    @nn.compact
    def __call__(self, x0: jnp.ndarray, x1: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, n0, d = x0.shape
        _, n1, _ = x1.shape
        head_dim = self.embed_dim // self.num_heads
        scale = 1.0 / jnp.sqrt(head_dim)
        
        to_qk = nn.Dense(self.embed_dim, use_bias=self.bias, name="to_qk")
        to_v = nn.Dense(self.embed_dim, use_bias=self.bias, name="to_v")
        
        qk0, qk1 = to_qk(x0), to_qk(x1)
        v0, v1 = to_v(x0), to_v(x1)
        
        # Unflatten to [B, H, N, D_head]
        qk0 = qk0.reshape((b, n0, self.num_heads, head_dim)).transpose((0, 2, 1, 3))
        qk1 = qk1.reshape((b, n1, self.num_heads, head_dim)).transpose((0, 2, 1, 3))
        v0 = v0.reshape((b, n0, self.num_heads, head_dim)).transpose((0, 2, 1, 3))
        v1 = v1.reshape((b, n1, self.num_heads, head_dim)).transpose((0, 2, 1, 3))
        
        # Cross Attention 0 -> 1
        logits01 = jnp.matmul(qk0 * jnp.sqrt(scale), qk1.transpose((0, 1, 3, 2)) * jnp.sqrt(scale))
        if mask is not None:
            logits01 = jnp.where(mask, logits01, -1e9)
        attn01 = jax.nn.softmax(logits01, axis=-1)
        m0 = jnp.matmul(attn01, v1)
        
        # Cross Attention 1 -> 0
        logits10 = logits01.transpose((0, 1, 3, 2))
        if mask is not None:
            logits10 = jnp.where(mask.transpose((0, 1, 3, 2)), logits10, -1e9)
        attn10 = jax.nn.softmax(logits10, axis=-1)
        m1 = jnp.matmul(attn10, v0)
        
        to_out = nn.Dense(self.embed_dim, use_bias=self.bias, name="to_out")
        
        m0 = m0.transpose((0, 2, 1, 3)).reshape((b, n0, self.embed_dim))
        m1 = m1.transpose((0, 2, 1, 3)).reshape((b, n1, self.embed_dim))
        
        m0, m1 = to_out(m0), to_out(m1)
        
        ffn_0 = nn.Dense(2 * self.embed_dim, name="ffn_0")
        ffn_1 = nn.LayerNorm(epsilon=1e-5, name="ffn_1")
        ffn_3 = nn.Dense(self.embed_dim, name="ffn_3")

        def ffn(x, m):
            y = ffn_0(jnp.concatenate([x, m], axis=-1))
            y = ffn_1(y)
            y = jax.nn.gelu(y, approximate=False)
            y = ffn_3(y)
            return x + y

        x0 = ffn(x0, m0)
        x1 = ffn(x1, m1)
        
        return x0, x1

class TransformerLayer(nn.Module):
    embed_dim: int
    num_heads: int
    bias: bool = True

    @nn.compact
    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray, 
                 encoding0: jnp.ndarray, encoding1: jnp.ndarray,
                 mask0: Optional[jnp.ndarray] = None, mask1: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        cross_mask = None
        if mask0 is not None and mask1 is not None:
            cross_mask = mask0 & mask1.transpose((0, 1, 3, 2))
            self_mask0 = mask0 & mask0.transpose((0, 1, 3, 2))
            self_mask1 = mask1 & mask1.transpose((0, 1, 3, 2))
        else:
            self_mask0 = None
            self_mask1 = None

        self_attn = SelfBlock(self.embed_dim, self.num_heads, bias=self.bias, name="self_attn")
        desc0 = self_attn(desc0, encoding0, mask=self_mask0)
        desc1 = self_attn(desc1, encoding1, mask=self_mask1)
        
        return CrossBlock(self.embed_dim, self.num_heads, bias=self.bias, name="cross_attn")(desc0, desc1, mask=cross_mask)

def sigmoid_log_double_softmax(sim: jnp.ndarray, z0: jnp.ndarray, z1: jnp.ndarray) -> jnp.ndarray:
    b, m, n = sim.shape
    certainties = jax.nn.log_sigmoid(z0) + jax.nn.log_sigmoid(z1).transpose((0, 2, 1))
    
    scores0 = jax.nn.log_softmax(sim, axis=2)
    scores1 = jax.nn.log_softmax(sim.transpose((0, 2, 1)), axis=2).transpose((0, 2, 1))
    
    main_scores = scores0 + scores1 + certainties
    bin0 = jax.nn.log_sigmoid(-z0) # [B, M, 1]
    bin1 = jax.nn.log_sigmoid(-z1).transpose((0, 2, 1)) # [B, 1, N]
    
    scores = jnp.concatenate([main_scores, bin0], axis=2) # [B, M, N+1]
    last_row = jnp.concatenate([bin1, jnp.zeros((b, 1, 1))], axis=2) # [B, 1, N+1]
    scores = jnp.concatenate([scores, last_row], axis=1) # [B, M+1, N+1]
    
    return scores

class MatchAssignment(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        matchability = nn.Dense(1, name="matchability")
        final_proj = nn.Dense(self.dim, name="final_proj")
        
        mdesc0, mdesc1 = final_proj(desc0), final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        
        sim = jnp.matmul(mdesc0, mdesc1.transpose((0, 2, 1)))
        z0 = matchability(desc0)
        z1 = matchability(desc1)
        
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

def normalize_keypoints(kpts: jnp.ndarray, size: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if size is None:
        size = 1 + jnp.max(kpts, axis=-2) - jnp.min(kpts, axis=-2)
    
    shift = size / 2
    # size shape: [B, 2]
    scale = jnp.max(size, axis=-1, keepdims=True) / 2
    # scale shape: [B, 1]
    kpts = (kpts - shift[:, None, :]) / scale[:, None, :]
    return kpts

class LightGlue(nn.Module):
    input_dim: int = 256
    descriptor_dim: int = 256
    n_layers: int = 9
    num_heads: int = 4
    add_scale_ori: bool = False
    filter_threshold: float = 0.1

    @nn.compact
    def __call__(self, data: dict) -> dict:
        desc0 = data["image0"]["descriptors"]
        desc1 = data["image1"]["descriptors"]
        kpts0 = data["image0"]["keypoints"]
        kpts1 = data["image1"]["keypoints"]
        
        size0 = data["image0"].get("image_size")
        size1 = data["image1"].get("image_size")
        
        kpts0 = normalize_keypoints(kpts0, size0)
        kpts1 = normalize_keypoints(kpts1, size1)
        
        # Input Projection
        if self.input_dim != self.descriptor_dim:
            input_proj = nn.Dense(self.descriptor_dim, name="input_proj")
            desc0 = input_proj(desc0)
            desc1 = input_proj(desc1)
            
        # Positional Encoding
        head_dim = self.descriptor_dim // self.num_heads
        posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.add_scale_ori, head_dim, head_dim, name="posenc"
        )
        
        encoding0 = posenc(kpts0)
        encoding1 = posenc(kpts1)
        
        # Transformers
        for i in range(self.n_layers):
            layer = TransformerLayer(self.descriptor_dim, self.num_heads, name=f"transformers_{i}")
            desc0, desc1 = layer(desc0, desc1, encoding0, encoding1)
            
            if i < self.n_layers - 1:
                token_conf = TokenConfidence(self.descriptor_dim, name=f"token_confidence_{i}")
                conf0 = token_conf(desc0)
                conf1 = token_conf(desc1)

        # Final Match Assignment
        scores, sim = MatchAssignment(self.descriptor_dim, name=f"log_assignment_{self.n_layers-1}")(desc0, desc1)
        
        return {
            "scores": scores,
            "sim": sim,
        }
