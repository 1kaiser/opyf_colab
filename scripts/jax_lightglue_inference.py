import jax
import jax.numpy as jnp
from models.jax.jax_lightglue.models.lightglue import LightGlue
from flax import serialization
import numpy as np
import os

jax.config.update("jax_default_matmul_precision", "highest")

def run_jax_inference():
    # Load inputs
    kpts0 = np.load("output/lightglue_parity/kpts0.npy")
    desc0 = np.load("output/lightglue_parity/desc0.npy")
    kpts1 = np.load("output/lightglue_parity/kpts1.npy")
    desc1 = np.load("output/lightglue_parity/desc1.npy")
    
    # Prepare data dict
    data = {
        "image0": {
            "keypoints": jnp.array(kpts0),
            "descriptors": jnp.array(desc0)
        },
        "image1": {
            "keypoints": jnp.array(kpts1),
            "descriptors": jnp.array(desc1)
        }
    }
    
    # Initialize model
    model = LightGlue(n_layers=9)
    
    # Load weights
    with open("weights/superpoint_lightglue.msgpack", "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    
    # Run inference
    # Manual forward
    kpts0_in = data["image0"]["keypoints"]
    kpts1_in = data["image1"]["keypoints"]
    desc0 = data["image0"]["descriptors"]
    desc1 = data["image1"]["descriptors"]
    
    from models.jax.jax_lightglue.models.lightglue import normalize_keypoints, LearnableFourierPositionalEncoding, TransformerLayer, MatchAssignment
    
    kpts0 = normalize_keypoints(kpts0_in, None)
    kpts1 = normalize_keypoints(kpts1_in, None)
    
    head_dim = model.descriptor_dim // model.num_heads
    posenc = LearnableFourierPositionalEncoding(
        2 + 2 * model.add_scale_ori, head_dim, head_dim
    )
    encoding0 = posenc.apply({'params': variables['params']['posenc']}, kpts0)
    encoding1 = posenc.apply({'params': variables['params']['posenc']}, kpts1)
    
    np.save("output/lightglue_parity/jax_enc0.npy", encoding0)
    np.save("output/lightglue_parity/jax_enc1.npy", encoding1)
    
    for i in range(model.n_layers):
        layer = TransformerLayer(model.descriptor_dim, model.num_heads)
        
        # Manually run self_attn for granular capture
        params_layer = variables['params'][f'transformers_{i}']
        params_self = params_layer['self_attn']
        
        from models.jax.jax_lightglue.models.lightglue import apply_rotary_emb, Attention
        
        # SelfBlock logic
        b, n, d = desc0.shape
        head_dim = model.descriptor_dim // model.num_heads
        
        # Wqkv
        wqkv_w = params_self['Wqkv']['kernel']
        wqkv_b = params_self['Wqkv']['bias']
        qkv = jnp.dot(desc0, wqkv_w) + wqkv_b
        qkv = qkv.reshape((b, n, model.num_heads, head_dim, 3)).transpose((0, 2, 1, 3, 4))
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_rotary_emb(encoding0, q)
        k = apply_rotary_emb(encoding0, k)
        
        np.save(f"output/lightglue_parity/jax_layer_{i}_self_q.npy", q)
        
        context = Attention()(q, k, v)
        message = context.transpose((0, 2, 1, 3)).reshape((b, n, model.descriptor_dim))
        message = jnp.dot(message, params_self['out_proj']['kernel']) + params_self['out_proj']['bias']
        
        # FFN Granular
        ffn_in = jnp.concatenate([desc0, message], axis=-1)
        ffn_0_out = jnp.dot(ffn_in, params_self['ffn_0']['kernel']) + params_self['ffn_0']['bias']
        
        # LayerNorm
        mean = jnp.mean(ffn_0_out, axis=-1, keepdims=True)
        var = jnp.var(ffn_0_out, axis=-1, keepdims=True)
        ffn_1_out = (ffn_0_out - mean) / jnp.sqrt(var + 1e-5)
        ffn_1_out = ffn_1_out * params_self['ffn_1']['scale'] + params_self['ffn_1']['bias']
        
        ffn_2_out = jax.nn.gelu(ffn_1_out, approximate=False)
        ffn_3_out = jnp.dot(ffn_2_out, params_self['ffn_3']['kernel']) + params_self['ffn_3']['bias']
        
        np.save(f"output/lightglue_parity/jax_layer_{i}_ffn_0.npy", ffn_0_out)
        np.save(f"output/lightglue_parity/jax_layer_{i}_ffn_1.npy", ffn_1_out)
        np.save(f"output/lightglue_parity/jax_layer_{i}_ffn_2.npy", ffn_2_out)

        desc0_self = desc0 + ffn_3_out
        np.save(f"output/lightglue_parity/jax_layer_{i}_self_out0.npy", desc0_self)

        desc0, desc1 = layer.apply({'params': params_layer}, desc0, desc1, encoding0, encoding1)
        np.save(f"output/lightglue_parity/jax_desc0_layer_{i}.npy", desc0)
        np.save(f"output/lightglue_parity/jax_desc1_layer_{i}.npy", desc1)

    assignment = MatchAssignment(model.descriptor_dim)
    scores, sim = assignment.apply({'params': variables['params'][f'log_assignment_{model.n_layers-1}']}, desc0, desc1)
    
    # Save outputs
    np.save("output/lightglue_parity/jax_log_scores.npy", scores)
    np.save("output/lightglue_parity/jax_sim.npy", sim)
    
    print("JAX inference completed.")

if __name__ == "__main__":
    run_jax_inference()
