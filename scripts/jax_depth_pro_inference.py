import jax
import jax.numpy as jnp
from models.jax.jax_depth_pro.models.depth_pro import DepthPro
from flax import serialization
import numpy as np
import os

jax.config.update("jax_default_matmul_precision", "highest")

def run_jax_depth_pro():
    # Load input image
    input_image = np.load("output/depth_pro_parity/input_image.npy")
    
    # Vit Config
    vit_config = {
        'img_size': 384,
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'init_values': 1e-5
    }
    
    # Initialize model
    model = DepthPro(vit_config=vit_config)
    
    # Load weights
    with open("weights/depth_pro.msgpack", "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    
    # Run inference
    # input_image is [1, 3, 1536, 1536]
    # We need to transpose to [1, 1536, 1536, 3] for JAX if we use NHWC Convs?
    # Wait, in my JAX code I used NHWC for patch_embed and others?
    # Let's check PatchEmbed:
    # x = nn.Conv(...)(x.transpose(0, 2, 3, 1))
    # So it expects NCHW and transposes it!
    
    # Actually, let's double check Encoder.
    # jax.image.resize(x, ...) handles channels if they are at the end? 
    # Usually it's (H, W, C).
    
    # I'll modify my models to be consistent. 
    # Usually JAX prefers NHWC.
    
    # Let's check my ViT implementation again.
    # PatchEmbed:
    # x = nn.Conv(...)(x.transpose(0, 2, 3, 1))
    # This means input x is NCHW.
    
    out_depth, out_fov, encodings, lowres_features, fov_debug = model.apply(variables, jnp.array(input_image))
    x_down, x_feat, h_low = fov_debug
    
    # Save outputs
    np.save("output/depth_pro_parity/jax_canonical_inv_depth.npy", out_depth)
    np.save("output/depth_pro_parity/jax_fov_deg.npy", out_fov)
    np.save("output/depth_pro_parity/jax_lowres_features.npy", lowres_features)
    np.save("output/depth_pro_parity/jax_x_down.npy", x_down)
    np.save("output/depth_pro_parity/jax_fov_feat.npy", x_feat)
    np.save("output/depth_pro_parity/jax_fov_low.npy", h_low)
    for i, enc in enumerate(encodings):
        np.save(f"output/depth_pro_parity/jax_enc_{i}.npy", enc)
    
    print("JAX Depth Pro inference completed.")

if __name__ == "__main__":
    run_jax_depth_pro()
