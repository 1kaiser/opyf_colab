import os
import sys
import jax
import jax.numpy as jnp
import flax
import numpy as np
import time
from PIL import Image
from models.jax.jax_mast3r.models.mast3r import FlaxAsymmetricMASt3R

def main():
    # Set high precision for JAX
    jax.config.update("jax_default_matmul_precision", "highest")
    
    # Model parameters
    enc_depth, dec_depth = 24, 12
    enc_embed_dim, dec_embed_dim = 1024, 768
    enc_num_heads, dec_num_heads = 16, 12
    
    print("Initializing JAX model...")
    model = FlaxAsymmetricMASt3R(
        enc_depth=enc_depth, dec_depth=dec_depth,
        enc_embed_dim=enc_embed_dim, dec_embed_dim=dec_embed_dim,
        enc_num_heads=enc_num_heads, dec_num_heads=dec_num_heads,
        has_conf=True, two_confs=True
    )
    
    print("Loading JAX weights...")
    with open("mast3r_jax_weights.msgpack", "rb") as f:
        params = flax.serialization.from_bytes(None, f.read())
    
    # Load PyTorch preprocessed images
    print("Loading PyTorch preprocessed images...")
    img1 = jnp.array(np.load("output/pytorch_inference/img1.npy"))
    img2 = jnp.array(np.load("output/pytorch_inference/img2.npy"))
    
    print("Running JAX inference...")
    # JIT the call
    @jax.jit
    def apply_model(p, i1, i2):
        return model.apply({'params': p}, i1, i2)
    
    # Returns (res1, res2, (feat1, feat2, dec1_blocks, dec2_blocks))
    res1, res2, debug = apply_model(params, img1, img2)
    pts1, conf1, desc1, dconf1 = res1
    pts2, conf2, desc2, dconf2 = res2
    f1, f2, dec1_blocks, dec2_blocks = debug
    
    # Save final outputs
    output_dir = "output/jax_inference"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "img1.npy"), np.array(img1))
    np.save(os.path.join(output_dir, "img2.npy"), np.array(img2))
    np.save(os.path.join(output_dir, "feat1.npy"), np.array(f1))
    np.save(os.path.join(output_dir, "feat2.npy"), np.array(f2))
    
    # Save all intermediate decoder block outputs
    decoder_output_dir = os.path.join(output_dir, "decoder_blocks")
    os.makedirs(decoder_output_dir, exist_ok=True)
    for i, block_out in enumerate(dec1_blocks):
        np.save(os.path.join(decoder_output_dir, f"dec1_blk_{i}.npy"), np.array(block_out))
    for i, block_out in enumerate(dec2_blocks):
        np.save(os.path.join(decoder_output_dir, f"dec2_blk_{i}.npy"), np.array(block_out))
    
    # Concatenate pts and conf to match PyTorch comparison format (B, H, W, 4)
    pts1_full = np.concatenate([np.array(pts1), np.array(conf1)[..., None]], axis=-1)
    pts2_full = np.concatenate([np.array(pts2), np.array(conf2)[..., None]], axis=-1)
    
    np.save(os.path.join(output_dir, "pts1.npy"), pts1_full)
    np.save(os.path.join(output_dir, "pts2.npy"), pts2_full)
    np.save(os.path.join(output_dir, "desc1.npy"), np.array(desc1))
    np.save(os.path.join(output_dir, "desc2.npy"), np.array(desc2))
    
    print(f"JAX inference complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
