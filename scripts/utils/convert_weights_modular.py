
import os
import sys
import torch
import flax
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from flax.core import unfreeze

# Add mast3r_repo to sys.path for loading pytorch model
sys.path.append(os.path.abspath("mast3r_repo"))

from mast3r.model import AsymmetricMASt3R
from models.jax.jax_mast3r.models.mast3r import FlaxAsymmetricMASt3R
from models.jax.jax_mast3r.utils.weights import convert_pytorch_to_flax

def main():
    model_path = "mast3r_repo/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    print(f"Loading PyTorch model from {model_path}...")
    pytorch_model = AsymmetricMASt3R.from_pretrained(model_path).to('cpu')
    pytorch_model.eval()
    pytorch_state_dict = pytorch_model.state_dict()

    print("Initializing Flax model...")
    flax_model = FlaxAsymmetricMASt3R(
        enc_depth=pytorch_model.enc_depth,
        dec_depth=pytorch_model.dec_depth,
        enc_embed_dim=pytorch_model.enc_embed_dim,
        dec_embed_dim=pytorch_model.dec_embed_dim,
        enc_num_heads=16, # Known from previous logs
        dec_num_heads=12, # Known from previous logs
        patch_size=16
    )
    
    dummy_img = jnp.ones((1, 3, 512, 512))
    variables = flax_model.init(jax.random.PRNGKey(0), dummy_img, dummy_img)
    flax_params_flat = flatten_dict(unfreeze(variables['params']))

    print("Converting weights...")
    new_params = convert_pytorch_to_flax(pytorch_state_dict, flax_params_flat)

    output_path = "mast3r_jax_weights.msgpack"
    with open(output_path, "wb") as f:
        f.write(flax.serialization.to_bytes(new_params))
    
    print(f"Weight conversion complete! Saved to {output_path}")

if __name__ == "__main__":
    main()
