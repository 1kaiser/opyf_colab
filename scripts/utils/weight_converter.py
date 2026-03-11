
import torch
import numpy as np
import flax
import jax
import jax.numpy as jnp
from mast3r.model import AsymmetricMASt3R
from mast3r_flax import FlaxAsymmetricMASt3R
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import freeze, unfreeze

def print_params(title, params):
    print(f"--- {title} ---")
    for k, v in params.items():
        print(f"{k}: {v.shape}")

def convert_weights(pytorch_model, flax_model):
    """
    Convert PyTorch weights to Flax.
    This is a complex process and will be built out iteratively.
    """
    
    # Initialize flax model and get its flat param structure
    dummy_img = jnp.ones((1, 3, 512, 512))
    # Dummy inputs for init, feat1 and query will be used for shape inference
    flax_params = flax_model.init({'params': jax.random.PRNGKey(0)}, dummy_img, None)['params']
    flax_params_flat = flatten_dict(unfreeze(flax_params))

    pytorch_params = pytorch_model.state_dict()
    
    # This dictionary will hold the converted weights
    new_flax_params_flat = {}

    print("Starting weight conversion...")
    
    # --- Encoder (backbone) Conversion ---
    # Patch embedding
    new_flax_params_flat[('encoder', 'patch_embed', 'proj', 'kernel')] = np.transpose(pytorch_params['patch_embed.proj.weight'].numpy(), (2, 3, 1, 0))
    new_flax_params_flat[('encoder', 'patch_embed', 'proj', 'bias')] = pytorch_params['patch_embed.proj.bias'].numpy()

    # Encoder Blocks
    for i in range(pytorch_model.enc_depth):
        # Layer Norm 1
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'norm1', 'scale')] = pytorch_params[f'enc_blocks.{i}.norm1.weight'].numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'norm1', 'bias')] = pytorch_params[f'enc_blocks.{i}.norm1.bias'].numpy()
        
        # Attention - QKV
        q_w, k_w, v_w = np.split(pytorch_params[f'enc_blocks.{i}.attn.qkv.weight'].numpy(), 3, axis=0)
        q_b, k_b, v_b = np.split(pytorch_params[f'enc_blocks.{i}.attn.qkv.bias'].numpy(), 3, axis=0)
        qkv_kernel = np.concatenate([q_w.T, k_w.T, v_w.T], axis=-1)
        qkv_bias = np.concatenate([q_b, k_b, v_b], axis=-1)
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'attn', 'qkv', 'kernel')] = qkv_kernel
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'attn', 'qkv', 'bias')] = qkv_bias

        # Attention - Projection
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'attn', 'proj', 'kernel')] = pytorch_params[f'enc_blocks.{i}.attn.proj.weight'].T.numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'attn', 'proj', 'bias')] = pytorch_params[f'enc_blocks.{i}.attn.proj.bias'].numpy()

        # Layer Norm 2
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'norm2', 'scale')] = pytorch_params[f'enc_blocks.{i}.norm2.weight'].numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'norm2', 'bias')] = pytorch_params[f'enc_blocks.{i}.norm2.bias'].numpy()

        # MLP
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'fc1', 'kernel')] = pytorch_params[f'enc_blocks.{i}.mlp.fc1.weight'].T.numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'fc1', 'bias')] = pytorch_params[f'enc_blocks.{i}.mlp.fc1.bias'].numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'fc2', 'kernel')] = pytorch_params[f'enc_blocks.{i}.mlp.fc2.weight'].T.numpy()
        new_flax_params_flat[(f'encoder', f'blocks.{i}', 'fc2', 'bias')] = pytorch_params[f'enc_blocks.{i}.mlp.fc2.bias'].numpy()

    # Final Encoder Layer Norm
    new_flax_params_flat[('encoder', 'norm', 'scale')] = pytorch_params['enc_norm.weight'].numpy()
    new_flax_params_flat[('encoder', 'norm', 'bias')] = pytorch_params['enc_norm.bias'].numpy()
    
    print(f"Matched {len([k for k in new_flax_params_flat if 'encoder' in k[0]])} parameters for the encoder.")

    # --- Decoder Conversion ---
    # Query token
    new_flax_params_flat[('params', 'query_token')] = pytorch_params['decoder_embed.weight'].numpy().T # Transpose for Flax
    
    for i in range(pytorch_model.dec_depth):
        # Self-Attention
        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'norm1', 'scale')] = pytorch_params[f'dec_blocks.{i}.norm1.weight'].numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'norm1', 'bias')] = pytorch_params[f'dec_blocks.{i}.norm1.bias'].numpy()
        
        qkv_w, qkv_b = pytorch_params[f'dec_blocks.{i}.attn.qkv.weight'].numpy(), pytorch_params[f'dec_blocks.{i}.attn.qkv.bias'].numpy()
        qkv_kernel = qkv_w.T
        qkv_bias = qkv_b
        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'attn', 'qkv', 'kernel')] = qkv_kernel
        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'attn', 'qkv', 'bias')] = qkv_bias

        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'attn', 'proj', 'kernel')] = pytorch_params[f'dec_blocks.{i}.attn.proj.weight'].T.numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'attn', 'attn', 'proj', 'bias')] = pytorch_params[f'dec_blocks.{i}.attn.proj.bias'].numpy()
        
        # Cross-Attention
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'norm2', 'scale')] = pytorch_params[f'dec_blocks.{i}.norm2.weight'].numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'norm2', 'bias')] = pytorch_params[f'dec_blocks.{i}.norm2.bias'].numpy()
        
        # PyTorch uses separate proj_q, proj_k, proj_v. Flax CrossAttention uses projq and projkv
        # We need to adapt the mapping here.
        # Query from x
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'projq', 'kernel')] = pytorch_params[f'dec_blocks.{i}.cross_attn.projq.weight'].T.numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'projq', 'bias')] = pytorch_params[f'dec_blocks.{i}.cross_attn.projq.bias'].numpy()
        
        # Key and Value from context (concatenated)
        k_w_ca, v_w_ca = pytorch_params[f'dec_blocks.{i}.cross_attn.projk.weight'].numpy(), pytorch_params[f'dec_blocks.{i}.cross_attn.projv.weight'].numpy()
        k_b_ca, v_b_ca = pytorch_params[f'dec_blocks.{i}.cross_attn.projk.bias'].numpy(), pytorch_params[f'dec_blocks.{i}.cross_attn.projv.bias'].numpy()
        projkv_kernel = np.concatenate([k_w_ca.T, v_w_ca.T], axis=-1)
        projkv_bias = np.concatenate([k_b_ca, v_b_ca], axis=-1)
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'projkv', 'kernel')] = projkv_kernel
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'projkv', 'bias')] = projkv_bias

        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'proj', 'kernel')] = pytorch_params[f'dec_blocks.{i}.cross_attn.proj.weight'].T.numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'cross_attn', 'cross_attn', 'proj', 'bias')] = pytorch_params[f'dec_blocks.{i}.cross_attn.proj.bias'].numpy()

        # MLP
        new_flax_params_flat[(f'dec_blocks.{i}', 'fc1', 'norm3', 'scale')] = pytorch_params[f'dec_blocks.{i}.norm3.weight'].numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'fc1', 'norm3', 'bias')] = pytorch_params[f'dec_blocks.{i}.norm3.bias'].numpy()

        new_flax_params_flat[(f'dec_blocks.{i}', 'fc1', 'kernel')] = pytorch_params[f'dec_blocks.{i}.mlp.fc1.weight'].T.numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'fc1', 'bias')] = pytorch_params[f'dec_blocks.{i}.mlp.fc1.bias'].numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'fc2', 'kernel')] = pytorch_params[f'dec_blocks.{i}.mlp.fc2.weight'].T.numpy()
        new_flax_params_flat[(f'dec_blocks.{i}', 'fc2', 'bias')] = pytorch_params[f'dec_blocks.{i}.mlp.fc2.bias'].numpy()

    # Final Decoder Layer Norm
    new_flax_params_flat[('decoder_norm', 'scale')] = pytorch_params['dec_norm.weight'].numpy()
    new_flax_params_flat[('decoder_norm', 'bias')] = pytorch_params['dec_norm.bias'].numpy()

    print(f"Matched {len([k for k in new_flax_params_flat if 'dec' in k[0]])} parameters for the decoder.")
    
    # Sanity check: ensure all converted keys exist in the Flax model
    for k_flax_path in new_flax_params_flat.keys():
        if k_flax_path not in flax_params_flat:
            print(f"Warning: Converted key '{k_flax_path}' not found in Flax model parameters.")

    # Convert to a frozen dict and return
    return freeze(unflatten_dict(new_flax_params_flat))


def main():
    # Load PyTorch model
    device = 'cpu'
    model_path = "mast3r_repo/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    pytorch_model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    pytorch_model.eval()

    # Initialize Flax model
    flax_model = FlaxAsymmetricMASt3R(
        enc_depth=pytorch_model.enc_depth,
        dec_depth=pytorch_model.dec_depth,
        enc_embed_dim=pytorch_model.enc_embed_dim,
        dec_embed_dim=pytorch_model.dec_embed_dim,
        enc_num_heads=16, # Hardcoded
        dec_num_heads=12, # Hardcoded
        patch_size=16 # Hardcoded
    )

    # Convert weights
    flax_weights = convert_weights(pytorch_model, flax_model)

    # Save the converted weights
    with open("mast3r_flax.msgpack", "wb") as f:
        f.write(flax.serialization.to_bytes(flax_weights))

    print("Flax weights saved to mast3r_flax.msgpack")

if __name__ == '__main__':
    main()
