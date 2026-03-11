import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import freeze, unfreeze

def convert_pytorch_to_flax(pytorch_state_dict, flax_params_flat):
    """
    Map PyTorch state_dict keys to Flax parameter keys.
    """
    new_params_flat = {}
    
    # Encoder mapping
    new_params_flat[('patch_embed', 'proj', 'kernel')] = np.transpose(pytorch_state_dict['patch_embed.proj.weight'].numpy(), (2, 3, 1, 0))
    new_params_flat[('patch_embed', 'proj', 'bias')] = pytorch_state_dict['patch_embed.proj.bias'].numpy()

    num_enc_blocks = len([k for k in pytorch_state_dict.keys() if k.startswith('enc_blocks.') and k.endswith('.norm1.weight')])
    for i in range(num_enc_blocks):
        new_params_flat[(f'enc_blocks.{i}', 'norm1', 'scale')] = pytorch_state_dict[f'enc_blocks.{i}.norm1.weight'].numpy()
        new_params_flat[(f'enc_blocks.{i}', 'norm1', 'bias')] = pytorch_state_dict[f'enc_blocks.{i}.norm1.bias'].numpy()
        new_params_flat[(f'enc_blocks.{i}', 'norm2', 'scale')] = pytorch_state_dict[f'enc_blocks.{i}.norm2.weight'].numpy()
        new_params_flat[(f'enc_blocks.{i}', 'norm2', 'bias')] = pytorch_state_dict[f'enc_blocks.{i}.norm2.bias'].numpy()
        
        q_w, k_w, v_w = np.split(pytorch_state_dict[f'enc_blocks.{i}.attn.qkv.weight'].numpy(), 3, axis=0)
        q_b, k_b, v_b = np.split(pytorch_state_dict[f'enc_blocks.{i}.attn.qkv.bias'].numpy(), 3, axis=0)
        new_params_flat[(f'enc_blocks.{i}', 'attn', 'qkv', 'kernel')] = np.concatenate([q_w.T, k_w.T, v_w.T], axis=-1)
        new_params_flat[(f'enc_blocks.{i}', 'attn', 'qkv', 'bias')] = np.concatenate([q_b, k_b, v_b], axis=-1)
        new_params_flat[(f'enc_blocks.{i}', 'attn', 'proj', 'kernel')] = pytorch_state_dict[f'enc_blocks.{i}.attn.proj.weight'].T.numpy()
        new_params_flat[(f'enc_blocks.{i}', 'attn', 'proj', 'bias')] = pytorch_state_dict[f'enc_blocks.{i}.attn.proj.bias'].numpy()

        new_params_flat[(f'enc_blocks.{i}', 'mlp', 'fc1', 'kernel')] = pytorch_state_dict[f'enc_blocks.{i}.mlp.fc1.weight'].T.numpy()
        new_params_flat[(f'enc_blocks.{i}', 'mlp', 'fc1', 'bias')] = pytorch_state_dict[f'enc_blocks.{i}.mlp.fc1.bias'].numpy()
        new_params_flat[(f'enc_blocks.{i}', 'mlp', 'fc2', 'kernel')] = pytorch_state_dict[f'enc_blocks.{i}.mlp.fc2.weight'].T.numpy()
        new_params_flat[(f'enc_blocks.{i}', 'mlp', 'fc2', 'bias')] = pytorch_state_dict[f'enc_blocks.{i}.mlp.fc2.bias'].numpy()

    new_params_flat[('enc_norm', 'scale')] = pytorch_state_dict['enc_norm.weight'].numpy()
    new_params_flat[('enc_norm', 'bias')] = pytorch_state_dict['enc_norm.bias'].numpy()

    # Decoder Embed
    new_params_flat[('decoder_embed', 'kernel')] = pytorch_state_dict['decoder_embed.weight'].numpy().T
    new_params_flat[('decoder_embed', 'bias')] = pytorch_state_dict['decoder_embed.bias'].numpy()

    # Decoder Mapping
    num_dec_blocks = len([k for k in pytorch_state_dict.keys() if k.startswith('dec_blocks.') and k.endswith('.norm1.weight')])
    def map_decoder_block(side_name, pytorch_prefix):
        for i in range(num_dec_blocks):
            p_prefix = f'{pytorch_prefix}.{i}'
            f_prefix = f'{side_name}.{i}'
            new_params_flat[(f_prefix, 'norm1', 'scale')] = pytorch_state_dict[f'{p_prefix}.norm1.weight'].numpy()
            new_params_flat[(f_prefix, 'norm1', 'bias')] = pytorch_state_dict[f'{p_prefix}.norm1.bias'].numpy()
            new_params_flat[(f_prefix, 'norm2', 'scale')] = pytorch_state_dict[f'{p_prefix}.norm2.weight'].numpy()
            new_params_flat[(f_prefix, 'norm2', 'bias')] = pytorch_state_dict[f'{p_prefix}.norm2.bias'].numpy()
            new_params_flat[(f_prefix, 'norm3', 'scale')] = pytorch_state_dict[f'{p_prefix}.norm3.weight'].numpy()
            new_params_flat[(f_prefix, 'norm3', 'bias')] = pytorch_state_dict[f'{p_prefix}.norm3.bias'].numpy()
            new_params_flat[(f_prefix, 'norm_y', 'scale')] = pytorch_state_dict[f'{p_prefix}.norm_y.weight'].numpy()
            new_params_flat[(f_prefix, 'norm_y', 'bias')] = pytorch_state_dict[f'{p_prefix}.norm_y.bias'].numpy()
            new_params_flat[(f_prefix, 'attn', 'qkv', 'kernel')] = pytorch_state_dict[f'{p_prefix}.attn.qkv.weight'].T.numpy()
            new_params_flat[(f_prefix, 'attn', 'qkv', 'bias')] = pytorch_state_dict[f'{p_prefix}.attn.qkv.bias'].numpy()
            new_params_flat[(f_prefix, 'attn', 'proj', 'kernel')] = pytorch_state_dict[f'{p_prefix}.attn.proj.weight'].T.numpy()
            new_params_flat[(f_prefix, 'attn', 'proj', 'bias')] = pytorch_state_dict[f'{p_prefix}.attn.proj.bias'].numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projq', 'kernel')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projq.weight'].T.numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projq', 'bias')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projq.bias'].numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projk', 'kernel')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projk.weight'].T.numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projk', 'bias')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projk.bias'].numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projv', 'kernel')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projv.weight'].T.numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'projv', 'bias')] = pytorch_state_dict[f'{p_prefix}.cross_attn.projv.bias'].numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'proj', 'kernel')] = pytorch_state_dict[f'{p_prefix}.cross_attn.proj.weight'].T.numpy()
            new_params_flat[(f_prefix, 'cross_attn', 'proj', 'bias')] = pytorch_state_dict[f'{p_prefix}.cross_attn.proj.bias'].numpy()
            new_params_flat[(f_prefix, 'mlp', 'fc1', 'kernel')] = pytorch_state_dict[f'{p_prefix}.mlp.fc1.weight'].T.numpy()
            new_params_flat[(f_prefix, 'mlp', 'fc1', 'bias')] = pytorch_state_dict[f'{p_prefix}.mlp.fc1.bias'].numpy()
            new_params_flat[(f_prefix, 'mlp', 'fc2', 'kernel')] = pytorch_state_dict[f'{p_prefix}.mlp.fc2.weight'].T.numpy()
            new_params_flat[(f_prefix, 'mlp', 'fc2', 'bias')] = pytorch_state_dict[f'{p_prefix}.mlp.fc2.bias'].numpy()

    map_decoder_block('dec_blocks', 'dec_blocks')
    map_decoder_block('dec_blocks2', 'dec_blocks2')

    new_params_flat[('dec_norm', 'scale')] = pytorch_state_dict['dec_norm.weight'].numpy()
    new_params_flat[('dec_norm', 'bias')] = pytorch_state_dict['dec_norm.bias'].numpy()

    # Heads Mapping
    def map_head(side_name, pytorch_prefix):
        # DPT Head
        def map_dpt():
            # act_postprocess
            for i in range(4):
                p_k = f'{pytorch_prefix}.dpt.act_postprocess.{i}.0.weight'
                p_b = f'{pytorch_prefix}.dpt.act_postprocess.{i}.0.bias'
                new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_0', 'kernel')] = np.transpose(pytorch_state_dict[p_k].numpy(), (2, 3, 1, 0))
                new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_0', 'bias')] = pytorch_state_dict[p_b].numpy()
                
                if i in [0, 1]:
                    p_k = f'{pytorch_prefix}.dpt.act_postprocess.{i}.1.weight'
                    p_b = f'{pytorch_prefix}.dpt.act_postprocess.{i}.1.bias'
                    # ConvTranspose PyTorch (in, out, k, k) -> Flax (k, k, out, in)
                    # Works with transpose_kernel=True in Flax
                    new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_1', 'kernel')] = np.transpose(pytorch_state_dict[p_k].numpy(), (2, 3, 1, 0))
                    new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_1', 'bias')] = pytorch_state_dict[p_b].numpy()
                elif i == 3:
                    p_k = f'{pytorch_prefix}.dpt.act_postprocess.{i}.1.weight'
                    p_b = f'{pytorch_prefix}.dpt.act_postprocess.{i}.1.bias'
                    new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_1', 'kernel')] = np.transpose(pytorch_state_dict[p_k].numpy(), (2, 3, 1, 0))
                    new_params_flat[(side_name, 'dpt', f'act_postprocess_{i}_1', 'bias')] = pytorch_state_dict[p_b].numpy()

            # layer_rn
            for i in range(4):
                p_k = f'{pytorch_prefix}.dpt.scratch.layer_rn.{i}.weight'
                new_params_flat[(side_name, 'dpt', f'scratch_layer_rn_{i}', 'kernel')] = np.transpose(pytorch_state_dict[p_k].numpy(), (2, 3, 1, 0))

            # refinenet
            for i in [1, 2, 3, 4]:
                p_rn = f'{pytorch_prefix}.dpt.scratch.refinenet{i}'
                f_rn = f'scratch_refinenet{i}'
                new_params_flat[(side_name, 'dpt', f_rn, 'out_conv', 'kernel')] = np.transpose(pytorch_state_dict[f'{p_rn}.out_conv.weight'].numpy(), (2, 3, 1, 0))
                new_params_flat[(side_name, 'dpt', f_rn, 'out_conv', 'bias')] = pytorch_state_dict[f'{p_rn}.out_conv.bias'].numpy()
                for j in [1, 2]:
                    p_rcu = f'{p_rn}.resConfUnit{j}'
                    f_rcu = f'resConfUnit{j}'
                    new_params_flat[(side_name, 'dpt', f_rn, f_rcu, 'conv1', 'kernel')] = np.transpose(pytorch_state_dict[f'{p_rcu}.conv1.weight'].numpy(), (2, 3, 1, 0))
                    new_params_flat[(side_name, 'dpt', f_rn, f_rcu, 'conv1', 'bias')] = pytorch_state_dict[f'{p_rcu}.conv1.bias'].numpy()
                    new_params_flat[(side_name, 'dpt', f_rn, f_rcu, 'conv2', 'kernel')] = np.transpose(pytorch_state_dict[f'{p_rcu}.conv2.weight'].numpy(), (2, 3, 1, 0))
                    new_params_flat[(side_name, 'dpt', f_rn, f_rcu, 'conv2', 'bias')] = pytorch_state_dict[f'{p_rcu}.conv2.bias'].numpy()

            # head
            for i in [0, 2, 4]:
                p_k = f'{pytorch_prefix}.dpt.head.{i}.weight'
                p_b = f'{pytorch_prefix}.dpt.head.{i}.bias'
                new_params_flat[(side_name, 'dpt', f'head_{i}', 'kernel')] = np.transpose(pytorch_state_dict[p_k].numpy(), (2, 3, 1, 0))
                new_params_flat[(side_name, 'dpt', f'head_{i}', 'bias')] = pytorch_state_dict[p_b].numpy()

        map_dpt()

        # Local Feature Head
        p_lf = f'{pytorch_prefix}.head_local_features'
        f_lf = 'head_local_features'
        new_params_flat[(side_name, f_lf, 'fc1', 'kernel')] = pytorch_state_dict[f'{p_lf}.fc1.weight'].numpy().T
        new_params_flat[(side_name, f_lf, 'fc1', 'bias')] = pytorch_state_dict[f'{p_lf}.fc1.bias'].numpy()
        new_params_flat[(side_name, f_lf, 'fc2', 'kernel')] = pytorch_state_dict[f'{p_lf}.fc2.weight'].numpy().T
        new_params_flat[(side_name, f_lf, 'fc2', 'bias')] = pytorch_state_dict[f'{p_lf}.fc2.bias'].numpy()

    map_head('downstream_head1', 'downstream_head1')
    map_head('downstream_head2', 'downstream_head2')

    return unflatten_dict(new_params_flat)
