import torch
import numpy as np
from flax import serialization
import os
import gc

def convert_vggt_weights(pt_path, output_path):
    print("Loading PyTorch weights...")
    state_dict = torch.load(pt_path, map_location="cpu")
    params = {}

    def set_param(path, value):
        curr = params
        for part in path[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[path[-1]] = np.array(value)

    def pop_torch(key):
        val = state_dict.pop(key)
        return val.numpy() if isinstance(val, torch.Tensor) else val

    def convert_block(p_b, j_b):
        set_param(j_b + ['norm1', 'scale'], pop_torch(f'{p_b}.norm1.weight'))
        set_param(j_b + ['norm1', 'bias'], pop_torch(f'{p_b}.norm1.bias'))
        
        set_param(j_b + ['attn', 'qkv', 'kernel'], pop_torch(f'{p_b}.attn.qkv.weight').T)
        set_param(j_b + ['attn', 'qkv', 'bias'], pop_torch(f'{p_b}.attn.qkv.bias'))
        set_param(j_b + ['attn', 'proj', 'kernel'], pop_torch(f'{p_b}.attn.proj.weight').T)
        set_param(j_b + ['attn', 'proj', 'bias'], pop_torch(f'{p_b}.attn.proj.bias'))
        
        q_norm_key = f'{p_b}.attn.q_norm.weight'
        if q_norm_key in state_dict:
            set_param(j_b + ['attn', 'q_norm', 'scale'], pop_torch(f'{p_b}.attn.q_norm.weight'))
            set_param(j_b + ['attn', 'q_norm', 'bias'], pop_torch(f'{p_b}.attn.q_norm.bias'))
            set_param(j_b + ['attn', 'k_norm', 'scale'], pop_torch(f'{p_b}.attn.k_norm.weight'))
            set_param(j_b + ['attn', 'k_norm', 'bias'], pop_torch(f'{p_b}.attn.k_norm.bias'))

        ls1_key = f'{p_b}.ls1.gamma'
        if ls1_key in state_dict:
            set_param(j_b + ['ls1', 'gamma'], pop_torch(ls1_key))
        
        set_param(j_b + ['norm2', 'scale'], pop_torch(f'{p_b}.norm2.weight'))
        set_param(j_b + ['norm2', 'bias'], pop_torch(f'{p_b}.norm2.bias'))
        
        set_param(j_b + ['mlp', 'fc1', 'kernel'], pop_torch(f'{p_b}.mlp.fc1.weight').T)
        set_param(j_b + ['mlp', 'fc1', 'bias'], pop_torch(f'{p_b}.mlp.fc1.bias'))
        set_param(j_b + ['mlp', 'fc2', 'kernel'], pop_torch(f'{p_b}.mlp.fc2.weight').T)
        set_param(j_b + ['mlp', 'fc2', 'bias'], pop_torch(f'{p_b}.mlp.fc2.bias'))
        
        ls2_key = f'{p_b}.ls2.gamma'
        if ls2_key in state_dict:
            set_param(j_b + ['ls2', 'gamma'], pop_torch(ls2_key))

    def convert_vit(prefix, j_prefix):
        set_param(j_prefix + ['cls_token'], pop_torch(f'{prefix}.cls_token'))
        set_param(j_prefix + ['pos_embed'], pop_torch(f'{prefix}.pos_embed'))
        if f'{prefix}.register_tokens' in state_dict:
            set_param(j_prefix + ['register_tokens'], pop_torch(f'{prefix}.register_tokens'))
        set_param(j_prefix + ['patch_embed', 'proj', 'kernel'], pop_torch(f'{prefix}.patch_embed.proj.weight').transpose(2, 3, 1, 0))
        set_param(j_prefix + ['patch_embed', 'proj', 'bias'], pop_torch(f'{prefix}.patch_embed.proj.bias'))
        i = 0
        while f'{prefix}.blocks.{i}.norm1.weight' in state_dict:
            convert_block(f'{prefix}.blocks.{i}', j_prefix + [f'blocks_{i}'])
            i += 1
            if i % 5 == 0: gc.collect()
        set_param(j_prefix + ['norm', 'scale'], pop_torch(f'{prefix}.norm.weight'))
        set_param(j_prefix + ['norm', 'bias'], pop_torch(f'{prefix}.norm.bias'))

    print("Converting Aggregator...")
    set_param(['aggregator', 'camera_token'], pop_torch('aggregator.camera_token'))
    set_param(['aggregator', 'register_token'], pop_torch('aggregator.register_token'))
    convert_vit('aggregator.patch_embed', ['aggregator', 'patch_embed'])
    for i in range(24):
        convert_block(f'aggregator.frame_blocks.{i}', ['aggregator', f'frame_blocks_{i}'])
        convert_block(f'aggregator.global_blocks.{i}', ['aggregator', f'global_blocks_{i}'])
        if i % 4 == 0: gc.collect()

    print("Converting Camera Head...")
    prefix = 'camera_head'
    j_prefix = ['camera_head']
    set_param(j_prefix + ['empty_pose_tokens'], pop_torch(f'{prefix}.empty_pose_tokens'))
    set_param(j_prefix + ['embed_pose', 'kernel'], pop_torch(f'{prefix}.embed_pose.weight').T)
    set_param(j_prefix + ['embed_pose', 'bias'], pop_torch(f'{prefix}.embed_pose.bias'))
    
    # poseLN_modulation explicitly
    set_param(j_prefix + ['poseLN_modulation_dense', 'kernel'], pop_torch(f'{prefix}.poseLN_modulation.1.weight').T)
    set_param(j_prefix + ['poseLN_modulation_dense', 'bias'], pop_torch(f'{prefix}.poseLN_modulation.1.bias'))
    
    if f'{prefix}.adaln_norm.weight' in state_dict:
        set_param(j_prefix + ['adaln_norm', 'scale'], pop_torch(f'{prefix}.adaln_norm.weight'))
        set_param(j_prefix + ['adaln_norm', 'bias'], pop_torch(f'{prefix}.adaln_norm.bias'))
    for i in range(4):
        convert_block(f'{prefix}.trunk.{i}', j_prefix + [f'trunk_{i}'])
    set_param(j_prefix + ['token_norm', 'scale'], pop_torch(f'{prefix}.token_norm.weight'))
    set_param(j_prefix + ['token_norm', 'bias'], pop_torch(f'{prefix}.token_norm.bias'))
    set_param(j_prefix + ['trunk_norm', 'scale'], pop_torch(f'{prefix}.trunk_norm.weight'))
    set_param(j_prefix + ['trunk_norm', 'bias'], pop_torch(f'{prefix}.trunk_norm.bias'))
    set_param(j_prefix + ['pose_branch', 'fc1', 'kernel'], pop_torch(f'{prefix}.pose_branch.fc1.weight').T)
    set_param(j_prefix + ['pose_branch', 'fc1', 'bias'], pop_torch(f'{prefix}.pose_branch.fc1.bias'))
    set_param(j_prefix + ['pose_branch', 'fc2', 'kernel'], pop_torch(f'{prefix}.pose_branch.fc2.weight').T)
    set_param(j_prefix + ['pose_branch', 'fc2', 'bias'], pop_torch(f'{prefix}.pose_branch.fc2.bias'))

    print("Converting DPT Heads...")
    def convert_dpt(prefix, j_prefix):
        set_param(j_prefix + ['norm', 'scale'], pop_torch(f'{prefix}.norm.weight'))
        set_param(j_prefix + ['norm', 'bias'], pop_torch(f'{prefix}.norm.bias'))
        for i in range(4):
            set_param(j_prefix + [f'projects_{i}', 'kernel'], pop_torch(f'{prefix}.projects.{i}.weight').transpose(2, 3, 1, 0))
            set_param(j_prefix + [f'projects_{i}', 'bias'], pop_torch(f'{prefix}.projects.{i}.bias'))
        for i in range(4):
            p_key = f'{prefix}.resize_layers.{i}'
            j_key = f'resize_layers_{i}'
            if f'{p_key}.weight' in state_dict:
                w = pop_torch(f'{p_key}.weight')
                set_param(j_prefix + [j_key, 'kernel'], w.transpose(2, 3, 1, 0))
                if f'{p_key}.bias' in state_dict:
                    set_param(j_prefix + [j_key, 'bias'], pop_torch(f'{p_key}.bias'))
        s_p = f'{prefix}.scratch'
        s_j = j_prefix
        for i in range(1, 5):
            set_param(s_j + [f'scratch_layer{i}_rn', 'kernel'], pop_torch(f'{s_p}.layer{i}_rn.weight').transpose(2, 3, 1, 0))
        for i in range(1, 5):
            p_r = f'{s_p}.refinenet{i}'
            j_r = s_j + [f'scratch_refinenet{i}']
            set_param(j_r + ['out_conv', 'kernel'], pop_torch(f'{p_r}.out_conv.weight').transpose(2, 3, 1, 0))
            set_param(j_r + ['out_conv', 'bias'], pop_torch(f'{p_r}.out_conv.bias'))
            for u in range(1, 3):
                p_u = f'{p_r}.resConfUnit{u}'
                j_u = j_r + [f'resConfUnit{u}']
                if f'{p_u}.conv1.weight' in state_dict:
                    set_param(j_u + ['conv1', 'kernel'], pop_torch(f'{p_u}.conv1.weight').transpose(2, 3, 1, 0))
                    set_param(j_u + ['conv1', 'bias'], pop_torch(f'{p_u}.conv1.bias'))
                    set_param(j_u + ['conv2', 'kernel'], pop_torch(f'{p_u}.conv2.weight').transpose(2, 3, 1, 0))
                    set_param(j_u + ['conv2', 'bias'], pop_torch(f'{p_u}.conv2.bias'))
        set_param(j_prefix + ['scratch_output_conv1', 'kernel'], pop_torch(f'{s_p}.output_conv1.weight').transpose(2, 3, 1, 0))
        set_param(j_prefix + ['scratch_output_conv1', 'bias'], pop_torch(f'{s_p}.output_conv1.bias'))
        if f'{s_p}.output_conv2.0.weight' in state_dict:
            set_param(j_prefix + ['scratch_output_conv2_0', 'kernel'], pop_torch(f'{s_p}.output_conv2.0.weight').transpose(2, 3, 1, 0))
            set_param(j_prefix + ['scratch_output_conv2_0', 'bias'], pop_torch(f'{s_p}.output_conv2.0.bias'))
            set_param(j_prefix + ['scratch_output_conv2_2', 'kernel'], pop_torch(f'{s_p}.output_conv2.2.weight').transpose(2, 3, 1, 0))
            set_param(j_prefix + ['scratch_output_conv2_2', 'bias'], pop_torch(f'{s_p}.output_conv2.2.bias'))

    convert_dpt('point_head', ['point_head'])
    convert_dpt('depth_head', ['depth_head'])

    print("Saving converted weights...")
    with open(output_path, "wb") as f:
        f.write(serialization.to_bytes({'params': params}))
    print(f"Converted VGGT weights saved to {output_path}")

if __name__ == "__main__":
    pt_path = "weights/vggt/vggt_1b.pt"
    output_path = "weights/vggt/vggt_1b.msgpack"
    convert_vggt_weights(pt_path, output_path)
