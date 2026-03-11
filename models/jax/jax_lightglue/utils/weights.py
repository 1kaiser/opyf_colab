import torch
import numpy as np
from flax import serialization
import os

def convert_lightglue_weights(pt_path, output_path, n_layers=9):
    state_dict = torch.load(pt_path, map_location="cpu")
    
    # Rename keys if they are from the old format
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for i in range(n_layers):
            new_k = new_k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn")
            new_k = new_k.replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn")
        new_state_dict[new_k] = v
    state_dict = new_state_dict

    params = {}

    def set_param(path, value):
        curr = params
        for part in path[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[path[-1]] = np.array(value)

    # Positional Encoding
    set_param(['posenc', 'Wr', 'kernel'], state_dict['posenc.Wr.weight'].T)

    # Input Projection
    if 'input_proj.weight' in state_dict:
        set_param(['input_proj', 'kernel'], state_dict['input_proj.weight'].T)
        set_param(['input_proj', 'bias'], state_dict['input_proj.bias'])

    # Transformers
    for i in range(n_layers):
        # Self Attention
        p_prefix = f'transformers.{i}.self_attn'
        j_prefix = f'transformers_{i}'
        
        set_param([j_prefix, 'self_attn', 'Wqkv', 'kernel'], state_dict[f'{p_prefix}.Wqkv.weight'].T)
        set_param([j_prefix, 'self_attn', 'Wqkv', 'bias'], state_dict[f'{p_prefix}.Wqkv.bias'])
        set_param([j_prefix, 'self_attn', 'out_proj', 'kernel'], state_dict[f'{p_prefix}.out_proj.weight'].T)
        set_param([j_prefix, 'self_attn', 'out_proj', 'bias'], state_dict[f'{p_prefix}.out_proj.bias'])
        
        set_param([j_prefix, 'self_attn', 'ffn_0', 'kernel'], state_dict[f'{p_prefix}.ffn.0.weight'].T)
        set_param([j_prefix, 'self_attn', 'ffn_0', 'bias'], state_dict[f'{p_prefix}.ffn.0.bias'])
        set_param([j_prefix, 'self_attn', 'ffn_1', 'scale'], state_dict[f'{p_prefix}.ffn.1.weight'])
        set_param([j_prefix, 'self_attn', 'ffn_1', 'bias'], state_dict[f'{p_prefix}.ffn.1.bias'])
        set_param([j_prefix, 'self_attn', 'ffn_3', 'kernel'], state_dict[f'{p_prefix}.ffn.3.weight'].T)
        set_param([j_prefix, 'self_attn', 'ffn_3', 'bias'], state_dict[f'{p_prefix}.ffn.3.bias'])

        # Cross Attention
        p_prefix = f'transformers.{i}.cross_attn'
        
        set_param([j_prefix, 'cross_attn', 'to_qk', 'kernel'], state_dict[f'{p_prefix}.to_qk.weight'].T)
        set_param([j_prefix, 'cross_attn', 'to_qk', 'bias'], state_dict[f'{p_prefix}.to_qk.bias'])
        set_param([j_prefix, 'cross_attn', 'to_v', 'kernel'], state_dict[f'{p_prefix}.to_v.weight'].T)
        set_param([j_prefix, 'cross_attn', 'to_v', 'bias'], state_dict[f'{p_prefix}.to_v.bias'])
        set_param([j_prefix, 'cross_attn', 'to_out', 'kernel'], state_dict[f'{p_prefix}.to_out.weight'].T)
        set_param([j_prefix, 'cross_attn', 'to_out', 'bias'], state_dict[f'{p_prefix}.to_out.bias'])
        
        set_param([j_prefix, 'cross_attn', 'ffn_0', 'kernel'], state_dict[f'{p_prefix}.ffn.0.weight'].T)
        set_param([j_prefix, 'cross_attn', 'ffn_0', 'bias'], state_dict[f'{p_prefix}.ffn.0.bias'])
        set_param([j_prefix, 'cross_attn', 'ffn_1', 'scale'], state_dict[f'{p_prefix}.ffn.1.weight'])
        set_param([j_prefix, 'cross_attn', 'ffn_1', 'bias'], state_dict[f'{p_prefix}.ffn.1.bias'])
        set_param([j_prefix, 'cross_attn', 'ffn_3', 'kernel'], state_dict[f'{p_prefix}.ffn.3.weight'].T)
        set_param([j_prefix, 'cross_attn', 'ffn_3', 'bias'], state_dict[f'{p_prefix}.ffn.3.bias'])

        # Token Confidence
        if i < n_layers - 1:
            set_param([f'token_confidence_{i}', 'token', 'kernel'], state_dict[f'token_confidence.{i}.token.0.weight'].T)
            set_param([f'token_confidence_{i}', 'token', 'bias'], state_dict[f'token_confidence.{i}.token.0.bias'])

        # Match Assignment (matchability + final_proj)
        # MatchAssignment is at the same level as transformers in ModuleList
        set_param([f'log_assignment_{i}', 'matchability', 'kernel'], state_dict[f'log_assignment.{i}.matchability.weight'].T)
        set_param([f'log_assignment_{i}', 'matchability', 'bias'], state_dict[f'log_assignment.{i}.matchability.bias'])
        set_param([f'log_assignment_{i}', 'final_proj', 'kernel'], state_dict[f'log_assignment.{i}.final_proj.weight'].T)
        set_param([f'log_assignment_{i}', 'final_proj', 'bias'], state_dict[f'log_assignment.{i}.final_proj.bias'])

    # Save to msgpack
    with open(output_path, "wb") as f:
        f.write(serialization.to_bytes({'params': params}))
    
    print(f"Converted weights saved to {output_path}")

if __name__ == "__main__":
    pt_path = "/root/.cache/torch/hub/checkpoints/superpoint_lightglue.pth"
    output_path = "weights/superpoint_lightglue.msgpack"
    os.makedirs("weights", exist_ok=True)
    convert_lightglue_weights(pt_path, output_path)
