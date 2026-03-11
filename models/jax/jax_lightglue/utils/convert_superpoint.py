import torch
import numpy as np
from flax import serialization
import os

def convert_superpoint_weights(pt_path, output_path):
    state_dict = torch.load(pt_path, map_location="cpu")
    params = {}

    def set_param(path, value):
        curr = params
        for part in path[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[path[-1]] = np.array(value)

    def convert_conv(p_key, j_key):
        w = state_dict[f'{p_key}.weight'].numpy()
        # [out, in, k, k] -> [k, k, in, out]
        set_param([j_key, 'kernel'], w.transpose(2, 3, 1, 0))
        set_param([j_key, 'bias'], state_dict[f'{p_key}.bias'].numpy())

    # Encoder
    for i in range(1, 5):
        for sub in ['a', 'b']:
            p_key = f'conv{i}{sub}'
            j_key = f'conv{i}{sub}'
            convert_conv(p_key, j_key)

    # Heads
    convert_conv('convPa', 'convPa')
    convert_conv('convPb', 'convPb')
    convert_conv('convDa', 'convDa')
    convert_conv('convDb', 'convDb')

    # Save
    with open(output_path, "wb") as f:
        f.write(serialization.to_bytes({'params': params}))
    print(f"Converted SuperPoint weights saved to {output_path}")

if __name__ == "__main__":
    # URL: https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth
    # It might be in the cache if we ran capture script
    pt_path = "/root/.cache/torch/hub/checkpoints/superpoint_v1.pth"
    output_path = "weights/superpoint.msgpack"
    os.makedirs("weights", exist_ok=True)
    
    if not os.path.exists(pt_path):
        print(f"Weights not found at {pt_path}. Please run capture script first or download them.")
    else:
        convert_superpoint_weights(pt_path, output_path)
