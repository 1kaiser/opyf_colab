import torch
import numpy as np
from flax import serialization
import os

def convert_depth_pro_weights(pt_path, output_path):
    state_dict = torch.load(pt_path, map_location="cpu")
    params = {}

    def set_param(path, value):
        curr = params
        for part in path[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[path[-1]] = np.array(value)

    def convert_vit(prefix, j_prefix):
        # prefix: e.g. 'encoder.patch_encoder'
        # j_prefix: e.g. ['encoder', 'patch_encoder']
        
        # cls_token
        set_param(j_prefix + ['cls_token'], state_dict[f'{prefix}.cls_token'])
        # pos_embed
        set_param(j_prefix + ['pos_embed'], state_dict[f'{prefix}.pos_embed'])
        
        # patch_embed
        w = state_dict[f'{prefix}.patch_embed.proj.weight'].numpy()
        set_param(j_prefix + ['patch_embed', 'proj', 'kernel'], w.transpose(2, 3, 1, 0))
        set_param(j_prefix + ['patch_embed', 'proj', 'bias'], state_dict[f'{prefix}.patch_embed.proj.bias'])
        
        # blocks
        for i in range(24): # Depth 24 for ViT-L
            p_b = f'{prefix}.blocks.{i}'
            j_b = j_prefix + [f'blocks_{i}']
            
            # norm1
            set_param(j_b + ['norm1', 'scale'], state_dict[f'{p_b}.norm1.weight'])
            set_param(j_b + ['norm1', 'bias'], state_dict[f'{p_b}.norm1.bias'])
            
            # attn
            set_param(j_b + ['attn', 'qkv', 'kernel'], state_dict[f'{p_b}.attn.qkv.weight'].T)
            set_param(j_b + ['attn', 'qkv', 'bias'], state_dict[f'{p_b}.attn.qkv.bias'])
            set_param(j_b + ['attn', 'proj', 'kernel'], state_dict[f'{p_b}.attn.proj.weight'].T)
            set_param(j_b + ['attn', 'proj', 'bias'], state_dict[f'{p_b}.attn.proj.bias'])
            
            # ls1
            if f'{p_b}.ls1.gamma' in state_dict:
                set_param(j_b + ['ls1', 'gamma'], state_dict[f'{p_b}.ls1.gamma'])
            
            # norm2
            set_param(j_b + ['norm2', 'scale'], state_dict[f'{p_b}.norm2.weight'])
            set_param(j_b + ['norm2', 'bias'], state_dict[f'{p_b}.norm2.bias'])
            
            # mlp
            set_param(j_b + ['mlp', 'fc1', 'kernel'], state_dict[f'{p_b}.mlp.fc1.weight'].T)
            set_param(j_b + ['mlp', 'fc1', 'bias'], state_dict[f'{p_b}.mlp.fc1.bias'])
            set_param(j_b + ['mlp', 'fc2', 'kernel'], state_dict[f'{p_b}.mlp.fc2.weight'].T)
            set_param(j_b + ['mlp', 'fc2', 'bias'], state_dict[f'{p_b}.mlp.fc2.bias'])
            
            # ls2
            if f'{p_b}.ls2.gamma' in state_dict:
                set_param(j_b + ['ls2', 'gamma'], state_dict[f'{p_b}.ls2.gamma'])
        
        # final norm
        set_param(j_prefix + ['norm', 'scale'], state_dict[f'{prefix}.norm.weight'])
        set_param(j_prefix + ['norm', 'bias'], state_dict[f'{prefix}.norm.bias'])

    # 1. Patch Encoder
    convert_vit('encoder.patch_encoder', ['encoder', 'patch_encoder'])
    # 2. Image Encoder
    convert_vit('encoder.image_encoder', ['encoder', 'image_encoder'])

    # 3. Encoder Projections and Upsampling
    def convert_conv(p_key, j_path, is_transpose=False):
        w = state_dict[f'{p_key}.weight'].numpy()
        if is_transpose:
            # PyTorch: [in, out, k, k] -> JAX: [k, k, out, in]
            set_param(j_path + ['kernel'], w.transpose(2, 3, 1, 0))
        else:
            # PyTorch: [out, in, k, k] -> JAX: [k, k, in, out]
            set_param(j_path + ['kernel'], w.transpose(2, 3, 1, 0))
        if f'{p_key}.bias' in state_dict:
            set_param(j_path + ['bias'], state_dict[f'{p_key}.bias'])

    # Encoder projections
    names = ['upsample_latent0', 'upsample_latent1', 'upsample0', 'upsample1', 'upsample2']
    for name in names:
        # Each is a Sequential
        p_base = f'encoder.{name}'
        # JAX names: upsample_latent0_0, upsample_latent0_1, etc.
        # Find all layers in Sequential
        i = 0
        while f'{p_base}.{i}.weight' in state_dict:
            is_tr = 'ConvTranspose' in str(type(dict(torch.load(pt_path, map_location='cpu')))) # Hacky
            # Actually, I know which ones are Transpose from my JAX model
            # layers 1+ for latent0 (3 layers), 1+ for latent1 (2 layers), 1 for others
            is_tr = (i > 0)
            convert_conv(f'{p_base}.{i}', ['encoder', f'{name}_{i}'], is_transpose=is_tr)
            i += 1

    convert_conv('encoder.upsample_lowres', ['encoder', 'upsample_lowres'], is_transpose=True)
    convert_conv('encoder.fuse_lowres', ['encoder', 'fuse_lowres'])

    # 4. Decoder
    # convs
    for i in range(5):
        if f'decoder.convs.{i}.weight' in state_dict:
            convert_conv(f'decoder.convs.{i}', ['decoder', f'convs_{i}'])
            
    # fusions
    for i in range(5):
        p_f = f'decoder.fusions.{i}'
        j_f = ['decoder', f'fusions_{i}']
        
        # FeatureFusionBlock2d has resnet1, resnet2, deconv, out_conv
        def convert_res(p_res, j_res):
            # resnet.residual is Sequential: ReLU, Conv, ReLU, Conv
            # My JAX ResidualBlock uses manual calls, names Conv_0, Conv_1
            # Actually, I named them block in __call__
            # JAX names will be 'Conv_0', 'Conv_1'
            convert_conv(f'{p_res}.residual.1', j_res + ['Conv_0'])
            convert_conv(f'{p_res}.residual.3', j_res + ['Conv_1'])

        convert_res(f'{p_f}.resnet1', j_f + ['resnet1'])
        convert_res(f'{p_f}.resnet2', j_f + ['resnet2'])
        
        if f'{p_f}.deconv.weight' in state_dict:
            convert_conv(f'{p_f}.deconv', j_f + ['deconv'], is_transpose=True)
        convert_conv(f'{p_f}.out_conv', j_f + ['out_conv'])

    # 5. Head
    convert_conv('head.0', ['head_0'])
    convert_conv('head.1', ['head_1'], is_transpose=True)
    convert_conv('head.2', ['head_2'])
    convert_conv('head.4', ['head_4'])

    # 6. FOV
    # encoder
    convert_vit('fov.encoder.0', ['fov', 'encoder_0'])
    set_param(['fov', 'encoder_1', 'kernel'], state_dict['fov.encoder.1.weight'].T)
    set_param(['fov', 'encoder_1', 'bias'], state_dict['fov.encoder.1.bias'])
    
    # downsample
    convert_conv('fov.downsample.0', ['fov', 'downsample_0'])
    
    # head
    convert_conv('fov.head.0', ['fov', 'head_0'])
    convert_conv('fov.head.2', ['fov', 'head_2'])
    convert_conv('fov.head.4', ['fov', 'head_4'])

    # Save
    with open(output_path, "wb") as f:
        f.write(serialization.to_bytes({'params': params}))
    print(f"Converted Depth Pro weights saved to {output_path}")

if __name__ == "__main__":
    pt_path = "depth_pro_repo/checkpoints/depth_pro.pt"
    output_path = "weights/depth_pro.msgpack"
    convert_depth_pro_weights(pt_path, output_path)
