
import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

def test_ct_4x():
    in_c, out_c, k, s = 3, 4, 4, 4
    t_conv = torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=0, bias=False)
    kernel = t_conv.weight.detach().cpu().numpy()
    f_kernel = np.transpose(kernel, (2, 3, 1, 0))
    
    x = np.random.randn(1, 4, 4, in_c).astype(np.float32)
    t_x = torch.from_numpy(x).permute(0, 3, 1, 2)
    t_out = t_conv(t_x).permute(0, 2, 3, 1).detach().cpu().numpy()
    
    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.ConvTranspose(features=out_c, kernel_size=(k, k), strides=(s, s), padding='VALID', 
                                    use_bias=False, transpose_kernel=True, name='ct')(x)
    
    model = Model()
    variables = {'params': {'ct': {'kernel': f_kernel}}}
    f_out = model.apply(variables, jnp.array(x))
    
    print(f'Max Diff CT 4x: {np.abs(t_out - f_out).max()}')

test_ct_4x()
