import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import torch

def test_conv_transpose(transpose_kernel=False, mapping=(2, 3, 0, 1)):
    # PyTorch setup
    in_c, out_c, k = 3, 4, 2
    t_conv = torch.nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k, padding=0, bias=False)
    
    kernel = t_conv.weight.detach().cpu().numpy()
    # PyTorch kernel: (in, out, k, k)
    
    # Flax mapping
    f_kernel = np.transpose(kernel, mapping)
    
    # Input
    x = np.random.randn(1, 4, 4, in_c).astype(np.float32)
    t_x = torch.from_numpy(x).permute(0, 3, 1, 2)
    
    # PyTorch output
    t_out = t_conv(t_x).permute(0, 2, 3, 1).detach().cpu().numpy()
    
    # Flax output
    class Model(nn.Module):
        features: int
        transpose_kernel: bool
        @nn.compact
        def __call__(self, x):
            return nn.ConvTranspose(features=self.features, kernel_size=(k, k), padding='VALID', 
                                    use_bias=False, transpose_kernel=self.transpose_kernel, name='conv_trans')(x)
    
    model = Model(features=out_c, transpose_kernel=transpose_kernel)
    variables = {'params': {'conv_trans': {'kernel': f_kernel}}}
    f_out = model.apply(variables, jnp.array(x))
    
    diff = np.abs(t_out - f_out).max()
    print(f"TK={transpose_kernel}, Mapping={mapping}, Max Diff={diff}")

print("Testing ConvTranspose mappings:")
try: test_conv_transpose(False, (2, 3, 0, 1))
except Exception as e: print(f"Failed False, (2,3,0,1): {e}")
try: test_conv_transpose(False, (2, 3, 1, 0))
except Exception as e: print(f"Failed False, (2,3,1,0): {e}")
try: test_conv_transpose(True, (2, 3, 0, 1))
except Exception as e: print(f"Failed True, (2,3,0,1): {e}")
try: test_conv_transpose(True, (2, 3, 1, 0))
except Exception as e: print(f"Failed True, (2,3,1,0): {e}")
