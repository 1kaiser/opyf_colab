
import torch
import jax
import jax.numpy as jnp
import numpy as np

def ps_jax(x, r):
    B, H, W, Crr = x.shape
    C = Crr // (r**2)
    # PyTorch pixel_shuffle expects elements to be arranged such that 
    # the last dimension (C*r*r) contains C chunks of r*r elements.
    # Each r*r chunk is reshaped to (r, r) and mapped to the spatial grid.
    # So we should reshape to (B, H, W, C, r, r)
    x = x.reshape(B, H, W, C, r, r)
    # Then transpose to (B, H, r, W, r, C)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    # Final reshape
    return x.reshape(B, H * r, W * r, C)

r = 2
C = 3
H, W = 4, 4
x = np.random.randn(1, H, W, C*r*r).astype(np.float32)
t_x = torch.from_numpy(x).permute(0, 3, 1, 2)
t_out = torch.nn.functional.pixel_shuffle(t_x, r).permute(0, 2, 3, 1).numpy()

j_out = ps_jax(jnp.array(x), r)
diff = np.abs(t_out - j_out)
print(f'Max diff: {diff.max()}')
