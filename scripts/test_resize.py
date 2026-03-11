
import torch
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

def resize_ac(x, shape):
    B, H, W, C = x.shape
    _, Hn, Wn, _ = shape
    
    hc = jnp.linspace(0, H - 1, Hn)
    wc = jnp.linspace(0, W - 1, Wn)
    gh, gw = jnp.meshgrid(hc, wc, indexing='ij')
    coords = jnp.stack([gh, gw], axis=0)
    
    xr = x.transpose(1, 2, 0, 3).reshape(H, W, -1)
    
    # map_coordinates order=1 is bilinear
    # mode='nearest' or 'constant' with cval? 
    # PyTorch align_corners=True doesn't need padding if we stay within [0, H-1]
    mapped = jax.vmap(lambda f: map_coordinates(f, coords, order=1, mode='nearest'), in_axes=-1, out_axes=-1)(xr)
    
    return mapped.reshape(Hn, Wn, B, C).transpose(2, 0, 1, 3)

x = np.random.randn(1, 4, 4, 3).astype(np.float32)
t_x = torch.from_numpy(x).permute(0, 3, 1, 2)
t_out = torch.nn.functional.interpolate(t_x, scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1).numpy()

j_out = resize_ac(jnp.array(x), (1, 8, 8, 3)).block_until_ready()
diff = np.abs(t_out - j_out)
print(f'Max diff: {diff.max()}')
print('Corners PT:', t_out[0,0,0], t_out[0,0,-1])
print('Corners JAX:', j_out[0,0,0], j_out[0,0,-1])
