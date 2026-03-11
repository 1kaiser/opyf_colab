
import numpy as np
for i in range(4):
    pt = np.load(f'output/pytorch_inference/dpt_layers_rn_{i}.npy')
    jax = np.load(f'output/jax_inference/dpt_layers_rn_{i}.npy')
    mse = np.mean((pt - jax)**2)
    print(f'Layer RN {i} MSE: {mse:.2e}')
