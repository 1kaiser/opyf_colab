
import numpy as np
for i in [4, 3, 2, 1]:
    pt = np.load(f'output/pytorch_inference/dpt_path{i}.npy')
    jax = np.load(f'output/jax_inference/dpt_path{i}.npy')
    mse = np.mean((pt - jax)**2)
    max_diff = np.max(np.abs(pt - jax))
    print(f'Path {i} MSE: {mse:.2e}, Max Diff: {max_diff:.2e}')
