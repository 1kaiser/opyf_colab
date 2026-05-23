# opyf_colab — Claude Code project settings

## Python environment

**Always** prefix Python commands with the conda environment:

```bash
conda run -n num_gpu python3 <script>
# or
conda run -n num_python python3 <script>
```

The system `/usr/bin/python3` has no packages. All dependencies (numpy, rasterio, JAX, scikit-image, matplotlib, etc.) live in the `num_gpu` conda environment. Use `num_gpu` as the default (has JAX GPU support); fall back to `num_python` if needed.

The same applies for Node.js scripts — use `conda run -n num_gpu node` or `conda run -n num_python node`.

## JAX

Always set `JAX_PLATFORMS=cpu` unless explicitly running on GPU:

```bash
JAX_PLATFORMS=cpu conda run -n num_gpu python3 pipeline.py
```
