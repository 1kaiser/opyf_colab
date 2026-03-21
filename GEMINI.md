# opyf_colab: Gemini Mandates

## Environment & Tooling
- **Conda Environment:** Always use the `num_python` environment located at `/home/kaiser/.miniforge/envs/num_python`.
- **Conda Executable:** Use `/home/kaiser/.miniforge/bin/conda`.
- **Package Management:** Prefer `uv` over `pip` for installations within the environment when available.
- **GitHub CLI:** For `gh` operations, ensure `GH_CONFIG_DIR=/home/kaiser/.config/gh` is set.

## Engineering Standards
- **Execution Tracking:** Use the `time_it`, `progress_bar`, and `Timer` utilities from `jax_reconstruction/utils.py` for all long-running scripts (PIV analysis, 3D reconstruction, CAD optimization).
- **Progress Bars:** Always include `tqdm` progress bars for batch processing.
- **Dual-Mode Search:** When researching methodology or objectives, perform both Grep and Semantic searches.

## Project Scope
- This is a PhD-level research project focused on JAX-based vision (3D reconstruction), fluid velocimetry (LSPIV via `opyflow`), and IS-compliant automated canal design (FreeCAD).
