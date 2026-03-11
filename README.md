# opyf_colab 🚀

A high-performance research ecosystem for fluid velocimetry (PIV), 3D metric reconstruction, and automated civil engineering design, optimized for JAX-based computation and Google Colab.

## ✅ [Open In Colab](https://colab.research.google.com/github/1kaiser/opyf_colab/blob/main/jax_3d_reconstruction_colab.ipynb)

## 🌟 Key Features

### 1. 🏗️ Automated Canal Design (FreeCAD + JAX)
Integrated tools for designing irrigation canals following **Indian Standard (IS) codes**.
- **Differentiable Design:** Use JAX to optimize canal dimensions (Bed Width, Depth) for maximum economy and hydraulic efficiency.
- **IS Compliance:** Automatic validation against **IS 5968** (Layout) and **IS 10430** (Lined Canals).
- **CAD Automation:** Scripts to generate 3D models (STEP/GLB) and 2D engineering drawings.

### 2. 📸 JAX 3D Reconstruction Pipeline
High-precision metric 3D reconstruction using state-of-the-art JAX-based vision models.
- **Metric Depth:** Powered by **Depth Pro** for absolute scale estimation.
- **Feature Matching:** **SuperPoint** + **LightGlue** for robust correspondence.
- **Zonal Alignment:** Concentric zonal registration for high-accuracy large-scale alignment.
- **Mesh Export:** Automatic generation of `.ply` point clouds and `.glb` Poisson-reconstructed meshes.

### 3. 🌊 LSPIV & Velocimetry (Opyflow)
Advanced fluid flow analysis using the `opyflow` library, enabling high-resolution velocity vector mapping from video data.
- **Brague Flood Case Study:** Integrated demonstration of Large-Scale Particle Image Velocimetry (LSPIV) on real-world flood data.

### 4. 🌐 Client-Side Optimization (JAX-JS)
High-performance engineering logic running directly in the browser via WebGPU/Wasm.

## 📁 Repository Structure

```text
opyf_colab/
├── data/
│   └── pinecone_subset/    # 📸 Example 3D Reconstruction Dataset
├── models/
│   └── jax/                # 🧠 JAX Vision Model Implementations
│       ├── jax_depth_pro/
│       ├── jax_lightglue/
│       ├── jax_reconstruction/
│       └── jax_vggt/
├── pipelines/              # 🏗️ Modular Reconstruction Pipelines
│   ├── pipeline_jax.py     # Zonal Reconstruction (Depth Pro + LG)
│   └── pipeline_vggt3_jax.py
├── inference/              # 🧪 Standalone Inference Scripts
├── canal_design/           # 📐 IS-Compliant Canal Design (JAX)
├── web/
│   └── jax-js-fem/         # 🌐 Browser-based JAX-JS Optimization
├── tests/
│   └── Test_Brague_flood/  # 🌊 LSPIV Brague River Case Study
├── jax_3d_reconstruction_colab.ipynb  # 🚀 Core 3D Vision Notebook
├── opyf_Eumetsat_velocimetry.ipynb    # 🌊 Core Velocimetry Notebook
└── README.md
```

## 📊 Visual Results

### 3D Metric Reconstruction
| Metric Depth (Depth Pro) | Feature Matching (LightGlue) |
| :---: | :---: |
| ![Depth Result](output/depth_result.jpg) | ![Match Result](output/match_result.jpg) |
| *Absolute depth estimation from a single view* | *Robust matching across different viewpoints* |

### Final 3D Mesh (GLB)
![3D Mesh Screenshot](https://raw.githubusercontent.com/1kaiser/opyf_colab/main/data/assets/mesh_preview.png)
*(Poisson-reconstructed mesh from aligned zonal point clouds)*

## ⚖️ Model Weights
Converted JAX weights for Vision Models are available in the [GitHub Releases](https://github.com/1kaiser/d_jax/releases).

| Model | Size | Purpose |
| :--- | :--- | :--- |
| **Depth Pro** | 1.8 GB | 3D Metric Depth |
| **LightGlue** | 46 MB | Feature Matching |
| **SuperPoint** | 5 MB | Feature Extraction |

---
Created and maintained by [1kaiser](https://github.com/1kaiser)
