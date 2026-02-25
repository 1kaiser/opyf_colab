# opyf_colab 🚀

A high-performance research ecosystem for fluid velocimetry (PIV) and automated civil engineering design, optimized for JAX-based computation and Google Colab.

## ✅ [Open In Colab](https://colab.research.google.com/github/1kaiser/opyf_colab/blob/main/opyf_Eumetsat_velocimetry.ipynb)

## 🌟 Key Features

### 1. 🏗️ Automated Canal Design (FreeCAD + JAX)
Integrated tools for designing irrigation canals following **Indian Standard (IS) codes**.
- **Differentiable Design:** Use JAX to optimize canal dimensions (Bed Width, Depth) for maximum economy and hydraulic efficiency.
- **IS Compliance:** Automatic validation against **IS 5968** (Layout) and **IS 10430** (Lined Canals).
- **CAD Automation:** Scripts to generate 3D models (STEP/GLB) and 2D engineering drawings.

### 2. 🌊 LSPIV & Velocimetry (Opyflow)
Advanced fluid flow analysis using the `opyflow` library, enabling high-resolution velocity vector mapping from video data.
- **Brague Flood Case Study:** Integrated demonstration of Large-Scale Particle Image Velocimetry (LSPIV) on real-world flood data.

### 4. 🌐 Client-Side Optimization (JAX-JS)
High-performance engineering logic running directly in the browser via WebGPU/Wasm.
- **Differentiable Linear Solvers:** Native implementation of Conjugate Gradient (CG) solvers in JavaScript.
- **Interactive Design:** Real-time canal dimension optimization for web-based 3D viewers.
- **Portability:** Move your JAX physics models from the cloud directly to the client's browser.

## 📁 Repository Structure

```text
.
├── canal_design/       # IS-Compliant Canal Design (Python/JAX)
├── tests/
│   └── Test_Brague_flood/  # LSPIV Brague River Case Study
├── web/
│   └── jax-js-fem/     # Browser-based JAX-JS FEM & Optimization
├── jax_3d_canal_reconstruction.ipynb # 3D Vision Pipeline
├── opyf_Eumetsat_velocimetry.ipynb # Core velocimetry notebook
└── README.md
```

## ⚖️ Model Weights
Converted JAX weights for Vision Models are available in the [GitHub Releases](https://github.com/1kaiser/opyf_colab/releases).

| Model | Size | Purpose |
| :--- | :--- | :--- |
| **Depth Pro** | 1.8 GB | 3D Metric Depth |
| **LightGlue** | 46 MB | Feature Matching |
| **SuperPoint** | 5 MB | Feature Extraction |

---
Created and maintained by [1kaiser](https://github.com/1kaiser)

