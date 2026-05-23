"""
modules/future_roadmap_viz.py
==============================
Generates assets/future_roadmap.png — a single figure showing the three
planned future extensions and how they connect to the existing pipeline output.

  Canal section (Stage 7)
        │
        ├── JAX CFD / Saint-Venant  ─── flood routing, Q_sim vs Q_target
        ├── JAX FEM                 ─── structural: lining stress, reinforcement
        └── Surface → Soil          ─── ortho texture → USCS class → bearing capacity
"""

from pathlib import Path


def build_roadmap_figure(out_path: str = "assets/future_roadmap.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    import numpy as np

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ── colour palette ────────────────────────────────────────────────
    C = {
        "existing":  "#1a4a7a",
        "cfd":       "#1a5a3a",
        "fem":       "#5a3a1a",
        "soil":      "#4a1a5a",
        "border_e":  "#3080d0",
        "border_c":  "#30c060",
        "border_f":  "#d08030",
        "border_s":  "#c030c0",
        "text":      "#e8e8e8",
        "dim":       "#a0a0a0",
        "arrow":     "#606070",
    }

    def box(ax, x, y, w, h, fc, ec, text_lines, fontsize=8.5, title=None):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 0.22, title, color="#ffffff",
                    fontsize=fontsize + 1, ha="center", va="top",
                    fontweight="bold", zorder=4)
        for i, line in enumerate(text_lines):
            ax.text(x + w/2, y + h - 0.52 - i * 0.32, line,
                    color=C["dim"], fontsize=fontsize - 0.5,
                    ha="center", va="top", zorder=4)

    def arrow(ax, x0, y0, x1, y1, color=C["arrow"]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                    zorder=5)

    # ──────────────────────────────────────────────────────────────────
    # Row 1 — existing pipeline outputs (inputs to future work)
    # ──────────────────────────────────────────────────────────────────
    ax.text(8, 8.7, "Future Extension Roadmap", color=C["text"], fontsize=14,
            ha="center", va="top", fontweight="bold")
    ax.text(8, 8.35, "Connecting existing pipeline outputs to structural, hydraulic, and geotechnical design",
            color=C["dim"], fontsize=9, ha="center", va="top")

    # Existing outputs (top row)
    box(ax, 0.3, 6.5, 3.2, 1.6, C["existing"], C["border_e"],
        ["h(x,y) flow depth raster", "Z_surface, Z_bed", "wet area ~39 700 m²",
         "water volume (m³)"],
        title="Flow Depth  (Stage 6a/b)")

    box(ax, 4.1, 6.5, 3.2, 1.6, C["existing"], C["border_e"],
        ["B = 2.59 m, D = 4.27 m", "V = 1.30 m/s, Q = 50 m³/s",
         "IS 5968 / IS 10430 ✓", "canal_params.json"],
        title="Canal Design  (Stage 7)")

    box(ax, 7.9, 6.5, 3.2, 1.6, C["existing"], C["border_e"],
        ["canal_model.obj / .step", "canal_section.step", "1 km IS alignment",
         "compound section option"],
        title="FreeCAD 3D Model  (Stage 7)")

    box(ax, 11.7, 6.5, 3.8, 1.6, C["existing"], C["border_e"],
        ["Ortho.tif  2.4 mm/px", "MNT.xyz  4.9 mm pts", "EPSG:2154 Lambert-93",
         "bank GCPs (LightGlue)"],
        title="Geospatial Data  (Stage 0–6b)")

    # ──────────────────────────────────────────────────────────────────
    # Row 2 — future modules
    # ──────────────────────────────────────────────────────────────────

    # A — JAX CFD Water Simulation
    box(ax, 0.3, 3.5, 4.6, 2.6, C["cfd"], C["border_c"],
        ["Saint-Venant 2D shallow water",
         "flood routing: Q_sim(t) vs Q_target",
         "inundation extent vs Ortho.tif",
         "velocity field V(x,y,t)",
         "Candidate: google/jax-cfd",
         "or web/jax-js-fem prototype"],
        title="JAX Water Simulation")

    # B — JAX FEM Structural
    box(ax, 5.5, 3.5, 4.6, 2.6, C["fem"], C["border_f"],
        ["Canal section → FEM mesh",
         "Hydrostatic + soil pressure loads",
         "IS 456:2000 concrete design",
         "IS 3370:2009 liquid-retaining",
         "Steel schedule, slab thickness",
         "Candidate: tianjuxue/jax-fem"],
        title="JAX FEM — Structural")

    # C — Surface → Soil
    box(ax, 10.7, 3.5, 5.0, 2.6, C["soil"], C["border_s"],
        ["Ortho.tif → SegNet / DINO features",
         "Pixel class: cobble / sand / concrete",
         "USCS / IS 1498 texture→soil lookup",
         "soil_class(X,Y): GW · SP · CL …",
         "Bearing capacity q_ult(x,y)",
         "0–1 m composition without borehole"],
        title="Surface Material → Soil Composition")

    # ──────────────────────────────────────────────────────────────────
    # Row 3 — combined outputs
    # ──────────────────────────────────────────────────────────────────
    box(ax, 1.5, 0.5, 5.5, 2.6, "#1a2a3a", C["border_e"],
        ["Q_sim ≈ Q_target  (hydraulic validation)",
         "Flood map overlay on Ortho  (accuracy check)",
         "Peak discharge timing",
         "Optimise slope / dimensions iteratively"],
        title="Hydraulic Validation")

    box(ax, 8.5, 0.5, 6.8, 2.6, "#2a1a2a", C["border_e"],
        ["Reinforcement drawing (mm² / m)",
         "Foundation depth from bearing capacity",
         "Settlement estimate from soil composition",
         "Complete design dossier (IS 456 / IS 3370 / IS 1498)"],
        title="Structural + Geotechnical Design")

    # ──────────────────────────────────────────────────────────────────
    # Arrows — top row → future modules
    # ──────────────────────────────────────────────────────────────────
    # flow depth → JAX CFD
    arrow(ax, 1.9, 6.5, 2.1, 6.1); arrow(ax, 2.1, 6.1, 2.6, 6.1)
    # canal params → JAX CFD
    arrow(ax, 5.7, 6.5, 3.8, 6.1)
    # canal params → JAX FEM
    arrow(ax, 5.7, 6.5, 7.8, 6.1)
    # canal 3D → JAX FEM
    arrow(ax, 9.5, 6.5, 7.8, 6.1)
    # geodata → soil
    arrow(ax, 13.6, 6.5, 13.2, 6.1)
    # canal params → soil (soil loading)
    arrow(ax, 7.3, 6.5, 12.0, 6.1)

    # future modules → outputs
    arrow(ax, 2.6, 3.5, 4.2, 3.1)
    arrow(ax, 7.8, 3.5, 10.0, 3.1)
    arrow(ax, 13.2, 3.5, 12.0, 3.1)

    # label lines
    for y_line in [6.45, 3.45]:
        ax.axhline(y_line, color="#2a2a3a", lw=0.8, ls="--", zorder=1)

    ax.text(0.15, 6.45, "Existing outputs", color=C["border_e"],
            fontsize=7.5, va="top", style="italic")
    ax.text(0.15, 3.45, "Future modules", color=C["dim"],
            fontsize=7.5, va="top", style="italic")

    legend_handles = [
        mpatches.Patch(fc=C["existing"], ec=C["border_e"], label="Existing pipeline"),
        mpatches.Patch(fc=C["cfd"],      ec=C["border_c"], label="JAX CFD / water sim"),
        mpatches.Patch(fc=C["fem"],      ec=C["border_f"], label="JAX FEM / structural"),
        mpatches.Patch(fc=C["soil"],     ec=C["border_s"], label="Surface → soil"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              facecolor="#1a1a2a", edgecolor="#3a3a4a", labelcolor="#e0e0e0",
              ncol=4, bbox_to_anchor=(1.0, -0.01))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Future roadmap figure → {out_path}")


if __name__ == "__main__":
    build_roadmap_figure()
    print("Done.")
