"""
modules/future_roadmap_viz.py
==============================
Generates assets/future_roadmap.png — architecture diagram showing five planned
future extensions and how they connect to the existing pipeline outputs.

  Canal section (Stage 7)  +  D8 thalweg (Stage 7b)
        │
        ├── GRAINnet          ─── d50/d90 → Manning's n → calibrated Q
        ├── JAX CFD           ─── supercritical flow stability, drop spacing
        ├── JAX FEM (super)   ─── lining stress IS 456 / IS 3370
        ├── JAX FEM (sub)     ─── foundation bearing IS 6403 / IS 8009
        └── Surface → Soil    ─── ortho texture → USCS class → soil params
"""

from pathlib import Path


def build_roadmap_figure(out_path: str = "assets/future_roadmap.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    C = {
        "existing":  "#1a4a7a",  "border_e":  "#3080d0",
        "grain":     "#0d4a4a",  "border_g":  "#20c0b0",
        "cfd":       "#1a5a3a",  "border_c":  "#30c060",
        "fem_sup":   "#5a3a1a",  "border_fs": "#d08030",
        "fem_sub":   "#5a2a1a",  "border_fb": "#e05020",
        "soil":      "#4a1a5a",  "border_s":  "#c030c0",
        "text":      "#e8e8e8",  "dim":       "#a0a0a0",
        "arrow":     "#505060",
    }

    def box(ax, x, y, w, h, fc, ec, lines, fs=7.8, title=None):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.07",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 0.20, title,
                    color="#ffffff", fontsize=fs + 1,
                    ha="center", va="top", fontweight="bold", zorder=4)
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h - 0.48 - i * 0.28, line,
                    color=C["dim"], fontsize=fs - 0.5,
                    ha="center", va="top", zorder=4)

    def arr(ax, x0, y0, x1, y1, col=C["arrow"]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6),
                    zorder=5)

    # ── title ─────────────────────────────────────────────────────────
    ax.text(10, 10.7, "Future Extension Roadmap", color=C["text"],
            fontsize=15, ha="center", va="top", fontweight="bold")
    ax.text(10, 10.32,
            "Grain size  ·  Flow stability  ·  Lining design  ·  Foundation design  ·  Soil composition",
            color=C["dim"], fontsize=9, ha="center", va="top")

    # ── Row 1: existing pipeline outputs ──────────────────────────────
    ax.text(0.15, 9.35, "Existing outputs", color=C["border_e"],
            fontsize=7.5, va="top", style="italic")
    ax.axhline(9.3, color="#252535", lw=0.8, ls="--", zorder=1)

    box(ax, 0.2,  7.6, 3.2, 1.6, C["existing"], C["border_e"],
        ["h(x,y)  Z_surface  Z_bed",
         "wet area · water volume",
         "D8 thalweg  S=1:55",
         "curvature R_min=1.6 m"],
        title="Flow Depth + D8 (6a–7b)")

    box(ax, 3.8,  7.6, 3.2, 1.6, C["existing"], C["border_e"],
        ["B=2.59 m  D=4.27 m",
         "V=1.30 m/s  Q=50 m³/s",
         "IS 5968 / IS 10430 ✓",
         "n=0.018 assumed"],
        title="Canal Design  (Stage 7)")

    box(ax, 7.4,  7.6, 3.2, 1.6, C["existing"], C["border_e"],
        ["canal_model.obj / .step",
         "canal_section.step",
         "IS alignment 1 km",
         "compound section option"],
        title="FreeCAD 3D  (Stage 7c)")

    box(ax, 11.0, 7.6, 4.2, 1.6, C["existing"], C["border_e"],
        ["Ortho.tif  2.4 mm/px",
         "MNT.xyz  4.9 mm pts",
         "EPSG:2154 Lambert-93",
         "bank GCPs (LightGlue)"],
        title="Geospatial Data  (Stage 0–6b)")

    box(ax, 15.6, 7.6, 4.2, 1.6, C["existing"], C["border_e"],
        ["337-px D8 path · S=1:55",
         "R_min=1.6 m vs IS 1000 m",
         "bearing_deg(s)",
         "slope_local(s)"],
        title="D8 Thalweg  (Stage 7b)")

    # ── Row 2: future modules ──────────────────────────────────────────
    ax.text(0.15, 7.0, "Future modules", color=C["dim"],
            fontsize=7.5, va="top", style="italic")
    ax.axhline(6.95, color="#252535", lw=0.8, ls="--", zorder=1)

    # A — GRAINnet
    box(ax, 0.2,  4.5, 3.5, 2.1, C["grain"], C["border_g"],
        ["d50 / d84 / d90 (mm)",
         "Strickler:  n = d90^(1/6)/Ks",
         "spatially variable n(X,Y)",
         "recalibrate Q  (±40–100 %)"],
        title="GRAINnet  (1kaiser/GRAINnet)")

    # B — JAX CFD
    box(ax, 4.1,  4.5, 3.5, 2.1, C["cfd"], C["border_c"],
        ["Fr = V/√(gD)  →  super/subcritical",
         "hydraulic jump loc X_j",
         "drop structure spacing",
         "EGL(x)  overtopping check"],
        title="JAX Fluid Solver  (superstructure)")

    # C — JAX FEM superstructure
    box(ax, 8.0,  4.5, 3.8, 2.1, C["fem_sup"], C["border_fs"],
        ["hydrostatic + earth pressure",
         "IS 456:2000 limit-state",
         "IS 3370:2009 crack ≤ 0.2 mm",
         "steel schedule  slab t"],
        title="JAX FEM — Lining  (IS 456/3370)")

    # D — JAX FEM substructure
    box(ax, 12.2, 4.5, 3.8, 2.1, C["fem_sub"], C["border_fb"],
        ["q_ult = cNc + γDNq  (IS 6403)",
         "settlement δ  (IS 8009)",
         "seepage uplift  cut-off depth",
         "scour apron length (IS 8237)"],
        title="JAX FEM — Foundation  (IS 6403/8009)")

    # E — Surface → Soil
    box(ax, 16.3, 4.5, 3.5, 2.1, C["soil"], C["border_s"],
        ["SegNet/DINO → surface class",
         "GW·SW·CL·CH (USCS/IS 1498)",
         "φ  c  γ  k  (0–1 m depth)",
         "no borehole needed"],
        title="Surface → Soil  (ortho classification)")

    # ── Row 3: combined design outputs ────────────────────────────────
    box(ax, 0.2,  1.2, 5.8, 2.9, "#0e1e2e", C["border_e"],
        ["Manning n(X,Y) from GRAINnet",
         "Q re-estimated with real n",
         "IS 10430 n=0.018 validated/corrected",
         "Drop structures: number · spacing",
         "Stilling basin: L_b · depth · apron",
         "Fr profile along thalweg"],
        title="Hydraulic Validation + Roughness Calibration")

    box(ax, 6.4,  1.2, 6.8, 2.9, "#1e1020", C["border_e"],
        ["Lining: slab t · steel A_s (mm²/m)",
         "Joint spacing (thermal IS 456)",
         "Foundation: D_f · raft dims",
         "Cut-off wall depth (anti-seepage)",
         "Scour apron length",
         "IS 456 / IS 3370 / IS 6403 / IS 8009 / IS 8237"],
        title="Complete Structural + Geotechnical Design Dossier")

    box(ax, 13.6, 1.2, 6.2, 2.9, "#1e0e20", C["border_e"],
        ["soil_class(X,Y): GW / SW / CL",
         "φ(X,Y)  c(X,Y)  γ(X,Y)  k(X,Y)",
         "Bearing capacity q_ult(X,Y)",
         "Settlement δ(X,Y)",
         "Permeability k → seepage head",
         "0–1 m profile without borehole"],
        title="Spatial Soil Parameter Map")

    # ── arrows top → middle ───────────────────────────────────────────
    arr(ax, 1.8,  7.6, 1.9,  6.6)    # flow depth → GRAINnet
    arr(ax, 1.8,  7.6, 5.8,  6.6)    # flow depth → CFD
    arr(ax, 5.4,  7.6, 9.8,  6.6)    # canal design → FEM super
    arr(ax, 9.0,  7.6, 14.0, 6.6)    # canal 3D → FEM sub
    arr(ax, 13.1, 7.6, 18.0, 6.6)    # geodata → soil
    arr(ax, 17.7, 7.6, 14.1, 6.6)    # D8 thalweg → FEM sub
    arr(ax, 5.4,  7.6, 5.9,  6.6)    # canal design → CFD

    # ── arrows middle → bottom ────────────────────────────────────────
    arr(ax, 2.0,  4.5, 3.2,  4.1)    # GRAINnet → hydraulic validation
    arr(ax, 5.9,  4.5, 4.5,  4.1)    # CFD → hydraulic validation
    arr(ax, 9.9,  4.5, 9.8,  4.1)    # FEM super → structural dossier
    arr(ax, 14.1, 4.5, 10.5, 4.1)    # FEM sub → structural dossier
    arr(ax, 18.1, 4.5, 16.7, 4.1)    # soil → soil map
    arr(ax, 18.1, 4.5, 13.0, 4.1)    # soil → structural dossier (params)

    # ── GRAINnet ↔ soil (shared grain info) ──────────────────────────
    ax.annotate("", xy=(3.75, 5.5), xytext=(3.55, 5.5),
                arrowprops=dict(arrowstyle="<->", color="#20c0b0", lw=1.2),
                zorder=5)

    # ── legend ────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc=C["existing"], ec=C["border_e"],  label="Existing pipeline"),
        mpatches.Patch(fc=C["grain"],    ec=C["border_g"],  label="GRAINnet — grain size → n"),
        mpatches.Patch(fc=C["cfd"],      ec=C["border_c"],  label="JAX fluid solver"),
        mpatches.Patch(fc=C["fem_sup"],  ec=C["border_fs"], label="JAX FEM — lining (IS 456/3370)"),
        mpatches.Patch(fc=C["fem_sub"],  ec=C["border_fb"], label="JAX FEM — foundation (IS 6403/8009)"),
        mpatches.Patch(fc=C["soil"],     ec=C["border_s"],  label="Surface → soil composition"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=8,
              facecolor="#1a1a2a", edgecolor="#3a3a4a",
              labelcolor="#e0e0e0", ncol=6,
              bbox_to_anchor=(0.5, -0.01))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Future roadmap → {out_path}")


if __name__ == "__main__":
    build_roadmap_figure()
    print("Done.")
