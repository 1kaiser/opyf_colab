"""
modules/canal_optimizer.py
===========================
Canal hydraulic optimizer (IS 5968:1987 + IS 10430:2000).

Method — analytical minimum wetted perimeter (Chow 1959, Das 2000):
-------
For a trapezoidal section the minimum-cost (minimum wetted perimeter) section
satisfies the classical result R = D/2, giving explicit formulas:

    coef = 2*sqrt(1+m²) - m
    D    = [Q*n*2^(2/3) / (sqrt(S)*coef)]^(3/8)   ← closed-form
    B    = 2*D*(sqrt(1+m²) - m)                    ← closed-form

No gradient descent is needed. If the resulting velocity violates IS 10430
limits, D is found by Newton's method along the V=V_limit line instead.

Public API
----------
    optimise_canal(Q_target, S_long, n, out_path) -> dict
    validate_is(params)  -> dict of pass/fail flags
    print_report(params)

CLI
---
    python3 modules/canal_optimizer.py --Q 50 --slope 0.0002 --out canal_design/canal_params.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import jax.numpy as jnp

# ── IS code tables ────────────────────────────────────────────────────────────

# IS 5968:1987, Table 1, Cl 8.1 — minimum curve radius (m) by discharge class
_Q_BINS_RADIUS   = jnp.array([0.3, 3.0, 15.0, 30.0, 80.0])
_RADIUS_VALUES   = jnp.array([100., 150., 300., 600., 1000., 1500.])

# IS 10430:2000, Table 1 — freeboard (m) by discharge class
_Q_BINS_FREEBOARD = jnp.array([0.75, 1.5, 85.0])
_FREEBOARD_VALUES  = jnp.array([0.30, 0.50, 0.60, 0.75])

# IS 10430:2000 — velocity limits (m/s) for concrete lining
V_MAX_CONCRETE = 2.5
V_MIN          = 0.6

# IS 10430:2000, Table 2 — standard side slopes
SIDE_SLOPE_CONCRETE = 1.5   # H:V
SIDE_SLOPE_MASONRY  = 1.0

# IS 10430:2000, Cl 4.1 — Manning's n for concrete lining
MANNING_N_CONCRETE = 0.018


def _is_min_radius(Q: float) -> float:
    return float(_RADIUS_VALUES[jnp.digitize(Q, _Q_BINS_RADIUS)])


def _is_freeboard(Q: float) -> float:
    return float(_FREEBOARD_VALUES[jnp.digitize(Q, _Q_BINS_FREEBOARD)])


# ── hydraulics (pure Python, no JAX needed) ───────────────────────────────────

def _trap_hydraulics(B: float, D: float, m: float, n: float, S: float):
    """Manning's equation for a trapezoidal section."""
    A = (B + m * D) * D
    P = B + 2.0 * D * math.sqrt(1.0 + m ** 2)
    R = A / P
    V = (1.0 / n) * R ** (2.0 / 3.0) * math.sqrt(S)
    Q = A * V
    return Q, V, A, P


def _analytical_section(Q_target: float, S: float, n: float, m: float):
    """
    Minimum wetted perimeter trapezoidal section (Chow 1959).

    Classical result: R = D/2 for the hydraulically efficient section.
    Combined with Manning:  Q = (1/n)*(2√(1+m²)-m)*D² * (D/2)^(2/3) * √S
    → D^(8/3) = Q*n*2^(2/3) / (√S * (2√(1+m²)-m))
    → B = 2*D*(√(1+m²) - m)
    """
    coef = 2.0 * math.sqrt(1.0 + m ** 2) - m          # 2.106 for m=1.5
    D = (Q_target * n * 2.0 ** (2.0 / 3.0) / (math.sqrt(S) * coef)) ** (3.0 / 8.0)
    B = max(2.0 * D * (math.sqrt(1.0 + m ** 2) - m), 0.3)
    return B, D


def _newton_D_for_velocity(Q_target: float, V_fix: float, S: float, n: float, m: float,
                            tol: float = 1e-9, max_iter: int = 100) -> tuple[float, float]:
    """
    When velocity at the optimal section is outside IS limits, fix V = V_fix and
    solve Q = Q_target via Newton's method on D, keeping B = 2D(√(1+m²)-m).

    For a minimum-P section: A = coef*D², P = 2*coef/2*(…) → V depends only on R=D/2.
    So V_fix fixes D directly: D = (V_fix * n / √S)^(3/2) * 2.
    Then B is computed from Q_target.
    """
    # V = (1/n)*(D/2)^(2/3)*√S  → D = 2*(V*n/√S)^(3/2)
    D = 2.0 * (V_fix * n / math.sqrt(S)) ** 1.5
    # With D fixed, solve for B from Q = (B+m*D)*D*V_fix = Q_target
    B = Q_target / (D * V_fix) - m * D
    B = max(B, 0.3)
    return B, D


# ── optimiser ─────────────────────────────────────────────────────────────────

def optimise_canal(
    Q_target: float,
    S_long: float = 1 / 5000,
    n: float = MANNING_N_CONCRETE,
    lining: str = "concrete",
    out_path: str | None = None,
) -> dict:
    """
    Find minimum-cost trapezoidal canal section for Q_target (m³/s).

    Uses the closed-form analytical solution (Chow 1959 / Das 2000) for
    minimum wetted perimeter, with IS 10430 velocity compliance check.
    Returns a dict with all hydraulic and IS-code parameters.
    If out_path is given, also writes canal_params.json there.
    """
    m = SIDE_SLOPE_CONCRETE if lining == "concrete" else SIDE_SLOPE_MASONRY

    # Step 1: analytical minimum-wetted-perimeter section
    B, D = _analytical_section(Q_target, S_long, n, m)
    Q, V, A, P = _trap_hydraulics(B, D, m, n, S_long)

    # Step 2: IS 10430 velocity check — adjust if needed
    if V > V_MAX_CONCRETE:
        # Fix velocity at max; widen B to satisfy Q
        B, D = _newton_D_for_velocity(Q_target, V_MAX_CONCRETE, S_long, n, m)
        Q, V, A, P = _trap_hydraulics(B, D, m, n, S_long)
    elif V < V_MIN:
        # Fix velocity at min; compute B for Q
        B, D = _newton_D_for_velocity(Q_target, V_MIN, S_long, n, m)
        Q, V, A, P = _trap_hydraulics(B, D, m, n, S_long)

    fb     = _is_freeboard(Q)
    min_r  = _is_min_radius(Q)
    top_w  = B + 2.0 * m * (D + fb)

    result = {
        "Q_target_m3s":         round(Q_target, 4),
        "bed_width_m":          round(B, 4),
        "water_depth_m":        round(D, 4),
        "side_slope":           round(m, 4),
        "freeboard_m":          round(fb, 4),
        "total_depth_m":        round(D + fb, 4),
        "top_width_m":          round(top_w, 4),
        "flow_area_m2":         round(A, 4),
        "wetted_perimeter_m":   round(P, 4),
        "Q_calculated_m3s":     round(Q, 4),
        "velocity_ms":          round(V, 4),
        "min_curve_radius_m":   round(min_r, 1),
        "manning_n":            round(n, 4),
        "long_slope":           round(S_long, 6),
        "lining":               lining,
        # IS compliance flags
        "IS_velocity_ok":       V_MIN <= V <= V_MAX_CONCRETE,
        "IS_discharge_ok":      Q >= Q_target * 0.99,
        "IS_side_slope_ok":     1.0 <= m <= 3.0,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# ── IS validation report ──────────────────────────────────────────────────────

def validate_is(params: dict) -> dict:
    """Run IS code compliance checks on a params dict. Returns pass/fail per clause."""
    Q  = params["Q_calculated_m3s"]
    V  = params["velocity_ms"]
    D  = params["water_depth_m"]
    fb = params["freeboard_m"]
    S  = params["side_slope"]

    fb_req  = _is_freeboard(Q)
    r_req   = _is_min_radius(Q)

    return {
        "IS5968_min_radius_m":          r_req,
        "IS10430_freeboard_ok":         fb >= fb_req,
        "IS10430_freeboard_req_m":      fb_req,
        "IS10430_velocity_ok":          V_MIN <= V <= V_MAX_CONCRETE,
        "IS10430_velocity_ms":          V,
        "IS10430_side_slope_ok":        1.0 <= S <= 3.0,
        "IS10430_discharge_ok":         Q >= params["Q_target_m3s"] * 0.99,
    }


def print_report(params: dict):
    print("\n" + "=" * 50)
    print("  CANAL DESIGN REPORT  (IS 5968:1987 / IS 10430:2000)")
    print("=" * 50)
    print(f"  Q target     : {params['Q_target_m3s']:.2f} m³/s")
    print(f"  Q calculated : {params['Q_calculated_m3s']:.2f} m³/s")
    print(f"  Bed width B  : {params['bed_width_m']:.3f} m")
    print(f"  Water depth D: {params['water_depth_m']:.3f} m")
    print(f"  Side slope   : {params['side_slope']:.2f}:1 (H:V)")
    print(f"  Top width    : {params['top_width_m']:.3f} m")
    print(f"  Velocity     : {params['velocity_ms']:.3f} m/s  [0.6–2.5 m/s]")
    print(f"  Freeboard    : {params['freeboard_m']:.2f} m  (IS 10430, Table 1)")
    print(f"  Min radius   : {params['min_curve_radius_m']:.0f} m  (IS 5968, Table 1)")
    print(f"  Slope        : 1:{int(1 / params['long_slope'])}")
    print(f"  Manning n    : {params['manning_n']}")
    is_ok = all([params["IS_velocity_ok"], params["IS_discharge_ok"], params["IS_side_slope_ok"]])
    print(f"\n  IS Compliance: {'✓ PASS' if is_ok else '✗ FAIL'}")
    print("=" * 50)


# ── visualisation ────────────────────────────────────────────────────────────

def plot_canal_section(params: dict, out_path: str = "assets/canal_section.png"):
    """
    Labeled cross-section figure: trapezoidal canal with water, freeboard,
    and all IS-code dimensions annotated.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    B  = params["bed_width_m"]
    D  = params["water_depth_m"]
    m  = params["side_slope"]
    fb = params["freeboard_m"]
    Dt = D + fb          # total depth (water + freeboard)
    Tw = params["top_width_m"]
    V  = params["velocity_ms"]
    Q  = params["Q_calculated_m3s"]

    # x-coordinates of the trapezoid (symmetric about x=0)
    half_b  = B / 2
    x_bot_l, x_bot_r = -half_b, half_b
    x_top_l = -(half_b + m * Dt)
    x_top_r =  (half_b + m * Dt)
    x_wat_l = -(half_b + m * D)
    x_wat_r =  (half_b + m * D)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # concrete lining (full trapezoid)
    lining_x = [x_top_l, x_bot_l, x_bot_r, x_top_r]
    lining_y = [Dt,       0,        0,        Dt]
    ax.fill(lining_x, lining_y, color="#5a5a6a", alpha=0.8, zorder=1)
    ax.plot(lining_x + [lining_x[0]], lining_y + [lining_y[0]],
            color="#9090aa", lw=1.5, zorder=2)

    # water body
    water_x = [x_wat_l, x_bot_l, x_bot_r, x_wat_r]
    water_y = [D,        0,       0,        D]
    ax.fill(water_x, water_y, color="#1a6fa8", alpha=0.75, zorder=3, label="Water")

    # water surface line
    ax.plot([x_wat_l, x_wat_r], [D, D], color="#56c8ff", lw=2, ls="--", zorder=4)

    # freeboard zone (hatched)
    fb_x = [x_wat_l, x_top_l, x_top_r, x_wat_r]
    fb_y = [D,        Dt,       Dt,       D]
    ax.fill(fb_x, fb_y, color="#3a3a4a", alpha=0.6, zorder=2,
            hatch="///", edgecolor="#6a6a8a")

    # ── dimension annotations ──────────────────────────────────────────
    ann = dict(arrowprops=dict(arrowstyle="<->", color="#e0e0e0", lw=1.2),
               color="#e0e0e0", fontsize=9, ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="none"))

    # B — bed width
    ax.annotate("", xy=(half_b, -0.35), xytext=(-half_b, -0.35),
                arrowprops=dict(arrowstyle="<->", color="#f0c040", lw=1.5))
    ax.text(0, -0.55, f"B = {B:.2f} m  (bed width)", color="#f0c040",
            fontsize=9, ha="center", va="top")

    # D — water depth
    ax.annotate("", xy=(x_top_r + 0.4, D), xytext=(x_top_r + 0.4, 0),
                arrowprops=dict(arrowstyle="<->", color="#56c8ff", lw=1.5))
    ax.text(x_top_r + 0.9, D / 2, f"D = {D:.2f} m", color="#56c8ff",
            fontsize=9, ha="left", va="center")

    # freeboard
    ax.annotate("", xy=(x_top_r + 0.4, Dt), xytext=(x_top_r + 0.4, D),
                arrowprops=dict(arrowstyle="<->", color="#a0c0a0", lw=1.5))
    ax.text(x_top_r + 0.9, D + fb / 2, f"fb = {fb:.2f} m", color="#a0c0a0",
            fontsize=9, ha="left", va="center")

    # total depth
    ax.annotate("", xy=(x_top_r + 1.8, Dt), xytext=(x_top_r + 1.8, 0),
                arrowprops=dict(arrowstyle="<->", color="#d0a0d0", lw=1.5))
    ax.text(x_top_r + 2.3, Dt / 2, f"Dtotal = {Dt:.2f} m", color="#d0a0d0",
            fontsize=9, ha="left", va="center")

    # top width Tw
    ax.annotate("", xy=(x_top_r, Dt + 0.3), xytext=(x_top_l, Dt + 0.3),
                arrowprops=dict(arrowstyle="<->", color="#e0a040", lw=1.5))
    ax.text(0, Dt + 0.5, f"Top width = {Tw:.2f} m", color="#e0a040",
            fontsize=9, ha="center")

    # side slope label
    sl_x = (x_bot_r + x_top_r) / 2
    sl_y = Dt / 2
    ax.text(sl_x + 0.2, sl_y, f"m = {m:.1f} H:V", color="#c0c0c0",
            fontsize=8, ha="left", va="center", style="italic")

    # velocity arrow inside water
    ax.annotate("", xy=(half_b * 0.6, D * 0.45), xytext=(-half_b * 0.6, D * 0.45),
                arrowprops=dict(arrowstyle="-|>", color="#90e0ff", lw=1.8))
    ax.text(0, D * 0.28, f"V = {V:.2f} m/s", color="#90e0ff",
            fontsize=8.5, ha="center", va="top")

    # IS tags (top-left)
    is_ok = params.get("IS_velocity_ok") and params.get("IS_discharge_ok")
    ax.text(x_top_l, Dt + 0.85,
            f"IS 10430:2000  ({'✓ PASS' if is_ok else '✗ FAIL'})",
            color="#80ff80" if is_ok else "#ff8080", fontsize=9)
    ax.text(x_top_l, Dt + 0.60,
            f"IS 5968:1987   min radius = {params['min_curve_radius_m']:.0f} m",
            color="#c0c0c0", fontsize=8)

    ax.set_xlim(x_top_l - 3.2, x_top_r + 4.5)
    ax.set_ylim(-0.85, Dt + 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Width (m)", color="#a0a0a0", fontsize=9)
    ax.set_ylabel("Elevation (m)", color="#a0a0a0", fontsize=9)
    ax.tick_params(colors="#a0a0a0", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#3a3a4a")

    ax.set_title(
        f"Trapezoidal Canal Cross-Section\n"
        f"Q = {Q:.1f} m³/s  ·  n = {params['manning_n']}  ·  S = 1:{int(1/params['long_slope'])}  ·  {params['lining'].capitalize()} lining",
        color="#e0e0e0", fontsize=11, pad=12)

    handles = [
        mpatches.Patch(color="#1a6fa8", alpha=0.75, label="Water"),
        mpatches.Patch(color="#5a5a6a", alpha=0.8,  label="Concrete lining"),
        mpatches.Patch(color="#3a3a4a", alpha=0.6,  hatch="///",
                       edgecolor="#6a6a8a", label="Freeboard"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              facecolor="#1a1a2a", edgecolor="#3a3a4a", labelcolor="#e0e0e0")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Canal section figure → {out_path}")


def plot_design_chain(params: dict,
                      h_mean: float = 1.152, h_max: float = 2.124,
                      wet_area_m2: float = 39700,
                      out_path: str = "assets/design_chain.png"):
    """
    Three-panel figure showing the measurement → discharge → IS design chain:
      Panel 1: Flow depth distribution bar chart
      Panel 2: Manning's equation components (A, R, V, Q)
      Panel 3: Optimal B–D parameter space with the IS-compliant solution marked
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    B   = params["bed_width_m"]
    D   = params["water_depth_m"]
    m   = params["side_slope"]
    n   = params["manning_n"]
    S   = params["long_slope"]
    Q   = params["Q_calculated_m3s"]
    V   = params["velocity_ms"]
    fb  = params["freeboard_m"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#a0a0a0", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a4a")

    # ── Panel 1: flow depth distribution ─────────────────────────────
    ax = axes[0]
    # synthetic approximate lognormal distribution matching h_mean/h_max
    rng = np.random.default_rng(42)
    sigma = 0.6
    mu    = np.log(h_mean) - sigma**2 / 2
    h_vals = np.clip(rng.lognormal(mu, sigma, 80000), 0, h_max * 1.05)
    bins = np.linspace(0, h_max * 1.1, 40)
    counts, edges = np.histogram(h_vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts / counts.max(), width=centers[1] - centers[0],
           color="#1a6fa8", alpha=0.8, edgecolor="#56c8ff", lw=0.4)
    ax.axvline(h_mean, color="#f0c040", lw=1.8, ls="--",
               label=f"h_mean = {h_mean:.3f} m")
    ax.axvline(h_max,  color="#ff6060", lw=1.8, ls=":",
               label=f"h_max  = {h_max:.3f} m")
    ax.set_xlabel("Flow depth h (m)", color="#a0a0a0", fontsize=9)
    ax.set_ylabel("Relative frequency", color="#a0a0a0", fontsize=9)
    ax.set_title("Stage 6a — Flow Depth Distribution\nh(x,y) = Z_surface − Z_bed",
                 color="#e0e0e0", fontsize=9, pad=8)
    ax.legend(fontsize=8, facecolor="#1a1a2a", edgecolor="#3a3a4a", labelcolor="#e0e0e0")
    ax.text(0.97, 0.97,
            f"Wet area ≈ {wet_area_m2/1000:.0f}k m²",
            transform=ax.transAxes, color="#c0c0c0", fontsize=8,
            ha="right", va="top")

    # ── Panel 2: Manning components (bar chart of A, R, V, Q relative) ─
    ax = axes[1]
    # show the hydraulic parameters as a waterfall / breakdown
    A = params["flow_area_m2"]
    P = params["wetted_perimeter_m"]
    R = A / P

    labels  = ["A\n(m²)", "P\n(m)", "R=A/P\n(m)", "V\n(m/s)", "Q=A·V\n(m³/s)"]
    values  = [A,          P,         R,             V,          Q]
    colors  = ["#3a8fd0", "#7040c0", "#40b080", "#f0a020", "#e04040"]
    norm_v  = [v / max(values) for v in values]

    bars = ax.bar(labels, norm_v, color=colors, alpha=0.85, width=0.5, edgecolor="#ffffff22")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", color="#e0e0e0", fontsize=8.5, ha="center", va="bottom")

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Normalised value", color="#a0a0a0", fontsize=9)
    ax.set_title(
        "Stage 7 — Manning's Equation\n"
        r"Q = $\frac{1}{n}$·A·R$^{2/3}$·S$^{1/2}$",
        color="#e0e0e0", fontsize=9, pad=8)
    ax.text(0.5, 0.97,
            f"n={n}  S=1:{int(1/S)}  → Q={Q:.1f} m³/s",
            transform=ax.transAxes, color="#c0c0c0", fontsize=8,
            ha="center", va="top")

    # ── Panel 3: B–D parameter space ──────────────────────────────────
    ax = axes[2]
    B_range = np.linspace(0.3, 8.0, 200)
    # For each B, find D such that Q_target is achieved with given m,n,S
    # using hydraulically efficient section: D from closed-form, then scale B
    D_range = np.linspace(0.5, 8.0, 200)
    BB, DD = np.meshgrid(B_range, D_range)
    AA = (BB + m * DD) * DD
    PP = BB + 2.0 * DD * np.sqrt(1.0 + m**2)
    RR = AA / PP
    VV = (1.0 / n) * RR**(2.0/3.0) * np.sqrt(S)
    QQ = AA * VV

    # Q contour
    cs = ax.contourf(BB, DD, QQ,
                     levels=[0, 20, 40, 50, 60, 80, 120],
                     cmap="Blues", alpha=0.7)
    ax.contour(BB, DD, QQ, levels=[Q], colors="#f0c040", linewidths=1.5,
               linestyles="--")

    # velocity limits
    ax.contour(BB, DD, VV, levels=[V_MIN],         colors="#a0ff80", linewidths=1.2)
    ax.contour(BB, DD, VV, levels=[V_MAX_CONCRETE], colors="#ff8080", linewidths=1.2)

    # optimal point
    ax.scatter([B], [D], color="#ff4040", s=120, zorder=10,
               edgecolors="white", lw=1.5, label=f"Optimal  B={B:.2f}, D={D:.2f}")

    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label("Q (m³/s)", color="#a0a0a0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#a0a0a0", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#a0a0a0")

    ax.set_xlabel("Bed width B (m)", color="#a0a0a0", fontsize=9)
    ax.set_ylabel("Water depth D (m)", color="#a0a0a0", fontsize=9)
    ax.set_title(
        "Stage 7 — IS Design Space\n"
        f"Yellow dashed: Q={Q:.0f} m³/s  ·  Green: V_min  ·  Red: V_max",
        color="#e0e0e0", fontsize=9, pad=8)
    ax.legend(fontsize=8, facecolor="#1a1a2a", edgecolor="#3a3a4a",
              labelcolor="#e0e0e0", loc="upper right")

    plt.suptitle(
        "Flow Measurement → Discharge → IS-Code Canal Design",
        color="#e0e0e0", fontsize=12, y=1.01)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Design chain figure → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="JAX canal optimizer (IS 5968 / IS 10430)")
    p.add_argument("--Q",      type=float, default=50.0,   help="Design discharge m³/s")
    p.add_argument("--slope",  type=float, default=0.0002, help="Longitudinal slope (e.g. 0.0002 = 1:5000)")
    p.add_argument("--n",      type=float, default=MANNING_N_CONCRETE, help="Manning's roughness")
    p.add_argument("--lining", choices=["concrete", "masonry"], default="concrete")
    p.add_argument("--out",    default="canal_design/canal_params.json")
    args = p.parse_args()

    params = optimise_canal(Q_target=args.Q, S_long=args.slope, n=args.n,
                            lining=args.lining, out_path=args.out)
    print_report(params)
    print(f"\nSaved → {args.out}")
