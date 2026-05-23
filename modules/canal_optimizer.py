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
