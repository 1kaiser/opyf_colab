"""
modules/canal_optimizer.py
===========================
JAX-based canal hydraulic optimizer (IS 5968:1987 + IS 10430:2000).

Process
-------
Q_target  ─▶  gradient descent on Manning's equation
          ─▶  IS code lookup (freeboard, min curve radius)
          ─▶  velocity and slope compliance check
          ─▶  canal_params dict / JSON

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
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

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


# ── hydraulics ────────────────────────────────────────────────────────────────

def _hydraulics(params, constants):
    """Manning's equation for a trapezoidal section. Pure JAX (differentiable)."""
    B, D, S_side = params
    n, S_long = constants
    area      = (B + S_side * D) * D
    perimeter = B + 2.0 * D * jnp.sqrt(1.0 + S_side ** 2)
    R_h       = area / perimeter
    velocity  = (1.0 / n) * R_h ** (2.0 / 3.0) * jnp.sqrt(S_long)
    discharge = area * velocity
    return discharge, velocity, area, perimeter


# ── objective ─────────────────────────────────────────────────────────────────

def _objective(params, Q_target, constants):
    B, D, S_side = params
    Q, V, A, P = _hydraulics(params, constants)

    cost = A + P * 0.1                                           # min excavation + lining

    # discharge must meet target (hard constraint via large penalty)
    pen_Q    = 5000.0 * jnp.maximum(0.0, Q_target - Q) ** 2
    # velocity limits (IS 10430, Cl 4.2)
    pen_Vmax = 5000.0 * jnp.maximum(0.0, V - V_MAX_CONCRETE) ** 2
    pen_Vmin = 2000.0 * jnp.maximum(0.0, V_MIN - V) ** 2
    # standard side slope (IS 10430, Table 2) — soft regularisation
    pen_side = 200.0  * (S_side - SIDE_SLOPE_CONCRETE) ** 2

    return cost + pen_Q + pen_Vmax + pen_Vmin + pen_side


# ── optimiser ─────────────────────────────────────────────────────────────────

def optimise_canal(
    Q_target: float,
    S_long: float = 1 / 5000,
    n: float = MANNING_N_CONCRETE,
    lining: str = "concrete",
    iters: int = 500,
    lr: float = 0.005,
    out_path: str | None = None,
) -> dict:
    """
    Find minimum-cost trapezoidal canal section for Q_target (m³/s).

    Returns a dict with all hydraulic and IS-code parameters.
    If out_path is given, also writes canal_params.json there.
    """
    side_slope_init = SIDE_SLOPE_CONCRETE if lining == "concrete" else SIDE_SLOPE_MASONRY
    params    = jnp.array([max(Q_target * 0.5, 1.0), 1.5, side_slope_init])
    constants = (n, S_long)

    grad_fn = jax.jit(jax.grad(_objective))

    B_min, D_min, S_min = 0.5,  0.2, 1.0
    B_max, D_max, S_max = 200., 15., 3.0

    for _ in range(iters):
        g      = grad_fn(params, Q_target, constants)
        params = params - lr * g
        params = jnp.clip(params,
                          jnp.array([B_min, D_min, S_min]),
                          jnp.array([B_max, D_max, S_max]))

    B, D, S_side = (float(x) for x in params)
    Q, V, A, P   = (_hydraulics(params, constants))
    Q, V, A, P   = float(Q), float(V), float(A), float(P)

    fb     = _is_freeboard(Q)
    min_r  = _is_min_radius(Q)
    top_w  = B + 2.0 * S_side * (D + fb)

    result = {
        "Q_target_m3s":         round(Q_target, 4),
        "bed_width_m":          round(B, 4),
        "water_depth_m":        round(D, 4),
        "side_slope":           round(S_side, 4),
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
        "IS_side_slope_ok":     1.0 <= S_side <= 3.0,
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
