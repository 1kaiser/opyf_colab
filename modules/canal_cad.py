"""
modules/canal_cad.py
=====================
FreeCAD-based canal CAD generator (IS 5968 / IS 10430).

Covers
------
- Trapezoidal section sweep along a 1 km IS-aligned path
- Compound / multi-stage section (berm + upper stage)
- 2D cross-section and long-section elevation views
- Reach reconstruction from DXF survey data (dxf_data.json)

Public API
----------
    generate_canal_cad(params, output_dir, reach_json=None)
    generate_reach_model(dxf_json_path, output_dir)

CLI
---
    freecadcmd modules/canal_cad.py --params canal_design/canal_params.json --output output/canal
    freecadcmd modules/canal_cad.py --reach  data/brague/dxf_data.json      --output output/reach
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# FreeCAD paths — conda / system / Flatpak in order of preference
for _p in ["/usr/lib/freecad/lib", "/usr/share/freecad/Mod",
           "/app/lib/freecad/lib", "/app/share/freecad/Mod"]:
    if _p not in sys.path:
        sys.path.append(_p)

try:
    import FreeCAD
    import Part
    import Mesh
    _FREECAD_OK = True
except ImportError:
    _FREECAD_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_freecad():
    if not _FREECAD_OK:
        raise RuntimeError(
            "FreeCAD not found. Run with `freecadcmd` or set FREECAD library paths.")


def _tangent_arc(p1, p2, p3, r):
    """Compute tangent points and arc apex for a corner bend of radius r."""
    v1 = (p1 - p2).normalize()
    v2 = (p3 - p2).normalize()
    angle  = math.acos(max(-1.0, min(1.0, v1.dot(v2))))
    t_dist = r * math.tan((math.pi - angle) / 2.0)
    t1 = p2 + v1 * t_dist
    t2 = p2 + v2 * t_dist
    bis  = (v1 + v2).normalize()
    apex = p2 + bis * (r / math.sin(angle / 2.0) - r)
    return t1, apex, t2


# ── trapezoidal section profile ───────────────────────────────────────────────

def _trap_section(B, D, S_side, freeboard, compound=False,
                  berm_w=4.0, D_upper=1.0, S_upper=2.0):
    """
    Build a trapezoidal (or compound) cross-section Wire.

    For compound sections an upper berm + secondary slope is added above the
    main section freeboard level (design_combined_canal logic).
    """
    FB    = freeboard
    total = D + FB

    if compound:
        pts = [
            FreeCAD.Vector(-B/2,  0, 0),
            FreeCAD.Vector( B/2,  0, 0),
            FreeCAD.Vector( B/2 + S_side*total, total, 0),
            FreeCAD.Vector( B/2 + S_side*total + berm_w, total, 0),
            FreeCAD.Vector( B/2 + S_side*total + berm_w + S_upper*D_upper, total+D_upper, 0),
            FreeCAD.Vector(-B/2 - S_side*total - berm_w - S_upper*D_upper, total+D_upper, 0),
            FreeCAD.Vector(-B/2 - S_side*total - berm_w, total, 0),
            FreeCAD.Vector(-B/2 - S_side*total, total, 0),
            FreeCAD.Vector(-B/2, 0, 0),
        ]
    else:
        pts = [
            FreeCAD.Vector(-B/2, 0, 0),
            FreeCAD.Vector( B/2, 0, 0),
            FreeCAD.Vector( B/2 + S_side*total, total, 0),
            FreeCAD.Vector(-B/2 - S_side*total, total, 0),
            FreeCAD.Vector(-B/2, 0, 0),
        ]

    return Part.makePolygon(pts)


# ── IS-aligned path ───────────────────────────────────────────────────────────

def _is_path(min_radius: float, length: float = 1000.0):
    """
    1 km alignment: straight → curve (IS 5968 radius) → straight.
    Inflection at 500 m, 45° turn.
    """
    ip0 = FreeCAD.Vector(0,   0,   0)
    ip1 = FreeCAD.Vector(500, 0,   0)
    ip2 = FreeCAD.Vector(length, 500, 0)

    t1, apex, t2 = _tangent_arc(ip0, ip1, ip2, min_radius)

    e1 = Part.LineSegment(ip0, t1).toShape()
    e2 = Part.Arc(t1, apex, t2).toShape()
    e3 = Part.LineSegment(t2, ip2).toShape()
    return Part.Wire([e1, e2, e3])


# ── main CAD generator ────────────────────────────────────────────────────────

def generate_canal_cad(
    params: dict,
    output_dir: str,
    compound: bool = False,
    berm_w: float = 4.0,
    D_upper: float = 1.0,
    S_upper: float = 2.0,
):
    """
    Generate 3D canal body + 2D section + elevation view and export to output_dir.

    Outputs
    -------
    canal_model.obj       3D sweep (Mesh)
    canal_model.step      3D sweep (STEP solid)
    canal_section.step    2D cross-section + elevation STEP
    """
    _require_freecad()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    B      = params["bed_width_m"]
    D      = params["water_depth_m"]
    S      = params["side_slope"]
    FB     = params["freeboard_m"]
    R      = params.get("min_curve_radius_m", 300.0)
    S_long = params.get("long_slope", 0.0002)

    label = "Compound_Canal" if compound else "Canal"
    doc   = FreeCAD.newDocument(label)

    # Section profile
    section = _trap_section(B, D, S, FB, compound=compound,
                            berm_w=berm_w, D_upper=D_upper, S_upper=S_upper)
    # Alignment
    path = _is_path(R)

    # 3D sweep
    print("  Sweeping 3D canal body …")
    sweep    = path.makePipeShell([section], True, False)
    body_obj = doc.addObject("Part::Feature", f"{label}_3D")
    body_obj.Shape = sweep

    # 2D cross-section (shifted left of origin)
    sec_2d       = doc.addObject("Part::Feature", "CrossSection_2D")
    sec_2d.Shape = section
    sec_2d.Placement.Base = FreeCAD.Vector(-B - 50, 0, 0)

    # 2D long-section elevation line
    elev_shape = Part.LineSegment(
        FreeCAD.Vector(0,    -100, 0),
        FreeCAD.Vector(1000, -100, -1000 * S_long)
    ).toShape()
    elev_obj       = doc.addObject("Part::Feature", "Elevation_2D")
    elev_obj.Shape = elev_shape

    doc.recompute()

    obj_path  = os.path.join(output_dir, "canal_model.obj")
    step_path = os.path.join(output_dir, "canal_model.step")
    sec_path  = os.path.join(output_dir, "canal_section.step")

    Mesh.export([body_obj], obj_path)
    Part.export([body_obj], step_path)
    Part.export([sec_2d, elev_obj], sec_path)

    print(f"  3D model  → {obj_path}")
    print(f"  STEP      → {step_path}")
    print(f"  Section   → {sec_path}")


# ── survey reach reconstruction ───────────────────────────────────────────────

def generate_reach_model(dxf_json_path: str, output_dir: str):
    """
    Reconstruct existing channel reach from DXF survey data (JSON format).

    Input JSON: {layer_name: [[[x, y], ...], ...], ...}
    Layers: "ground level", "water level", "stone pitching", "measurementa"

    Output: reach_model.obj (Mesh) + reach_model.step (Part)
    """
    _require_freecad()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(dxf_json_path) as f:
        layer_data = json.load(f)

    # Find origin reference point
    ref_x = ref_y = None
    for entities in layer_data.values():
        for pts in entities:
            if pts and len(pts[0]) >= 2:
                ref_x, ref_y = pts[0][0], pts[0][1]
                break
        if ref_x is not None:
            break

    if ref_x is None:
        raise ValueError("No valid geometry in DXF JSON")

    print(f"  Reference origin: ({ref_x:.3f}, {ref_y:.3f})")

    doc  = FreeCAD.newDocument("Reach_Survey")
    objs = []

    layer_colors = {
        "ground level":   (0.6, 0.3, 0.1),
        "water level":    (0.0, 0.5, 1.0),
        "stone pitching": (0.5, 0.5, 0.5),
        "measurementa":   (1.0, 1.0, 0.0),
    }

    for layer, entities in layer_data.items():
        if layer not in layer_colors:
            continue
        print(f"  Layer '{layer}': {len(entities)} entities")
        edges = []
        for seg_pts_raw in entities:
            pts = []
            for p in seg_pts_raw:
                if len(p) >= 2:
                    pts.append(FreeCAD.Vector(p[0] - ref_x, 0, p[1] - ref_y))
            for j in range(len(pts) - 1):
                edges.append(Part.LineSegment(pts[j], pts[j + 1]).toShape())

        if not edges:
            continue
        wire    = Part.Wire(edges)
        obj     = doc.addObject("Part::Feature", layer.replace(" ", "_"))
        obj.Shape = wire
        objs.append(obj)

    doc.recompute()

    obj_path  = os.path.join(output_dir, "reach_model.obj")
    step_path = os.path.join(output_dir, "reach_model.step")
    Mesh.export(objs, obj_path)
    Part.export(objs, step_path)
    print(f"  Reach model → {obj_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FreeCAD canal CAD generator")
    p.add_argument("--params",   help="canal_params.json path")
    p.add_argument("--reach",    help="dxf_data.json path for survey reach reconstruction")
    p.add_argument("--output",   default="output/canal", help="Output directory")
    p.add_argument("--compound", action="store_true",    help="Use compound (multi-stage) section")
    args = p.parse_args()

    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        print(f"Generating canal CAD from {args.params} …")
        generate_canal_cad(params, args.output, compound=args.compound)

    if args.reach:
        print(f"Generating reach model from {args.reach} …")
        generate_reach_model(args.reach, args.output)

    if not args.params and not args.reach:
        p.print_help()
        sys.exit(1)

    sys.exit(0)
