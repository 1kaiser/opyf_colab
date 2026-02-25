import sys
import os
import math
import json
import argparse

# --- FREECAD CONFIGURATION ---
# Ensure FreeCAD libraries are accessible
sys.path.append("/usr/lib/freecad/lib")
sys.path.append("/usr/share/freecad/Mod")

import FreeCAD
import Part
import Mesh

def get_tangent_points(p1, p2, p3, r):
    """Calculates tangent points for a smooth circular transition."""
    v1 = (p1 - p2).normalize()
    v2 = (p3 - p2).normalize()
    angle = math.acos(v1.dot(v2))
    t_dist = r * math.tan((math.pi - angle) / 2.0)
    t1 = p2 + v1 * t_dist
    t2 = p2 + v2 * t_dist
    bisector = (v1 + v2).normalize()
    apex = p2 + bisector * (r / math.sin(angle/2.0) - r)
    return t1, apex, t2

def generate_assets(params_path, output_dir):
    # 1. Load Parameters
    with open(params_path, "r") as f:
        p = json.load(f)
    
    print(f"--- Generating Canal Assets ---")
    print(f"B={p['bed_width']:.2f}m, D={p['water_depth']:.2f}m, Slope={p['side_slope']}:1")
    
    doc = FreeCAD.newDocument("Canal_Engineering_Assets")
    
    # 2. Alignment Logic (IS 5968)
    # Using a standard 1km test alignment
    ips = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(500, 0, 0),
        FreeCAD.Vector(1000, 500, 0)
    ]
    r = p.get('min_radius', 300.0)
    
    t1, apex, t2 = get_tangent_points(ips[0], ips[1], ips[2], r)
    
    e1 = Part.LineSegment(ips[0], t1).toShape()
    e2 = Part.Arc(t1, apex, t2).toShape()
    e3 = Part.LineSegment(t2, ips[2]).toShape()
    path_wire = Part.Wire([e1, e2, e3])
    
    # 3. Cross Section (IS 10430)
    B, D, FB, S = p['bed_width'], p['water_depth'], p['freeboard'], p['side_slope']
    total_D = D + FB
    
    # section plane logic
    sec_pts = [
        FreeCAD.Vector(0, -B/2.0, 0),
        FreeCAD.Vector(0, B/2.0, 0),
        FreeCAD.Vector(0, B/2.0 + S*total_D, total_D),
        FreeCAD.Vector(0, -B/2.0 - S*total_D, total_D),
        FreeCAD.Vector(0, -B/2.0, 0)
    ]
    section = Part.makePolygon(sec_pts)
    
    # 4. Create 3D Model
    print("Sweeping 3D Canal Body...")
    sweep = path_wire.makePipeShell([section], True, False)
    canal_3d = doc.addObject("Part::Feature", "Canal_3D")
    canal_3d.Shape = sweep
    
    # 5. Create 2D Representations
    print("Generating 2D Views...")
    # Section View
    sec_2d = doc.addObject("Part::Feature", "Section_2D")
    sec_2d.Shape = section
    sec_2d.Placement.Base = FreeCAD.Vector(-200, 0, 0)
    
    # Long-Section (Elevation)
    length = 1000.0
    s_long = p.get('long_slope', 0.0002)
    elevation_shape = Part.LineSegment(FreeCAD.Vector(0, -200, 0), 
                                       FreeCAD.Vector(length, -200, -length * s_long)).toShape()
    elev_2d = doc.addObject("Part::Feature", "Elevation_2D")
    elev_2d.Shape = elevation_shape

    doc.recompute()
    
    # 6. Exports
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D
    Part.export([canal_3d], os.path.join(output_dir, "canal_model.step"))
    Mesh.export([canal_3d], os.path.join(output_dir, "canal_model.obj"))
    
    # 2D Drawings
    Part.export([sec_2d, elev_2d, path_wire], os.path.join(output_dir, "canal_drawings.step"))
    
    print(f"--- SUCCESS ---")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FreeCAD Canal Assets")
    parser.add_argument("--params", default="canal_params.json", help="Path to JSON params")
    parser.add_argument("--output", default="output/assets", help="Output directory")
    args = parser.parse_args()
    
    generate_assets(args.params, args.output)
