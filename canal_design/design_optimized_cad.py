import sys
import os
import math
import json

# Logic to generate FreeCAD model from pre-calculated parameters
def generate_cad_from_json(json_path, output_path):
    # Set up FreeCAD paths
    sys.path.append("/usr/lib/freecad/lib")
    sys.path.append("/usr/share/freecad/Mod")
    
    import FreeCAD
    import Part
    import Mesh

    with open(json_path, 'r') as f:
        params = json.load(f)
    
    print("Generating Optimized 3D Geometry in FreeCAD from pre-calculated params...")
    doc = FreeCAD.newDocument("Optimized_IS_Canal")
    
    r = params['min_radius']
    edges = []
    
    # Define IPs
    ips = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(1500, 0, 0),
        FreeCAD.Vector(3000, 1500, 0),
        FreeCAD.Vector(5000, 1500, 0)
    ]
    
    prev_end = ips[0]
    for i in range(1, len(ips) - 1):
        p1, p2, p3 = ips[i-1], ips[i], ips[i+1]
        v1 = (p1 - p2).normalize()
        v2 = (p3 - p2).normalize()
        angle = math.acos(v1.dot(v2))
        t_dist = r * math.tan((math.pi - angle) / 2.0)
        t1, t2 = p2 + v1 * t_dist, p2 + v2 * t_dist
        
        if (t1 - prev_end).Length > 1e-3:
            edges.append(Part.LineSegment(prev_end, t1).toShape())
        
        bisector = (v1 + v2).normalize()
        dist_to_apex = r / math.sin(angle/2.0) - r
        apex = p2 + bisector * dist_to_apex
        arc = Part.Arc(t1, apex, t2)
        edges.append(arc.toShape())
        prev_end = t2
        
    if (ips[-1] - prev_end).Length > 1e-3:
        edges.append(Part.LineSegment(prev_end, ips[-1]).toShape())
        
    path_wire = Part.Wire(edges)
    
    B = params['bed_width']
    total_D = params['total_depth']
    S = params['side_slope']
    
    start_edge = edges[0]
    normal = start_edge.tangentAt(start_edge.FirstParameter)
    up = FreeCAD.Vector(0, 0, 1)
    right = normal.cross(up).normalize()
    
    def to_global(lx, ly):
        return ips[0] + right * lx + up * ly

    sec_pts = [
        to_global(-B/2.0, 0),
        to_global(B/2.0, 0),
        to_global(B/2.0 + S*total_D, total_D),
        to_global(-B/2.0 - S*total_D, total_D),
        to_global(-B/2.0, 0)
    ]
    section = Part.makePolygon(sec_pts)
    sweep = Part.Wire(path_wire.Edges).makePipeShell([section], True, False)
    canal_obj = doc.addObject("Part::Feature", "Optimized_Canal")
    canal_obj.Shape = sweep
    doc.recompute()
    
    base_name = os.path.splitext(output_path)[0]
    Part.export([canal_obj], f"{base_name}.step")
    Mesh.export([canal_obj], f"{base_name}.obj")
    print(f"Optimized models exported successfully.")

if __name__ == "__main__":
    # If run via freecadcmd, it won't have sys.argv[1] correctly usually
    # So we look for a fixed file name
    json_input = "canal_params.json"
    output_obj = "/home/kaiser/gemini_project2/is_canal_optimized.obj"
    
    if os.path.exists(json_input):
        generate_cad_from_json(json_input, output_obj)
        sys.exit(0)
    else:
        print("Error: canal_params.json not found.")
        sys.exit(1)
