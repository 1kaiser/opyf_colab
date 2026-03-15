import sys
import os
import json
import math

# Set up FreeCAD paths
sys.path.append("/usr/lib/freecad/lib")
sys.path.append("/usr/share/freecad/Mod")

import FreeCAD
import Part
import Mesh

def generate_reach():
    with open("dxf_data.json", 'r') as f:
        layer_data = json.load(f)
    
    doc = FreeCAD.newDocument("Reach12_Reconstruction")
    
    # 1. Find Reference Point (Shift to Origin)
    ref_x, ref_y = None, None
    for layer, entities in layer_data.items():
        for points in entities:
            if points and len(points[0]) >= 2:
                ref_x, ref_y = points[0][0], points[0][1]
                break
        if ref_x is not None: break
    
    if ref_x is None:
        print("No valid geometry found.")
        return

    print(f"Ref Point: ({ref_x}, {ref_y}). Shifting all points...")

    # Colors for layers
    colors = {
        "ground level": (0.6, 0.3, 0.1), # Brown
        "water level": (0.0, 0.5, 1.0),  # Blue
        "stone pitching": (0.5, 0.5, 0.5), # Gray
        "measurementa": (1.0, 1.0, 0.0)  # Yellow
    }

    objs = []
    for layer, entities in layer_data.items():
        if layer not in colors: continue # Only process main layers for now
        
        print(f"Processing Layer: {layer} ({len(entities)} entities)")
        
        edges = []
        for points in entities:
            if len(points) < 2: continue
            
            # Create segments
            seg_pts = []
            for p in points:
                if len(p) < 2: continue
                # Shift and treat Y as Z for a vertical profile view in 3D
                seg_pts.append(FreeCAD.Vector(p[0] - ref_x, 0, p[1] - ref_y))
            
            if len(seg_pts) >= 2:
                for j in range(len(seg_pts) - 1):
                    edges.append(Part.LineSegment(seg_pts[j], seg_pts[j+1]).toShape())
        
        if edges:
            wire = Part.Wire(edges)
            compound = doc.addObject("Part::Feature", layer.replace(" ", "_"))
            compound.Shape = wire
            # FreeCAD colors are 0-1 range
            # Note: cmd mode doesn't support ViewObject easily but we can try
            objs.append(compound)

    doc.recompute()
    
    output_obj = "/home/kaiser/gemini_project2/reach12_model.obj"
    print(f"Exporting to {output_obj}...")
    Mesh.export(objs, output_obj)
    print("Export complete.")

if __name__ == "__main__":
    generate_reach()
    sys.exit(0)
