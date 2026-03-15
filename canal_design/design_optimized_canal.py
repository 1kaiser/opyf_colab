import sys
import os
import math
import json

# Set up FreeCAD and JAX environment paths
sys.path.append("/usr/lib/freecad/lib")
sys.path.append("/usr/share/freecad/Mod")
sys.path.append("/home/kaiser/.miniforge/envs/num_python/lib/python3.12/site-packages")

# Now import modules that depend on these paths
import FreeCAD
import Part
import Mesh
import jax
import jax.numpy as jnp

from jax_canal_optimizer import JAXCanalOptimizer

class OptimizedISCanalDesigner:
    def __init__(self, opt_params):
        self.params = opt_params
        
    def generate_3d_model(self, ips, output_path):
        print("Generating Optimized 3D Geometry in FreeCAD...")
        doc = FreeCAD.newDocument("Optimized_IS_Canal")
        
        r = self.params['min_radius']
        edges = []
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
        
        B = self.params['bed_width']
        total_D = self.params['total_depth']
        S = self.params['side_slope']
        
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

# EXECUTION LOGIC
print("Starting Optimized Canal Design...")
# 1. RUN JAX OPTIMIZATION
Q_TARGET = 50.0
SLOPE = 1/5000
solution = JAXCanalOptimizer.run_optimization(Q_target=Q_TARGET, S_long=SLOPE)

# 2. GENERATE CAD
ALIGNMENT = [
    FreeCAD.Vector(0, 0, 0),
    FreeCAD.Vector(1500, 0, 0),
    FreeCAD.Vector(3000, 1500, 0),
    FreeCAD.Vector(5000, 1500, 0)
]

designer = OptimizedISCanalDesigner(solution)
designer.generate_3d_model(ALIGNMENT, "/home/kaiser/gemini_project2/is_canal_optimized.obj")

print("Done. Exiting FreeCAD...")
sys.exit(0)
