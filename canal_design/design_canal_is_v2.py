import sys
import os
import math

# Set up FreeCAD paths
sys.path.append("/usr/lib/freecad/lib")
sys.path.append("/usr/share/freecad/Mod")

import FreeCAD
import Part
import Mesh

class ISCanalDesigner:
    """
    Automated Canal Designer following Indian Standard Codes:
    - IS 5968:1987 (Planning and Layout)
    - IS 10430:2000 (Lined Canal Design)
    """
    
    def __init__(self, Q, bed_slope, lining_type="concrete"):
        self.Q = Q # Discharge in cumecs
        self.S = bed_slope # Longitudinal slope
        self.lining_type = lining_type
        self.params = {}
        
    def calculate_is_parameters(self):
        print(f"--- Calculating Parameters for Q={self.Q} cumecs ---")
        
        # 1. MINIMUM CURVE RADIUS (Reference: IS 5968:1987, Table 1)
        # Clause 8.1: Radius depends on discharge capacity
        if self.Q < 0.3: self.params['min_radius'] = 100
        elif self.Q < 3.0: self.params['min_radius'] = 150
        elif self.Q < 15.0: self.params['min_radius'] = 300
        elif self.Q < 30.0: self.params['min_radius'] = 600
        elif self.Q < 80.0: self.params['min_radius'] = 1000
        else: self.params['min_radius'] = 1500
        print(f"[IS 5968] Min Radius: {self.params['min_radius']}m")

        # 2. FREEBOARD (Reference: IS 10430:2000, Table 1)
        # Mandatory safety height above Full Supply Level (FSL)
        if self.Q < 0.75: self.params['freeboard'] = 0.30
        elif self.Q < 1.5: self.params['freeboard'] = 0.50
        elif self.Q < 85.0: self.params['freeboard'] = 0.60
        else: self.params['freeboard'] = 0.75
        print(f"[IS 10430] Freeboard: {self.params['freeboard']}m")

        # 3. SIDE SLOPES (Reference: IS 10430:2000, Table 2)
        # Standard for Concrete Lining is 1.5:1 (Horizontal:Vertical)
        if self.lining_type == "concrete":
            self.params['side_slope'] = 1.5
        else:
            self.params['side_slope'] = 1.0 # Masonry standard
        print(f"[IS 10430] Side Slope: {self.params['side_slope']}:1")

        # 4. HYDRAULIC SECTION (Reference: Manning's Formula as per IS 10430 Clause 4.1)
        # V = (1/n) * R^(2/3) * S^(1/2)
        n = 0.018 # Manning's n for concrete lining
        b_d_ratio = 4.0 # Initial assumption for Bed Width / Depth
        
        s_val = self.params['side_slope']
        def solve_depth(d_test):
            b_test = b_d_ratio * d_test
            area = (b_test + s_val * d_test) * d_test
            perimeter = b_test + 2 * d_test * math.sqrt(1 + s_val**2)
            rh = area / perimeter
            velocity = (1.0 / n) * (rh**(2.0/3.0)) * (self.S**0.5)
            return area * velocity

        d = 1.0
        for _ in range(50):
            q_calc = solve_depth(d)
            if abs(q_calc - self.Q) < 0.01: break
            d = d * (self.Q / q_calc)**0.4
            
        self.params['depth'] = round(d, 3)
        self.params['bed_width'] = round(b_d_ratio * d, 3)
        self.params['velocity'] = round(self.Q / ((self.params['bed_width'] + s_val*d)*d), 3)
        
        # 5. VELOCITY CHECK (Reference: IS 10430, Clause 4.2)
        # Max permissible velocity for concrete lining is 2.5 m/s
        if self.params['velocity'] > 2.5:
            print(f"⚠️ WARNING: Velocity {self.params['velocity']} exceeds IS limit of 2.5m/s")
        else:
            print(f"[IS 10430] Velocity Validated: {self.params['velocity']} m/s")

    def generate_3d_model(self, ips, output_path):
        """Generates the FreeCAD geometry using calculated IS parameters."""
        print("Generating 3D Geometry in FreeCAD...")
        doc = FreeCAD.newDocument("IS_Canal_V2")
        
        # Logic for Path with Curves (IS 5968 Compliance)
        r = self.params['min_radius']
        edges = []
        prev_end = ips[0]
        
        for i in range(1, len(ips) - 1):
            p1, p2, p3 = ips[i-1], ips[i], ips[i+1]
            
            # Vector math for tangents
            v1 = (p1 - p2).normalize()
            v2 = (p3 - p2).normalize()
            angle = math.acos(v1.dot(v2))
            t_dist = r * math.tan((math.pi - angle) / 2.0)
            
            t1, t2 = p2 + v1 * t_dist, p2 + v2 * t_dist
            
            if (t1 - prev_end).Length > 1e-3:
                edges.append(Part.LineSegment(prev_end, t1).toShape())
            
            # Arc for the curve
            bisector = (v1 + v2).normalize()
            dist_to_apex = r / math.sin(angle/2.0) - r
            apex = p2 + bisector * dist_to_apex
            arc = Part.Arc(t1, apex, t2)
            edges.append(arc.toShape())
            prev_end = t2
            
        if (ips[-1] - prev_end).Length > 1e-3:
            edges.append(Part.LineSegment(prev_end, ips[-1]).toShape())
            
        path_wire = Part.Wire(edges)
        
        # Cross-Section Assembly
        B = self.params['bed_width']
        total_D = self.params['depth'] + self.params['freeboard']
        S = self.params['side_slope']
        
        # Setup Section Plane (YZ plane relative to path start)
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
        
        # Sweep
        sweep = Part.Wire(path_wire.Edges).makePipeShell([section], True, False)
        canal_obj = doc.addObject("Part::Feature", "IS_Canal_Body")
        canal_obj.Shape = sweep
        
        doc.recompute()
        
        # Export
        base_name = os.path.splitext(output_path)[0]
        Part.export([canal_obj], f"{base_name}.step")
        Mesh.export([canal_obj], f"{base_name}.obj")
        print(f"Models exported to {base_name}.step/obj")

if __name__ == "__main__":
    # USER INPUTS
    DESIGN_Q = 50.0 # Cumecs
    SLOPE = 1/5000
    
    # ALIGNMENT POINTS (IPs)
    ALIGNMENT = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(1500, 0, 0), # Long reach to accommodate 1000m radius
        FreeCAD.Vector(3000, 1500, 0),
        FreeCAD.Vector(5000, 1500, 0)
    ]
    
    designer = ISCanalDesigner(Q=DESIGN_Q, bed_slope=SLOPE)
    designer.calculate_is_parameters()
    designer.generate_3d_model(ALIGNMENT, "/home/kaiser/gemini_project2/is_canal_v2.glb")
