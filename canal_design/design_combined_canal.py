import sys
import os
import math

# Set up FreeCAD paths
sys.path.append("/usr/lib/freecad/lib")
sys.path.append("/usr/share/freecad/Mod")

import FreeCAD
import Part
import Mesh

print("🚀 Starting Combined Section Canal Design...")
doc = FreeCAD.newDocument("Combined_Canal")

# --- DIMENSIONS FROM TABLE ---
# BASE SECTION
B_base = 100.0
D_base = 3.10
S_base = 2.0  # 2:1

# UPPER SECTION
Berm = 4.0
D_upper = 1.00
S_upper = 2.0  # 2:1

# --- GEOMETRIC PROFILE (Cross-Section) ---
pts = [
    FreeCAD.Vector(-B_base/2.0, 0, 0),
    FreeCAD.Vector(B_base/2.0, 0, 0),
    FreeCAD.Vector(B_base/2.0 + S_base*D_base, D_base, 0),
    FreeCAD.Vector(B_base/2.0 + S_base*D_base + Berm, D_base, 0),
    FreeCAD.Vector(B_base/2.0 + S_base*D_base + Berm + S_upper*D_upper, D_base + D_upper, 0),
    FreeCAD.Vector(-(B_base/2.0 + S_base*D_base + Berm + S_upper*D_upper), D_base + D_upper, 0),
    FreeCAD.Vector(-(B_base/2.0 + S_base*D_base + Berm), D_base, 0),
    FreeCAD.Vector(-(B_base/2.0 + S_base*D_base), D_base, 0),
    FreeCAD.Vector(-B_base/2.0, 0, 0)
]

section = Part.makePolygon(pts)
print(f"✅ Profile created. Top Width: {2 * (B_base/2.0 + S_base*D_base + Berm + S_upper*D_upper)}m")

# --- PATH (1km Reach) ---
path_pts = [
    FreeCAD.Vector(0, 0, 0),
    FreeCAD.Vector(0, 0, 1000)
]
path = Part.makePolygon(path_pts)

# --- SWEEP ---
print("🛠️ Sweeping section along 1000m path...")
canal_shell = Part.Wire(path.Edges).makePipeShell([section], True, False)
canal_obj = doc.addObject("Part::Feature", "Combined_Canal_Body")
canal_obj.Shape = canal_shell

doc.recompute()

# --- EXPORT ---
output_path = "/home/kaiser/gemini_project2/combined_canal_section.obj"
print(f"📦 Exporting to {output_path} ...")
Mesh.export([canal_obj], output_path)
print("✅ Export complete. Exiting...")

sys.exit(0)
