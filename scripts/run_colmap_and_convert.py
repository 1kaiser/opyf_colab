import pycolmap
from pathlib import Path
import os
import open3d as o3d
import numpy as np

def run_reconstruction(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = output_dir / "database.db"
    if database_path.exists():
        database_path.unlink()

    # 1. Feature extraction
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)

    # 2. Matching
    print("Matching features...")
    pycolmap.match_exhaustive(database_path)

    # 3. Mapping
    print("Running incremental mapping...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

    if not reconstructions:
        print("Reconstruction failed.")
        return None

    # Use the first reconstruction
    recon = reconstructions[0]
    ply_path = output_dir / "colmap_points.ply"
    recon.export_PLY(ply_path)
    print(f"Points exported to {ply_path}")
    
    return ply_path

def convert_to_glb(ply_path, output_glb):
    print(f"Converting {ply_path} to GLB...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)

    print("Generating mesh via Poisson reconstruction...")
    # Depth 9 is a good balance between detail and performance
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # Clean up the mesh: remove low density regions
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(str(output_glb), mesh)
    print(f"Mesh saved to {output_glb}")

if __name__ == "__main__":
    IMAGE_DIR = "/home/kaiser/gemini_project2/reconstruction_data/frames"
    OUTPUT_DIR = "/home/kaiser/gemini_project2/reconstruction_data/output/colmap_reconstruction"
    GLB_OUTPUT = "/home/kaiser/gemini_project2/reconstruction_data/output/colmap_mesh.glb"
    
    ply_file = run_reconstruction(IMAGE_DIR, OUTPUT_DIR)
    if ply_file:
        convert_to_glb(ply_file, GLB_OUTPUT)
    else:
        print("Failed to produce PLY file from COLMAP.")
