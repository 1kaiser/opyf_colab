import pycolmap
from pathlib import Path
import os

def run_colmap_reconstruction(image_dir, output_dir):
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
    # incremental_mapping returns a dict of reconstructions
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

    if reconstructions:
        print(f"Reconstruction successful. Found {len(reconstructions)} models.")
        for i, recon in reconstructions.items():
            model_path = output_dir / str(i)
            model_path.mkdir(parents=True, exist_ok=True)
            recon.write(model_path)
            print(f"Model {i} written to {model_path}")
    else:
        print("Reconstruction failed.")

if __name__ == "__main__":
    image_dir = "data/pinecone_subset"
    output_dir = "output/colmap_run"
    run_colmap_reconstruction(image_dir, output_dir)
