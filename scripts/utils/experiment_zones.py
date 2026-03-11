from pipelines.pipelines.pipeline_jax import ReconstructionPipeline
import os

def run_experiment():
    pipeline = ReconstructionPipeline(
        "weights/depth_pro.msgpack",
        "weights/superpoint.msgpack",
        "weights/superpoint_lightglue.msgpack"
    )
    
    image_dir = "data/pinecone_subset"
    
    for nz in [2, 3, 4, 5]:
        print("\n" + "="*30)
        print(f"STARTING EXPERIMENT: {nz} ZONES")
        print("="*30)
        pipeline.run(image_dir, num_zones=nz)

if __name__ == "__main__":
    run_experiment()
