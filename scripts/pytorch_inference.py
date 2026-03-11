
import torch
import numpy as np
import os
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.path_to_dust3r import *
from dust3r.utils.image import load_images
from dust3r.inference import inference
import open3d as o3d

def save_point_cloud(pts, colors, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    model_path = "mast3r_repo/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    model.eval()

    # Load a pair of images
    image_dir = "data/pinecone/images"
    img_path1 = os.path.join(image_dir, "IMG_7238.JPG")
    img_path2 = os.path.join(image_dir, "IMG_7239.JPG")
    
    images = load_images([img_path1, img_path2], size=512)

    # Run inference
    with torch.no_grad():
        output = inference([tuple(images)], model, device, batch_size=1, verbose=True)

    # Extract predictions
    pred1 = output['pred1']
    pred2 = output['pred2']
    
    # Get 3D points and confidence
    pts3d_1 = pred1['pts3d'].cpu().numpy()
    conf_1 = pred1['conf'].cpu().numpy()
    
    pts3d_2 = pred2['pts3d_in_other_view'].cpu().numpy()
    conf_2 = pred2['conf'].cpu().numpy()

    # Get colors from input images
    rgb1 = images[0]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb2 = images[1]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    h1, w1 = pts3d_1.shape[:2]
    h2, w2 = pts3d_2.shape[:2]

    colors1 = torch.nn.functional.interpolate(torch.from_numpy(rgb1).permute(2,0,1).unsqueeze(0), size=(h1,w1), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0).numpy()
    colors2 = torch.nn.functional.interpolate(torch.from_numpy(rgb2).permute(2,0,1).unsqueeze(0), size=(h2,w2), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0).numpy()

    # Create output directory
    os.makedirs("output/pytorch_inference", exist_ok=True)

    # Save outputs
    np.save("output/pytorch_inference/pts3d_1.npy", pts3d_1)
    np.save("output/pytorch_inference/conf_1.npy", conf_1)
    np.save("output/pytorch_inference/pts3d_2.npy", pts3d_2)
    np.save("output/pytorch_inference/conf_2.npy", conf_2)

    # Save point clouds
    save_point_cloud(pts3d_1.reshape(-1, 3), colors1.reshape(-1, 3), "output/pytorch_inference/pcd1.ply")
    save_point_cloud(pts3d_2.reshape(-1, 3), colors2.reshape(-1, 3), "output/pytorch_inference/pcd2.ply")
    
    print("PyTorch inference complete. Outputs saved to output/pytorch_inference/")

if __name__ == '__main__':
    main()
