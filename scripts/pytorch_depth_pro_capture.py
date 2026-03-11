import sys
import os
import torch
import numpy as np
from PIL import Image

# Add depth_pro_repo/src to path
sys.path.append(os.path.join(os.getcwd(), "depth_pro_repo", "src"))

from depth_pro.depth_pro import DepthProConfig
import depth_pro

def capture_depth_pro():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="depth_pro_repo/checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    
    # Load example image
    img_path = "depth_pro_repo/data/example.jpg"
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_tensor = transform(image)
    
    # Run inference
    with torch.no_grad():
        # DepthPro operates at 1536x1536. It resizes if needed.
        # But we want to capture the internal forward to compare.
        # model.infer() does resizing.
        # Let's run model.forward() on a 1536x1536 image.
        input_image = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), size=(1536, 1536), mode="bilinear", align_corners=False
        )
        
        # Capture intermediate hooks
        encodings = model.encoder(input_image)
        features, lowres_features = model.decoder(encodings)
        canonical_inv_depth = model.head(features)
        
        # FOV shapes
        x_down = torch.nn.functional.interpolate(input_image, scale_factor=0.25, mode="bilinear", align_corners=False)
        np.save("output/depth_pro_parity/pt_x_down.npy", x_down.cpu().numpy())
        
        fov_feat = model.fov.encoder(x_down)[:, 1:].permute(0, 2, 1)
        fov_feat_2d = fov_feat.reshape(1, 128, 24, 24)
        np.save("output/depth_pro_parity/pt_fov_feat.npy", fov_feat_2d.detach().cpu().numpy())
        
        fov_low = model.fov.downsample(lowres_features)
        np.save("output/depth_pro_parity/pt_fov_low.npy", fov_low.detach().cpu().numpy())
        
        fov_deg = model.fov(input_image, lowres_features)
        
    os.makedirs("output/depth_pro_parity", exist_ok=True)
    np.save("output/depth_pro_parity/input_image.npy", input_image.cpu().numpy())
    
    for i, enc in enumerate(encodings):
        np.save(f"output/depth_pro_parity/pt_enc_{i}.npy", enc.detach().cpu().numpy())
        
    np.save("output/depth_pro_parity/pt_canonical_inv_depth.npy", canonical_inv_depth.detach().cpu().numpy())
    np.save("output/depth_pro_parity/pt_fov_deg.npy", fov_deg.detach().cpu().numpy())
    np.save("output/depth_pro_parity/pt_lowres_features.npy", lowres_features.detach().cpu().numpy())
    
    print("PyTorch Depth Pro outputs captured successfully.")

if __name__ == "__main__":
    capture_depth_pro()
