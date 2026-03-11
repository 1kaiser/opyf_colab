import jax
import jax.numpy as jnp
from models.jax.jax_lightglue.models.superpoint import SuperPoint
from flax import serialization
import numpy as np
import torch
import os
import sys

# Add LightGlue to path for original SP
sys.path.append(os.getcwd())
from LightGlue.lightglue.superpoint import SuperPoint as PTSuperPoint

def run_parity():
    # Load input image from previous capture
    img_path = "LightGlue/assets/sacre_coeur1.jpg"
    from LightGlue.lightglue.utils import read_image, numpy_image_to_torch
    image_np = read_image(img_path, grayscale=True)
    image_torch = numpy_image_to_torch(image_np).unsqueeze(0)
    
    # 1. Run PyTorch
    pt_model = PTSuperPoint().eval()
    # We want dense outputs for comparison before keypoint selection
    with torch.no_grad():
        # Manual forward to get dense scores and descriptors
        # Shared Encoder
        x = pt_model.relu(pt_model.conv1a(image_torch))
        x = pt_model.relu(pt_model.conv1b(x))
        x = pt_model.pool(x)
        x = pt_model.relu(pt_model.conv2a(x))
        x = pt_model.relu(pt_model.conv2b(x))
        x = pt_model.pool(x)
        x = pt_model.relu(pt_model.conv3a(x))
        x = pt_model.relu(pt_model.conv3b(x))
        x = pt_model.pool(x)
        x = pt_model.relu(pt_model.conv4a(x))
        x = pt_model.relu(pt_model.conv4b(x))

        cPa = pt_model.relu(pt_model.convPa(x))
        pt_scores_raw = pt_model.convPb(cPa)
        
        cDa = pt_model.relu(pt_model.convDa(x))
        pt_desc_raw = pt_model.convDb(cDa)
        pt_desc_raw = torch.nn.functional.normalize(pt_desc_raw, p=2, dim=1)

    # 2. Run JAX
    jax_model = SuperPoint()
    with open("weights/superpoint.msgpack", "rb") as f:
        variables = serialization.from_bytes(None, f.read())
    
    # JAX expects [B, H, W, 1]
    image_jax = jnp.array(image_torch.permute(0, 2, 3, 1).cpu().numpy())
    
    # We need to modify JAX model to return raw descriptors if we want to compare before sampling
    # But my JAX model already returns dense descriptors [B, H/8, W/8, D]
    out = jax_model.apply(variables, image_jax)
    jax_scores = out['scores']
    jax_desc = out['descriptors']
    
    # Compare
    print(f"JAX Scores shape: {jax_scores.shape}")
    print(f"JAX Desc shape: {jax_desc.shape}")
    
    # For PyTorch scores, we need to apply the same softmax and reshape as JAX
    pt_scores = torch.nn.functional.softmax(pt_scores_raw, 1)[:, :-1]
    b, _, h, w = pt_scores.shape
    pt_scores = pt_scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
    pt_scores = pt_scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
    # Apply NMS
    from LightGlue.lightglue.superpoint import simple_nms as pt_nms
    pt_scores = pt_nms(pt_scores, 4)
    # Border
    pad = 4
    pt_scores[:, :pad] = -1
    pt_scores[:, :, :pad] = -1
    pt_scores[:, -pad:] = -1
    pt_scores[:, :, -pad:] = -1
    
    pt_scores_np = pt_scores.cpu().numpy()
    pt_desc_np = pt_desc_raw.permute(0, 2, 3, 1).cpu().numpy()
    
    mse_scores = np.mean((pt_scores_np - jax_scores)**2)
    mse_desc = np.mean((pt_desc_np - jax_desc)**2)
    
    print(f"Scores MSE: {mse_scores:.2e}")
    print(f"Desc MSE: {mse_desc:.2e}")
    
    if mse_scores < 1e-10 and mse_desc < 1e-10:
        print("SUCCESS: SuperPoint parity achieved!")
    else:
        print("WARNING: Differences found in SuperPoint.")

if __name__ == "__main__":
    run_parity()
