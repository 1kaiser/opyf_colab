import torch
import numpy as np
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import load_image, rbd
import os

def capture_pytorch_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load images
    img0_path = "LightGlue/assets/sacre_coeur1.jpg"
    img1_path = "LightGlue/assets/sacre_coeur2.jpg"
    
    from LightGlue.lightglue.utils import read_image, numpy_image_to_torch
    image0 = numpy_image_to_torch(read_image(img0_path, grayscale=True)).to(device).unsqueeze(0)
    image1 = numpy_image_to_torch(read_image(img1_path, grayscale=True)).to(device).unsqueeze(0)
    
    # Extract features using SuperPoint
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
    feats0 = extractor({'image': image0})
    feats1 = extractor({'image': image1})
    
    # Run LightGlue
    matcher = LightGlue(features='superpoint', flash=False).eval().to(device)
    
    # We'll use hooks or just run manually
    data0, data1 = feats0, feats1
    kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
    b, m, _ = kpts0.shape
    b, n, _ = kpts1.shape
    
    from LightGlue.lightglue.lightglue import normalize_keypoints, apply_cached_rotary_emb
    kpts0 = normalize_keypoints(kpts0, None)
    kpts1 = normalize_keypoints(kpts1, None)
    
    desc0 = data0["descriptors"].detach().contiguous()
    desc1 = data1["descriptors"].detach().contiguous()
    
    # input_proj is identity for superpoint
    # cache positional embeddings
    encoding0 = matcher.posenc(kpts0)
    encoding1 = matcher.posenc(kpts1)
    
    np.save("output/lightglue_parity/enc0.npy", encoding0.detach().cpu().numpy())
    np.save("output/lightglue_parity/enc1.npy", encoding1.detach().cpu().numpy())

    for i in range(matcher.conf.n_layers):
        layer = matcher.transformers[i]
        
        # Self-Attention Part
        qkv = layer.self_attn.Wqkv(desc0)
        qkv = qkv.unflatten(-1, (layer.self_attn.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding0, q)
        k = apply_cached_rotary_emb(encoding0, k)
        context = layer.self_attn.inner_attn(q, k, v)
        message = layer.self_attn.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        
        np.save(f"output/lightglue_parity/pt_layer_{i}_self_q.npy", q.detach().cpu().numpy())
        np.save(f"output/lightglue_parity/pt_layer_{i}_self_msg.npy", message.detach().cpu().numpy())

        # FFN Granular
        ffn_in = torch.cat([desc0, message], -1)
        ffn_0_out = layer.self_attn.ffn[0](ffn_in)
        ffn_1_out = layer.self_attn.ffn[1](ffn_0_out)
        ffn_2_out = layer.self_attn.ffn[2](ffn_1_out)
        ffn_3_out = layer.self_attn.ffn[3](ffn_2_out)
        
        np.save(f"output/lightglue_parity/pt_layer_{i}_ffn_0.npy", ffn_0_out.detach().cpu().numpy())
        np.save(f"output/lightglue_parity/pt_layer_{i}_ffn_1.npy", ffn_1_out.detach().cpu().numpy())
        np.save(f"output/lightglue_parity/pt_layer_{i}_ffn_2.npy", ffn_2_out.detach().cpu().numpy())

        desc0_self = desc0 + ffn_3_out
        desc1_self = layer.self_attn(desc1, encoding1)
        np.save(f"output/lightglue_parity/pt_layer_{i}_self_out0.npy", desc0_self.detach().cpu().numpy())
        
        desc0, desc1 = layer.cross_attn(desc0_self, desc1_self)
        np.save(f"output/lightglue_parity/pt_desc0_layer_{i}.npy", desc0.detach().cpu().numpy())

    scores, _ = matcher.log_assignment[i](desc0, desc1)
    
    # Save inputs for JAX
    os.makedirs("output/lightglue_parity", exist_ok=True)
    
    np.save("output/lightglue_parity/kpts0.npy", feats0['keypoints'].detach().cpu().numpy())
    np.save("output/lightglue_parity/desc0.npy", feats0['descriptors'].detach().cpu().numpy())
    np.save("output/lightglue_parity/kpts1.npy", feats1['keypoints'].detach().cpu().numpy())
    np.save("output/lightglue_parity/desc1.npy", feats1['descriptors'].detach().cpu().numpy())
    
    from LightGlue.lightglue.lightglue import filter_matches
    m0, m1, mscores0, mscores1 = filter_matches(scores, matcher.conf.filter_threshold)
    
    # Save outputs for comparison
    np.save("output/lightglue_parity/pt_scores.npy", mscores0.detach().cpu().numpy())
    # Note: sim might not be returned in default forward if not requested, 
    # but our JAX version returns it. Let's check if we can get it.
    
    print("PyTorch outputs captured successfully.")

if __name__ == "__main__":
    capture_pytorch_outputs()
