
import torch
import numpy as np
import os
import sys
from PIL import Image

# Add mast3r_repo to sys.path
sys.path.append(os.path.abspath("mast3r_repo"))

from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.inference import inference

def main():
    device = 'cpu'
    print(f"Using device: {device}")

    # Load Model
    model_path = "mast3r_repo/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    model.eval()

    # --- Comprehensive Decoder Block Capture ---
    dec1_block_outputs = [None] * 12
    dec2_block_outputs = [None] * 12
    dec1_norm_out = None
    dec2_norm_out = None
    hooks = []

    def make_hook(index, side):
        def hook_fn(module, input, output):
            if side == 1:
                dec1_block_outputs[index] = output[0].detach().cpu().numpy()
            else:
                dec2_block_outputs[index] = output[0].detach().cpu().numpy()
        return hook_fn

    def norm_hook_fn1(module, input, output):
        nonlocal dec1_norm_out
        dec1_norm_out = output.detach().cpu().numpy()
    def norm_hook_fn2(module, input, output):
        nonlocal dec2_norm_out
        dec2_norm_out = output.detach().cpu().numpy()

    # Register a hook for each decoder block
    for i in range(12):
        hooks.append(model.dec_blocks[i].register_forward_hook(make_hook(i, 1)))
        hooks.append(model.dec_blocks2[i].register_forward_hook(make_hook(i, 2)))
    
    # Hook for path_1 (output of refinenet1)
    path1_outs = []
    def path1_hook(module, input, output):
        path1_outs.append(output.detach().cpu().numpy())
    
    hooks.append(model.downstream_head1.dpt.scratch.refinenet1.register_forward_hook(path1_hook))
    hooks.append(model.downstream_head2.dpt.scratch.refinenet1.register_forward_hook(path1_hook))

    # Load a pair of images
    image_dir = "data/pinecone/images"
    img_path1 = os.path.join(image_dir, "IMG_7238.JPG")
    img_path2 = os.path.join(image_dir, "IMG_7239.JPG")
    images = load_images([img_path1, img_path2], size=512)

    # Run inference
    with torch.no_grad():
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Save captured path1
    output_dir = "output/pytorch_inference"
    if len(path1_outs) >= 2:
        # Transpose to NHWC
        p1_1 = np.transpose(path1_outs[0], (0, 2, 3, 1))
        p1_2 = np.transpose(path1_outs[1], (0, 2, 3, 1))
        np.save(os.path.join(output_dir, "dpt_act0_1.npy"), p1_1)
        np.save(os.path.join(output_dir, "dpt_act0_2.npy"), p1_2)

    # Save all captured intermediate tensors
    output_dir = "output/pytorch_inference"
    block_dir = os.path.join(output_dir, "decoder_blocks")
    os.makedirs(block_dir, exist_ok=True)
    for i in range(12):
        if dec1_block_outputs[i] is not None:
            np.save(os.path.join(block_dir, f"dec1_blk_{i}.npy"), dec1_block_outputs[i])
        if dec2_block_outputs[i] is not None:
            np.save(os.path.join(block_dir, f"dec2_blk_{i}.npy"), dec2_block_outputs[i])
    
    if dec1_norm_out is not None:
        # dec1_norm_out will contain both side 1 and side 2 if hooked twice or called twice.
        # Actually in AsymmetricMASt3R.forward:
        # q1, q2 = self.dec_norm(q1), self.dec_norm(q2)
        # So the hook will be called twice. My hook function needs to handle that.
        pass

    # Actually, let's just save the final predictions from 'output'
    pred1 = output['pred1']
    pred2 = output['pred2']
    
    pts1 = pred1['pts3d'].cpu().numpy()
    conf1 = pred1['conf'].cpu().numpy()
    pts1_full = np.concatenate([pts1, conf1[..., None]], axis=-1)
    np.save(os.path.join(output_dir, "pts1.npy"), pts1_full)
    
    pts2 = pred2['pts3d_in_other_view'].cpu().numpy()
    conf2 = pred2['conf'].cpu().numpy()
    pts2_full = np.concatenate([pts2, conf2[..., None]], axis=-1)
    np.save(os.path.join(output_dir, "pts2.npy"), pts2_full)
    
    np.save(os.path.join(output_dir, "desc1.npy"), pred1['desc'].cpu().numpy())
    np.save(os.path.join(output_dir, "desc2.npy"), pred2['desc'].cpu().numpy())

    print(f"PyTorch intermediate decoder blocks and final predictions saved to {output_dir}")

if __name__ == '__main__':
    main()
