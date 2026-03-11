import numpy as np
import os

def compare(name, pt, jax):
    mse = np.mean((pt - jax)**2)
    max_diff = np.max(np.abs(pt - jax))
    print(f"{name}: MSE: {mse:.2e}, Max Diff: {max_diff:.2e}")

def compare_lightglue():
    # Positional Encodings
    enc0_pt = np.load("output/lightglue_parity/enc0.npy")
    enc0_jax = np.load("output/lightglue_parity/jax_enc0.npy")
    compare("Encoding 0", enc0_pt, enc0_jax)
    
    enc1_pt = np.load("output/lightglue_parity/enc1.npy")
    enc1_jax = np.load("output/lightglue_parity/jax_enc1.npy")
    compare("Encoding 1", enc1_pt, enc1_jax)

    # Layers
    for i in range(9):
        q_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_self_q.npy")
        q_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_self_q.npy")
        compare(f"Layer {i} Self Q", q_pt, q_jax)
        
        msg_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_self_msg.npy")
        msg_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_self_msg.npy")
        compare(f"Layer {i} Self Msg", msg_pt, msg_jax)
        
        ffn0_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_ffn_0.npy")
        ffn0_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_ffn_0.npy")
        compare(f"Layer {i} FFN 0", ffn0_pt, ffn0_jax)
        
        ffn1_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_ffn_1.npy")
        ffn1_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_ffn_1.npy")
        compare(f"Layer {i} FFN 1", ffn1_pt, ffn1_jax)
        
        ffn2_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_ffn_2.npy")
        ffn2_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_ffn_2.npy")
        compare(f"Layer {i} FFN 2", ffn2_pt, ffn2_jax)
        
        if i == 0:
            print("Layer 0 FFN 1 sample (PT):", ffn1_pt[0, 0, :5])
            print("Layer 0 FFN 1 sample (JAX):", ffn1_jax[0, 0, :5])
            print("Layer 0 FFN 2 sample (PT):", ffn2_pt[0, 0, :5])
            print("Layer 0 FFN 2 sample (JAX):", ffn2_jax[0, 0, :5])

        self_out_pt = np.load(f"output/lightglue_parity/pt_layer_{i}_self_out0.npy")
        self_out_jax = np.load(f"output/lightglue_parity/jax_layer_{i}_self_out0.npy")
        compare(f"Layer {i} Self Out 0", self_out_pt, self_out_jax)

        pt0 = np.load(f"output/lightglue_parity/pt_desc0_layer_{i}.npy")
        jax0 = np.load(f"output/lightglue_parity/jax_desc0_layer_{i}.npy")
        compare(f"Layer {i} Desc 0", pt0, jax0)

    # Final scores
    pt_scores = np.load("output/lightglue_parity/pt_scores.npy")
    jax_log_scores = np.load("output/lightglue_parity/jax_log_scores.npy")
    
    b, m_p_1, n_p_1 = jax_log_scores.shape
    m, n = m_p_1 - 1, n_p_1 - 1
    scores = jax_log_scores[:, :m, :n]
    m0 = np.argmax(scores, axis=2)
    m1 = np.argmax(scores, axis=1)
    m_idx = np.arange(m)[None, :]
    m1_at_m0 = np.take_along_axis(m1, m0, axis=1)
    mutual0 = (m_idx == m1_at_m0)
    max0_exp = np.exp(np.max(scores, axis=2))
    jax_matching_scores0 = np.where(mutual0, max0_exp, 0.0)
    
    compare("Final Matching Scores 0", pt_scores, jax_matching_scores0)

if __name__ == "__main__":
    compare_lightglue()
