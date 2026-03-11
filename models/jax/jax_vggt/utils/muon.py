import jax
import jax.numpy as jnp

def newton_schulz_orthogonalization(G, iters=5):
    """
    Orthogonalize gradient matrix G using Newton-Schulz iterations.
    G: (..., D_in, D_out)
    """
    # G_norm: (..., D_in, D_out)
    # We use a standard set of coefficients for 5 iters to converge to orthogonal matrix
    # Based on Muon optimizer implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Scale G to avoid overflow/underflow
    # For a matrix G, we want to compute G (G^T G)^{-1/2}
    # Initial X = G / ||G||_F
    norm = jnp.linalg.norm(G, axis=(-2, -1), keepdims=True)
    X = G / (norm + 1e-7)
    
    # Transpose for the update if D_in > D_out to maintain (D_in, D_out)
    D_in, D_out = G.shape[-2], G.shape[-1]
    if D_in < D_out:
        X = X.swapaxes(-1, -2) # (..., D_out, D_in)

    for _ in range(iters):
        # A = X @ X^T
        A = X @ X.swapaxes(-1, -2)
        # Update X: (aI + bA + cA^2) @ X
        X = (a * jnp.eye(X.shape[-2]) + b * A + c * A @ A) @ X
    
    if D_in < D_out:
        X = X.swapaxes(-1, -2) # (..., D_in, D_out)
        
    return X

def muon_update(params, grads, lr=0.1, ns_iters=5):
    """
    Muon optimizer update rule.
    params: (..., D_in, D_out)
    grads: (..., D_in, D_out)
    """
    # Orthogonalize grads
    u_grads = newton_schulz_orthogonalization(grads, iters=ns_iters)
    # Scale by learning rate
    return params - lr * u_grads
