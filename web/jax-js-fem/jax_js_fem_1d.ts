import * as jax from "./src";
import * as np from "./src/library/numpy";
import * as linalg from "./src/library/numpy-linalg";

async function run_1d_fem() {
  console.log("--- 1D FEM Solver in jax-js ---");

  const n_elements = 5;
  const n_nodes = n_elements + 1;
  const L = 10.0;
  const element_L = L / n_elements;
  const E = 210000.0;
  const A = 100.0;
  const force = 1000.0;

  const k_val = (E * A) / element_L;

  const solve_fem = (k_param: jax.Array) => {
    const K_data = [
        [1, -1, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0],
        [0, -1, 2, -1, 0, 0],
        [0, 0, -1, 2, -1, 0],
        [0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, -1, 1]
    ];
    let K = np.array(K_data).mul(k_param.ref);
    
    const row_mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ]);
    
    const bc_row = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]);
    
    const K_bc = K.mul(row_mask).add(bc_row);
    const F = np.array([0, 0, 0, 0, 0, force]);
    return linalg.solve(K_bc, F);
  };

  const k_init = np.array(k_val);
  const u = solve_fem(k_init.ref);
  console.log("Displacements (mm):", await u.jsAsync());
  
  const objective = (k: jax.Array) => {
    const displacements = solve_fem(k);
    return displacements.slice(5).reshape([]).ref;
  };
  
  const grad_obj = jax.grad(objective);
  const sensitivity = grad_obj(k_init.ref);
  
  console.log("Initial Stiffness k:", k_val);
  console.log("JAX Sensitivity (du_tip / dk):", await sensitivity.jsAsync());
  
  // Finite Difference Verification
  const eps = 1.0;
  const u1 = await objective(np.array(k_val + eps)).jsAsync();
  const u0 = await objective(np.array(k_val)).jsAsync();
  const fd_sensitivity = (u1 - u0) / eps;
  console.log("Finite Difference Sensitivity:", fd_sensitivity);

  const theoretical = -n_elements * force / (k_val * k_val);
  console.log("Theoretical Sensitivity:", theoretical);
}

run_1d_fem().catch(console.error);
