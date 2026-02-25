import * as jax from "./src";
import * as np from "./src/library/numpy";

async function run_canal_opt() {
  console.log("--- Canal Optimization in JAX-JS ---");

  const Q_target = 50.0;
  const S_long = 1/5000;
  const n = 0.018;
  const s_side = 1.5;

  const calculate_discharge = (B: jax.Array, D: jax.Array) => {
    const area = B.ref.add(D.ref.mul(s_side)).mul(D.ref);
    const perimeter = B.ref.add(D.ref.mul(2 * Math.sqrt(1 + s_side**2)));
    const rh = area.ref.div(perimeter);
    const velocity = np.power(rh, 2/3).mul(Math.sqrt(S_long) / n);
    return area.mul(velocity);
  };

  const objective = (params: jax.Array) => {
    const B = params.ref.slice(0);
    const D = params.slice(1);
    const Q = calculate_discharge(B.ref, D.ref);
    const q_diff = np.array(Q_target).sub(Q);
    const penalty = q_diff.mul(q_diff.ref).mul(10.0); // Reduced penalty factor
    const area = B.add(D.ref.mul(s_side)).mul(D);
    return area.add(penalty);
  };

  const grad_fn = jax.grad(objective);
  let p = np.array([10.0, 3.0]);
  
  console.log("Starting Optimization Loop...");
  for (let i = 0; i < 10; i++) {
    const g = grad_fn(p.ref);
    
    // Update p = p - lr * g. Use very small LR
    const p_next = p.sub(g.mul(0.0001));
    p = p_next;
    
    // Basic clipping logic (not easily differentiable here, but fine for loop)
    // Actually, clipping in JAX-JS would need np.where
    
    const val = await p.ref.jsAsync();
    console.log(`Step ${i}: B=${val[0].toFixed(2)}, D=${val[1].toFixed(2)}`);
  }
  
  const final_p = await p.jsAsync();
  console.log(`\nFinal Optimized: B=${final_p[0].toFixed(2)}m, D=${final_p[1].toFixed(2)}m`);
}

run_canal_opt().catch(console.error);
