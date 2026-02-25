import * as jax from "./src";
import * as np from "./src/library/numpy";

export const cg_solve = (
  A_op: (x: jax.Array) => jax.Array,
  b: jax.Array,
  x0: jax.Array,
  max_iter: number = 100
) => {
  let x = x0;
  let r = b.sub(A_op(x.ref));
  let p = r.ref;
  let rsold = r.ref.mul(r.ref).sum();

  for (let i = 0; i < max_iter; i++) {
    const Ap = A_op(p.ref);
    const denom = p.ref.mul(Ap.ref).sum();
    
    const alpha = np.where(denom.ref.notEqual(0), rsold.ref.div(np.where(denom.ref.notEqual(0), denom, 1.0)), 0.0);
    
    const x_next = x.add(p.ref.mul(alpha.ref));
    const r_next = r.sub(Ap.mul(alpha));
    const rsnew = r_next.ref.mul(r_next.ref).sum();
    
    const beta = np.where(rsold.ref.notEqual(0), rsnew.ref.div(np.where(rsold.ref.notEqual(0), rsold, 1.0)), 0.0);
    const p_next = r_next.ref.add(p.mul(beta));
    
    x = x_next;
    r = r_next;
    p = p_next;
    rsold = rsnew;
  }
  
  return x;
};
