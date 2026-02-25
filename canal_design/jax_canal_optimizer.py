import jax
import jax.numpy as jnp
import json
import os

from jax import config
config.update("jax_enable_x64", True)

class JAXCanalOptimizer:
    @staticmethod
    def get_is_min_radius(Q):
        q_bins = jnp.array([0.3, 3.0, 15.0, 30.0, 80.0])
        r_values = jnp.array([100.0, 150.0, 300.0, 600.0, 1000.0, 1500.0])
        return r_values[jnp.digitize(Q, q_bins)]

    @staticmethod
    def get_is_freeboard(Q):
        q_bins = jnp.array([0.75, 1.5, 85.0])
        fb_values = jnp.array([0.30, 0.50, 0.60, 0.75])
        return fb_values[jnp.digitize(Q, q_bins)]

    @staticmethod
    def calculate_hydraulics(params, constants):
        B, D, S_side = params
        n, S_long = constants
        area = (B + S_side * D) * D
        perimeter = B + 2 * D * jnp.sqrt(1 + S_side**2)
        rh = area / perimeter
        velocity = (1.0 / n) * (rh**(2.0/3.0)) * (jnp.sqrt(S_long))
        discharge = area * velocity
        return discharge, velocity, area, perimeter

    @classmethod
    def objective_fn(cls, params, Q_target, constants):
        B, D, S_side = params
        discharge, velocity, area, perimeter = cls.calculate_hydraulics(params, constants)
        cost_excavation = area
        cost_lining = perimeter * 0.1
        penalty_q = 1000.0 * (jnp.maximum(0.0, Q_target - discharge))**2
        penalty_v_max = 5000.0 * (jnp.maximum(0.0, velocity - 2.5))**2
        penalty_v_min = 5000.0 * (jnp.maximum(0.0, 0.6 - velocity))**2
        penalty_slope = 100.0 * (S_side - 1.5)**2
        return cost_excavation + cost_lining + penalty_q + penalty_v_max + penalty_v_min + penalty_slope

    @classmethod
    def run_optimization(cls, Q_target, S_long, n=0.018):
        print(f"Optimizing for Q={Q_target}, Slope={S_long}...")
        params = jnp.array([10.0, 3.0, 1.5])
        constants = (n, S_long)
        grad_fn = jax.jit(jax.grad(cls.objective_fn))
        lr = 0.01
        for i in range(200):
            grads = grad_fn(params, Q_target, constants)
            params = params - lr * grads
            params = jnp.clip(params, jnp.array([1.0, 0.5, 1.0]), jnp.array([50.0, 10.0, 3.0]))
        B, D, S_side = params
        q, v, a, p = cls.calculate_hydraulics(params, constants)
        fb, min_r = float(cls.get_is_freeboard(q)), float(cls.get_is_min_radius(q))
        results = {
            "is_discharge_target": float(Q_target),
            "bed_width": float(B),
            "water_depth": float(D),
            "side_slope": float(S_side),
            "freeboard": fb,
            "total_depth": float(D + fb),
            "calculated_discharge": float(q),
            "velocity": float(v),
            "min_radius": min_r,
            "manning_n": float(n),
            "long_slope": float(S_long)
        }
        print("\n--- OPTIMIZATION COMPLETE ---")
        return results

if __name__ == "__main__":
    opt = JAXCanalOptimizer()
    solution = opt.run_optimization(Q_target=50.0, S_long=1/5000)
    with open("canal_params.json", "w") as f:
        json.dump(solution, f, indent=4)
    print("Saved parameters to canal_params.json")
