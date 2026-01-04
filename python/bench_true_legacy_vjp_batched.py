"""
A3.5: Performance benchmark for batched VJP vs looped VJP

Measures and compares:
- Looped per-sample VJP
- Batched VJP with multi-RHS solve

Reports median time and speedup factor.
"""

import sys
import time
import numpy as np
import crm_diff_py

from control import (
    load_default_catheter_params,
    true_legacy_state_dim,
)


# Global params
PARAMS_DICT = None
N_ACT = 1


def load_params():
    """Load catheter parameters once."""
    global PARAMS_DICT
    if PARAMS_DICT is None:
        param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
        config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
        PARAMS_DICT = load_default_catheter_params(param_file, config_file)
    return PARAMS_DICT


def get_warmstart_state(params, dt=0.01, L_inserted=100.0):
    """Get a converged state for testing."""
    state_dim = true_legacy_state_dim(N_ACT)

    # Fixed seed for deterministic results
    np.random.seed(42)
    x_init = np.random.randn(state_dim) * 0.001
    u_init = np.array([[0.01, 0.0, 0.0]])

    # Split into x_coil and xf
    x_coil = x_init[:N_ACT*18].reshape(N_ACT, 18)
    xf = x_init[N_ACT*18:]

    return x_coil, xf, u_init


def benchmark_looped_vjp(x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, num_runs):
    """Benchmark looped per-sample VJP."""
    num_rhs = grad_tip_p_batch.shape[0]
    times = []

    for run_idx in range(num_runs):
        start = time.time()

        for i in range(num_rhs):
            result = crm_diff_py.true_legacy_step_vjp(
                x_coil, xf, u, dt, L_inserted, params, grad_tip_p_batch[i]
            )

        elapsed = time.time() - start
        times.append(elapsed)

    return np.array(times)


def benchmark_batched_vjp(x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, num_runs):
    """Benchmark batched VJP with multi-RHS solve."""
    times = []

    for run_idx in range(num_runs):
        start = time.time()

        result = crm_diff_py.true_legacy_step_vjp_batched(
            x_coil, xf, u, dt, L_inserted, params, grad_tip_p_batch
        )

        elapsed = time.time() - start
        times.append(elapsed)

    return np.array(times)


def run_benchmark(num_rhs, num_runs=10):
    """Run benchmark for given batch size."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: num_rhs={num_rhs}, num_runs={num_runs}")
    print(f"{'='*70}")

    params = load_params()
    dt = 0.01
    L_inserted = 100.0

    # Get warmstart
    x_coil, xf, u = get_warmstart_state(params, dt, L_inserted)

    # Create batch of upstream gradients
    np.random.seed(42 + num_rhs)  # Deterministic but different per batch size
    grad_tip_p_batch = np.random.randn(num_rhs, 3)

    print(f"  State: x_coil {x_coil.shape}, xf {xf.shape}, u {u.shape}")
    print(f"  Upstream gradients: {grad_tip_p_batch.shape}")

    # Warmup (1 run each)
    print("\n  Warmup...")
    _ = benchmark_looped_vjp(x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, 1)
    _ = benchmark_batched_vjp(x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, 1)

    # Benchmark looped
    print(f"\n  Benchmarking looped VJP ({num_runs} runs)...")
    times_looped = benchmark_looped_vjp(
        x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, num_runs
    )

    # Benchmark batched
    print(f"  Benchmarking batched VJP ({num_runs} runs)...")
    times_batched = benchmark_batched_vjp(
        x_coil, xf, u, params, grad_tip_p_batch, dt, L_inserted, num_runs
    )

    # Compute statistics
    median_looped = np.median(times_looped)
    median_batched = np.median(times_batched)
    mean_looped = np.mean(times_looped)
    mean_batched = np.mean(times_batched)
    speedup = median_looped / median_batched

    print(f"\n  {'Method':<20} {'Median (ms)':<15} {'Mean (ms)':<15} {'Std (ms)':<12}")
    print(f"  {'-'*62}")
    print(f"  {'Looped VJP':<20} {median_looped*1000:>14.3f} {mean_looped*1000:>14.3f} {np.std(times_looped)*1000:>11.3f}")
    print(f"  {'Batched VJP':<20} {median_batched*1000:>14.3f} {mean_batched*1000:>14.3f} {np.std(times_batched)*1000:>11.3f}")
    print(f"\n  Speedup (median): {speedup:.2f}x")

    return {
        'num_rhs': num_rhs,
        'median_looped': median_looped,
        'median_batched': median_batched,
        'speedup': speedup,
    }


def main():
    """Run benchmarks for various batch sizes."""
    print("\n" + "="*70)
    print("A3.5: BATCHED VJP PERFORMANCE BENCHMARK")
    print("="*70)

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    num_runs = 10

    results = []
    for num_rhs in batch_sizes:
        try:
            result = run_benchmark(num_rhs, num_runs)
            results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] Benchmark failed for num_rhs={num_rhs}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'num_rhs':<10} {'Looped (ms)':<15} {'Batched (ms)':<15} {'Speedup':<10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['num_rhs']:<10} {r['median_looped']*1000:>14.3f} {r['median_batched']*1000:>14.3f} {r['speedup']:>9.2f}x")

    print(f"\n  Batched VJP uses ONE factorization per sample, shared across all RHS.")
    print(f"  Speedup increases with num_rhs as the factorization cost is amortized.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
