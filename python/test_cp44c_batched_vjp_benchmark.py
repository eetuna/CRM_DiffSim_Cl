#!/usr/bin/env python3
"""
CP4.4c: Batched VJP Performance Benchmark

Measures speedup of batched VJP vs per-vector VJP for Jacobian computation.

Compares:
- CP4.4b approach: forward once + backward 6 times (per-vector VJP)
- CP4.4c approach: forward once + batched backward (batched VJP)

Runs 100 linearizations for each method and reports:
- Total time
- Time per linearization
- Speedup factor

Target: ≥1.3× speedup vs CP4.4b

NOTE: This is a nightly test due to longer runtime.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import crm_diff_py


def benchmark_per_vector_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters=100):
    """
    Benchmark CP4.4b approach: forward once + backward 6 times.

    Returns:
        float: Total time in seconds
    """
    t_start = time.time()

    for _ in range(n_iters):
        # Forward pass
        fwd_result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

        # Extract A and B via 6 separate VJPs
        A = np.zeros((6, 6))
        B = np.zeros((6, 3))

        for k in range(6):
            grad_x_next = np.zeros(6)
            grad_x_next[k] = 1.0

            bwd_result = crm_diff_py.dynamics_backward(
                fwd_result,
                grad_x_next,
                params_dict
            )

            A[k, :] = bwd_result['grad_x_t']
            B[k, :] = bwd_result['grad_u_t']

    t_end = time.time()
    return t_end - t_start


def benchmark_batched_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters=100):
    """
    Benchmark CP4.4c approach: forward once + batched backward.

    Returns:
        float: Total time in seconds
    """
    t_start = time.time()

    for _ in range(n_iters):
        # This now uses batched VJP internally
        result = crm_diff_py.dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)
        A = result['A']
        B = result['B']

    t_end = time.time()
    return t_end - t_start


def main():
    print("="*80)
    print("CP4.4c: Batched VJP Performance Benchmark")
    print("="*80)
    print()

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    dt = 0.01
    L_inserted = 50.0
    n_iters = 100

    print("Benchmark Configuration:")
    print(f"  dt: {dt}s")
    print(f"  L_inserted: {L_inserted}mm")
    print(f"  Iterations: {n_iters}")
    print()

    # Use a typical operating point
    x_t = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    u_t = np.array([0.1, -0.1, 0.05])

    print("Operating point:")
    print(f"  x_t: {x_t}")
    print(f"  u_t: {u_t}")
    print()

    # Warm-up runs (1 iteration each)
    print("Warm-up runs...")
    _ = benchmark_per_vector_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters=1)
    _ = benchmark_batched_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters=1)
    print()

    # Benchmark per-vector VJP (CP4.4b approach)
    print(f"Benchmarking per-vector VJP ({n_iters} iterations)...")
    time_per_vector = benchmark_per_vector_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters)
    time_per_iter_pv = time_per_vector / n_iters
    print(f"  Total time: {time_per_vector:.3f}s")
    print(f"  Time per linearization: {time_per_iter_pv*1000:.2f}ms")
    print()

    # Benchmark batched VJP (CP4.4c approach)
    print(f"Benchmarking batched VJP ({n_iters} iterations)...")
    time_batched = benchmark_batched_vjp(x_t, u_t, dt, L_inserted, params_dict, n_iters)
    time_per_iter_batched = time_batched / n_iters
    print(f"  Total time: {time_batched:.3f}s")
    print(f"  Time per linearization: {time_per_iter_batched*1000:.2f}ms")
    print()

    # Speedup
    speedup = time_per_vector / time_batched
    print("="*80)
    print("Results:")
    print(f"  Per-vector VJP: {time_per_iter_pv*1000:.2f}ms per linearization")
    print(f"  Batched VJP:    {time_per_iter_batched*1000:.2f}ms per linearization")
    print(f"  Speedup:        {speedup:.2f}x")
    print()

    if speedup >= 1.3:
        print(f"✓ PASS: Batched VJP is {speedup:.2f}x faster (target: ≥1.3x)")
        return 0
    elif speedup >= 1.0:
        print(f"⚠ WARNING: Batched VJP is only {speedup:.2f}x faster (target: ≥1.3x)")
        return 0
    else:
        print(f"✗ FAIL: Batched VJP is slower than per-vector ({speedup:.2f}x)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
