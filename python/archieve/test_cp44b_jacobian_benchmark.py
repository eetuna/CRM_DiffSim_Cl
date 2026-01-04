#!/usr/bin/env python3
"""
CP4.4b: Jacobian Performance Benchmark

Measures speedup of C++ dynamics_linearize vs PyTorch autograd.

Runs 100 linearizations for each method and reports:
- Total time
- Time per linearization
- Speedup factor
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import crm_diff_py
from crm_dynamics_torch import dynamics_step


def benchmark_pytorch(x_t, u_t, dt, L_inserted, params_dict, n_iters=100):
    """
    Benchmark PyTorch autograd Jacobian extraction.

    Returns:
        float: Total time in seconds
    """
    x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
    u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

    t_start = time.time()

    for _ in range(n_iters):
        # A = ∂x_next/∂x_t
        A = torch.autograd.functional.jacobian(
            lambda x: dynamics_step(x, u_t_torch, dt, L_inserted, params_dict),
            x_t_torch
        ).numpy()

        # B = ∂x_next/∂u_t
        B = torch.autograd.functional.jacobian(
            lambda u: dynamics_step(x_t_torch, u, dt, L_inserted, params_dict),
            u_t_torch
        ).numpy()

    t_end = time.time()
    return t_end - t_start


def benchmark_cpp(x_t, u_t, dt, L_inserted, params_dict, n_iters=100):
    """
    Benchmark C++ dynamics_linearize.

    Returns:
        float: Total time in seconds
    """
    t_start = time.time()

    for _ in range(n_iters):
        result = crm_diff_py.dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)
        A = result['A']
        B = result['B']

    t_end = time.time()
    return t_end - t_start


def main():
    print("="*80)
    print("CP4.4b: Jacobian Performance Benchmark")
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
    _ = benchmark_pytorch(x_t, u_t, dt, L_inserted, params_dict, n_iters=1)
    _ = benchmark_cpp(x_t, u_t, dt, L_inserted, params_dict, n_iters=1)
    print()

    # Benchmark PyTorch
    print(f"Benchmarking PyTorch autograd ({n_iters} iterations)...")
    time_pytorch = benchmark_pytorch(x_t, u_t, dt, L_inserted, params_dict, n_iters)
    time_per_pytorch = time_pytorch / n_iters
    print(f"  Total time: {time_pytorch:.3f}s")
    print(f"  Time per linearization: {time_per_pytorch*1000:.2f}ms")
    print()

    # Benchmark C++
    print(f"Benchmarking C++ dynamics_linearize ({n_iters} iterations)...")
    time_cpp = benchmark_cpp(x_t, u_t, dt, L_inserted, params_dict, n_iters)
    time_per_cpp = time_cpp / n_iters
    print(f"  Total time: {time_cpp:.3f}s")
    print(f"  Time per linearization: {time_per_cpp*1000:.2f}ms")
    print()

    # Speedup
    speedup = time_pytorch / time_cpp
    print("="*80)
    print("Results:")
    print(f"  PyTorch: {time_per_pytorch*1000:.2f}ms per linearization")
    print(f"  C++:     {time_per_cpp*1000:.2f}ms per linearization")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    if speedup >= 1.5:
        print(f"✓ PASS: C++ is {speedup:.2f}x faster than PyTorch")
        return 0
    elif speedup >= 1.0:
        print(f"⚠ WARNING: C++ is only {speedup:.2f}x faster (expected >1.5x)")
        return 0
    else:
        print(f"✗ FAIL: C++ is slower than PyTorch ({speedup:.2f}x)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
