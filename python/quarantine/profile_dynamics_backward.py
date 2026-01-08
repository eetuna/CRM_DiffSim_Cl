"""
Profiling script for dynamics backward pass.
Runs a subset of CP2.4 gradcheck with cProfile to identify hotspots.
"""
import sys
import os
import cProfile
import pstats
import io
import torch
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

from . import crm_dynamics_torch
import crm_diff_py


def profile_gradcheck():
    """Run a lightweight gradcheck to profile backward pass."""

    # Load parameters
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

    # Operating point: Actuated
    x_t_np = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u_t_np = np.array([0.1, 0.0, 0.0], dtype=np.float64)

    # Convert to torch tensors
    x_t = torch.from_numpy(x_t_np).clone().requires_grad_(True)
    u_t = torch.from_numpy(u_t_np).clone().requires_grad_(True)

    # Function to test gradients w.r.t. u_t
    def func_u_t(u):
        """Test gradients w.r.t. u_t with x_t fixed."""
        return crm_dynamics_torch.dynamics_step(x_t.detach(), u, dt, L_inserted, params_dict)

    print("Profiling gradcheck for ∂x_next/∂u_t...")
    print(f"  This mimics torch.autograd.gradcheck behavior")
    print()

    u_t_test = u_t.clone().detach().requires_grad_(True).to(dtype=torch.float64)

    # Run gradcheck (limited iterations for profiling)
    torch.autograd.gradcheck(
        func_u_t,
        u_t_test,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        raise_exception=True
    )


def main():
    print("="*70)
    print("Profiling dynamics_backward hotspots")
    print("="*70)
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Run profiling
    profiler.enable()
    profile_gradcheck()
    profiler.disable()

    # Print results
    print("\n" + "="*70)
    print("PROFILING RESULTS")
    print("="*70)
    print()

    # Create string buffer for stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    # Sort by cumulative time and print top 30 functions
    ps.sort_stats('cumulative')
    ps.print_stats(30)

    print(s.getvalue())

    # Also save to file
    output_file = "/tmp/dynamics_profile.txt"
    with open(output_file, 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats(50)

    print(f"\nFull profile saved to: {output_file}")

    # Print summary of key hotspots
    print("\n" + "="*70)
    print("KEY HOTSPOTS IDENTIFIED")
    print("="*70)
    print()

    # Filter for dynamics-related functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats('dynamics|equilibrium|crm_diff')

    print(s.getvalue())


if __name__ == "__main__":
    sys.exit(main())
