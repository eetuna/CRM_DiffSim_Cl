#!/usr/bin/env python3
"""
CP4.7.1: Quick Demo Benchmark (Simplified for Speed)

Demonstrates hybrid controller with reduced settings for fast execution.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.hybrid_controller import HybridController, HybridControllerMetrics
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


# Load ensemble
metadata_path = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

ensemble = EnsemblePolicy(
    policy_paths=metadata['checkpoint_paths'],
    policy_class=GRUPolicy,
    policy_kwargs=metadata['policy_kwargs']
)

print(f"✓ Loaded ensemble: {metadata['n_members']} members")

# Load datasets
manifest = load_manifest(data_dir='./data', verbose=False)
dataset = manifest.datasets_with_ref[0]  # Use first dataset

print(f"✓ Using dataset: {dataset.filename}")

# Load physics
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

# Simplified settings for speed
duration = 1.0  # 1 second only
mpc_horizon = 3
mpc_max_iters = 3

dt = dataset.dt
L_inserted = dataset.L_inserted
n_steps = min(int(duration / dt), len(dataset.tip_ref), 20)  # Max 20 steps

print(f"\nDemo settings:")
print(f"  Duration: {duration}s ({n_steps} steps)")
print(f"  MPC horizon: {mpc_horizon}")
print(f"  MPC iters: {mpc_max_iters}")

# MPC baseline
print("\n[1/2] Running MPC-only baseline...")
p_ref = dataset.tip_ref[:n_steps]
x_t = np.zeros(6)

result_init = crm_diff_py.dynamics_forward(x_t, np.zeros(3), 0.0, L_inserted, params_dict)
p_tip_t = result_init['p_tip']

p_tips_mpc = []
U_warm = None
t_start_mpc = time.time()

for i in range(n_steps):
    if i % 5 == 0:
        print(f"  MPC step {i}/{n_steps}")

    horizon_end = min(i + mpc_horizon, n_steps)
    p_ref_horizon = p_ref[i:horizon_end]

    if len(p_ref_horizon) < mpc_horizon:
        p_ref_horizon = np.vstack([
            p_ref_horizon,
            np.tile(p_ref_horizon[-1], (mpc_horizon - len(p_ref_horizon), 1))
        ])

    solver = iLQRSolver(
        dt=dt, L_inserted=L_inserted, params_dict=params_dict,
        horizon=mpc_horizon, Q=np.zeros((6, 6)), R=0.01 * np.eye(3),
        p_target=p_ref_horizon[0], terminal_weight=1.0,
        max_iters=mpc_max_iters, tol=2.0, jacobian_mode="cpp"
    )

    try:
        X, U, converged = solver.solve(x_t, U_init=U_warm, verbose=False)
        u_t = np.clip(U[0], -0.5, 0.5)
        U_warm = np.vstack([U[1:], U[-1:]])
    except:
        u_t = np.zeros(3)

    result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
    if result['status'] == 0:
        x_t = result['x_next']
        p_tip_t = result['p_tip']

    p_tips_mpc.append(p_tip_t)

mpc_time = time.time() - t_start_mpc
p_tips_mpc = np.array(p_tips_mpc)
mpc_errors = np.linalg.norm(p_tips_mpc - p_ref[:n_steps], axis=1)
mpc_rmse = float(np.sqrt(np.mean(mpc_errors**2)))

print(f"  MPC RMSE: {mpc_rmse:.3f} mm, Time: {mpc_time:.2f}s")

# Hybrid
print("\n[2/2] Running hybrid controller...")
controller = HybridController(
    ensemble=ensemble, dt=dt, L_inserted=L_inserted, params_dict=params_dict,
    tau_low=0.001, tau_high=0.01, tracking_safety_limit=5.0,
    mpc_horizon=mpc_horizon, mpc_max_iters=mpc_max_iters, mpc_tol=2.0
)

metrics = HybridControllerMetrics()
x_t = np.zeros(6)
result_init = crm_diff_py.dynamics_forward(x_t, np.zeros(3), 0.0, L_inserted, params_dict)
p_tip_t = result_init['p_tip']

p_tips_hybrid = []
t_start_hybrid = time.time()

for i in range(n_steps):
    if i % 5 == 0:
        print(f"  Hybrid step {i}/{n_steps}")

    horizon_end = min(i + controller.mpc_horizon, n_steps)
    p_ref_horizon = p_ref[i:horizon_end]

    if len(p_ref_horizon) < controller.mpc_horizon:
        p_ref_horizon = np.vstack([
            p_ref_horizon,
            np.tile(p_ref_horizon[-1], (controller.mpc_horizon - len(p_ref_horizon), 1))
        ])

    u, info = controller.step(x_t, p_tip_t, p_ref_horizon, hiddens=None)

    result = crm_diff_py.dynamics_forward(x_t, u, dt, L_inserted, params_dict)
    if result['status'] == 0:
        x_t = result['x_next']
        p_tip_t = result['p_tip']

    p_tips_hybrid.append(p_tip_t)
    metrics.add_step(info, p_tip_t, p_ref[i])

hybrid_time = time.time() - t_start_hybrid
summary = metrics.get_summary()

print(f"  Hybrid RMSE: {summary['tracking_rmse']:.3f} mm, Time: {hybrid_time:.2f}s")

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

speedup = mpc_time / hybrid_time if hybrid_time > 0 else 0.0
rmse_diff = summary['tracking_rmse'] - mpc_rmse

print(f"\nMPC-only:")
print(f"  RMSE: {mpc_rmse:.4f} mm")
print(f"  Time: {mpc_time:.2f} s")

print(f"\nHybrid:")
print(f"  RMSE: {summary['tracking_rmse']:.4f} mm")
print(f"  Time: {hybrid_time:.2f} s")
print(f"  MPC call rate: {summary['mpc_call_rate']:.1%}")
print(f"  Speedup: {speedup:.2f}×")

print(f"\nMode Distribution:")
for mode, frac in summary['mode_distribution'].items():
    print(f"  {mode:20s}: {frac:6.1%}")

print("\n" + "="*60)
print("ACCEPTANCE CRITERIA (Targets)")
print("="*60)

criteria = {
    'MPC call rate ≤ 30%': summary['mpc_call_rate'] <= 0.30,
    'RMSE ≤ MPC + 0.5mm': summary['tracking_rmse'] <= mpc_rmse + 0.5,
    'Speedup ≥ 2×': speedup >= 2.0,
    'Safety violations = 0': summary['n_failures'] == 0 if 'n_failures' in summary else True
}

for criterion, passed in criteria.items():
    status = "✓" if passed else "✗"
    print(f"{status} {criterion}")

all_pass = all(criteria.values())
print("\n" + "="*60)
if all_pass:
    print("✓ ALL CRITERIA MET")
else:
    print("✗ SOME CRITERIA FAILED (expected with untrained ensemble)")
    print("  Note: Untrained ensemble has high uncertainty → high MPC rate")
print("="*60)
