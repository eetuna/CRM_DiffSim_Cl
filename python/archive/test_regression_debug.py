"""Debug test to check reference harness vs single Python step."""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import crm_diff_py
from control.true_legacy_step import true_legacy_step
from control.true_legacy_state_adapter import pack_true_legacy_state, unpack_true_legacy_state
from control.step_legacy_contract import load_default_catheter_params

# Parameters
CATH_PARAMS_FILE = "./catheterdata/CatheterParameterSet_1_dyn.txt"
CATH_CONFIG_FILE = "./catheterdata/CatheterSpatialConfiguration_1.txt"
INSERTED_LENGTH = 94.3
DT = 0.05
INTEGRATION_STEP_SIZE = 0.2

# Initial state
XF_INITIAL = np.array([
    -0.458414144062750, 34.411241976876518, 70.457561147732264,
    0.999932718178103, 0.009921777042635, -0.006009780134551,
    -0.004651734390922, 0.817579117734723, 0.575797488368325,
    0.010626405041486, -0.575730791763330, 0.817570262993625,
    -0.015378744286498, 0.000001280646594, -0.000349413951059
], dtype=np.float64)

X_COIL_INITIAL = np.array([[
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    -0.248418562587657, 17.707660318406560, 46.752162601547091,
    0.999919687839427, 0.009924211584043, -0.007882125064742,
    -0.003571217614502, 0.817374079004311, 0.576096225796181,
    0.012159945552960, -0.576021809479719, 0.817343875445250,
]], dtype=np.float64)

ACTUATION = np.array([[[0.0, 0.0, 0.1]]], dtype=np.float64)
DAMPING = np.array([[12.1761626666366, 12.1761626666366, 284.429938756989,
                     0.0304776127617393, 0.0304776127617393, 0.00502712804532508]], dtype=np.float64)
ACT_INERTIA = np.array([[3.1929e-05, 0.0, 0.0, 0.0, 3.1929e-05, 0.0, 0.0, 0.0, 2.9287e-05]], dtype=np.float64)

print("Testing single step...")
params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)
params['L_inserted'] = INSERTED_LENGTH
params['IntegrationStepSize'] = INTEGRATION_STEP_SIZE
params['damping'] = DAMPING
params['ActInertia'] = ACT_INERTIA

# Python step
x_coil_torch = torch.from_numpy(X_COIL_INITIAL).to(dtype=torch.float64)
xf_torch = torch.from_numpy(XF_INITIAL).to(dtype=torch.float64)
x_py = pack_true_legacy_state(x_coil_torch, xf_torch)
u_torch = torch.from_numpy(ACTUATION[0]).to(dtype=torch.float64)

print(f"Initial state packed shape: {x_py.shape}")
print(f"Initial tip position (from xf): {XF_INITIAL[:3]}")
print(f"Initial coil position: {X_COIL_INITIAL[0, 6:9]}")

x_next, obs = true_legacy_step(x_py, u_torch, DT, n_act=1, catheter_params=params)

print(f"\nPython step result:")
print(f"  tip_p: {obs['tip_p'].numpy()}")
print(f"  converged: {obs['converged']}")

x_coil_next, xf_next = unpack_true_legacy_state(x_next, 1)
print(f"  xf_next[:3] (tip pos): {xf_next[:3].numpy()}")
print(f"  xf_next (full): {xf_next.numpy()}")

# Reference step
u_seq = np.tile(ACTUATION, (1, 1, 1))
print(f"\nReference harness (1 step):")
ref_result = crm_diff_py.crmdyn_reference_rollout(
    X_COIL_INITIAL, XF_INITIAL, u_seq, DT, params, use_warmstart=True
)

print(f"  P_tip_traj[0]: {ref_result['P_tip_traj'][0]}")
print(f"  P_tip_traj[1]: {ref_result['P_tip_traj'][1]}")
print(f"  X_traj[1][:3]: {ref_result['X_traj'][1, :3]}")
print(f"  X_traj[1] (full): {ref_result['X_traj'][1]}")

print(f"\nComparison:")
print(f"  Diff in tip_p: {np.abs(obs['tip_p'].numpy() - ref_result['P_tip_traj'][1])}")
print(f"  Diff in xf[:3]: {np.abs(xf_next[:3].numpy() - ref_result['X_traj'][1, :3])}")
