"""Compare batched vs single VJP flags."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import crm_diff_py

# Load params
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

# Simple state
x_coil = np.zeros((1, 18), dtype=np.float64)
xf = np.zeros(15, dtype=np.float64)
R_identity = np.eye(3).flatten()
x_coil[0, 9:18] = R_identity
xf[3:12] = R_identity
x_coil[0, 6:9] = [0.0, 0.0, 1.0]
xf[:3] = [0.0, 0.0, 10.0]
u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

# Upstream gradient
grad_tip_p = np.array([1.0, 0.5, 0.2], dtype=np.float64)

print("="*80)
print("BATCHED VJP (num_rhs=1)")
print("="*80)
result_batch = crm_diff_py.true_legacy_step_vjp_batched(
    x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p.reshape(1, 3)
)
grad_xf_batch = result_batch['grad_xf'][0]  # Extract first RHS
print(f"grad_xf shape: {grad_xf_batch.shape}")
print(f"grad_xf flags: {grad_xf_batch.flags}")
print(f"grad_xf contents: {grad_xf_batch}")

print("\n" + "="*80)
print("SINGLE VJP")
print("="*80)
result_single = crm_diff_py.true_legacy_step_vjp(
    x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p
)
grad_xf_single = result_single['grad_xf']
print(f"grad_xf shape: {grad_xf_single.shape}")
print(f"grad_xf flags: {grad_xf_single.flags}")
print(f"grad_xf contents: {grad_xf_single}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Batched: {grad_xf_batch}")
print(f"Single:  {grad_xf_single}")
print(f"Match: {np.allclose(grad_xf_batch, grad_xf_single)}")
