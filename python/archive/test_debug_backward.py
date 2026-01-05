import sys
import os
import numpy as np

build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py

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

# Test case
u = np.array([0.0, 0.0, 0.0], dtype=np.float64)
L_inserted = 50.0

# Forward pass
print("Forward pass:")
fwd_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)
print(f"  p_tip = {fwd_result['p_tip']}")
print(f"  status = {fwd_result['status']}")

print("\nCached Jacobians:")
print(f"  J_p_u0 shape = {fwd_result['J_p_u0'].shape}")
print(f"  J_u_u0 shape = {fwd_result['J_u_u0'].shape}")
print(f"  J_p_zc shape = {fwd_result['J_p_zc'].shape}")
print(f"  J_u_zc shape = {fwd_result['J_u_zc'].shape}")
print(f"  K_tip shape = {fwd_result['K_tip'].shape}")

print(f"\nJ_p_u0 =")
print(fwd_result['J_p_u0'])

print(f"\nJ_p_zc =")
print(fwd_result['J_p_zc'])

print(f"\nK_tip =")
print(fwd_result['K_tip'])

# Backward pass for each component
print("\n" + "="*60)
print("Backward passes:")
print("="*60)

for i in range(3):
    grad_p_tip = np.zeros(3, dtype=np.float64)
    grad_p_tip[i] = 1.0

    print(f"\nBackward pass {i}: grad_p_tip = {grad_p_tip}")
    bwd_result = crm_diff_py.equilibrium_backward(fwd_result, grad_p_tip)

    print(f"  grad_u = {bwd_result['grad_u']}")
    print(f"  status = {bwd_result['status']}")
    print(f"  lu_rank = {bwd_result['lu_rank']}")
    print(f"  rel_residual = {bwd_result['rel_residual']:.6e}")
