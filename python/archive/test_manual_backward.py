import sys
import os
import numpy as np
from numpy.linalg import solve

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
fwd_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

# Extract Jacobians
J_p_u0 = np.array(fwd_result['J_p_u0'])  # (3, 3)
J_u_u0 = np.array(fwd_result['J_u_u0'])  # (3, 3)
J_p_zc = np.array(fwd_result['J_p_zc'])  # (3, 3)
J_u_zc = np.array(fwd_result['J_u_zc'])  # (3, 3)
K_tip = np.array(fwd_result['K_tip'])    # (3, 3)

print("Cached Jacobians:")
print(f"J_p_u0 =\n{J_p_u0}\n")
print(f"J_u_u0 =\n{J_u_u0}\n")
print(f"J_p_zc =\n{J_p_zc}\n")
print(f"J_u_zc =\n{J_u_zc}\n")
print(f"K_tip =\n{K_tip}\n")

# Manual computation of backward pass for grad_p_tip = [1, 0, 0]
grad_p_tip = np.array([1.0, 0.0, 0.0])

print("="*60)
print("Manual backward pass computation:")
print("="*60)
print(f"grad_p_tip = {grad_p_tip}\n")

# Step 1: Compute K_tip * J_u_u0
K_J_u = K_tip @ J_u_u0
print(f"Step 1: K_J_u = K_tip @ J_u_u0 =\n{K_J_u}\n")

# Step 2: Compute RHS = J_p_u0^T * grad_p_tip
rhs = J_p_u0.T @ grad_p_tip
print(f"Step 2: rhs = J_p_u0.T @ grad_p_tip = {rhs}\n")

# Step 3: Solve (K_J_u)^T * lambda = rhs
A = K_J_u.T
print(f"Step 3: Solving A^T * lambda = rhs")
print(f"  A.T =\n{A}\n")

lambda_sol = solve(A, rhs)
print(f"  lambda = {lambda_sol}\n")

# Verify solution
residual = A @ lambda_sol - rhs
rel_residual = np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0)
print(f"  Residual check: ||A @ lambda - rhs|| / ||rhs|| = {rel_residual:.6e}\n")

# Step 4: Compute K_tip * lambda
K_lambda = K_tip @ lambda_sol
print(f"Step 4: K_lambda = K_tip @ lambda = {K_lambda}\n")

# Step 5: Compute grad_u
grad_u_manual = J_p_zc.T @ grad_p_tip - J_u_zc.T @ K_lambda
print(f"Step 5: grad_u = J_p_zc.T @ grad_p_tip - J_u_zc.T @ K_lambda")
print(f"  J_p_zc.T @ grad_p_tip = {J_p_zc.T @ grad_p_tip}")
print(f"  J_u_zc.T @ K_lambda = {J_u_zc.T @ K_lambda}")
print(f"  grad_u (manual) = {grad_u_manual}\n")

# Compare with C++ result
bwd_result = crm_diff_py.equilibrium_backward(fwd_result, grad_p_tip)
grad_u_cpp = bwd_result['grad_u']

print("="*60)
print("Comparison:")
print("="*60)
print(f"grad_u (C++):    {grad_u_cpp}")
print(f"grad_u (manual): {grad_u_manual}")
print(f"Difference:      {grad_u_cpp - grad_u_manual}")
print(f"Max abs diff:    {np.max(np.abs(grad_u_cpp - grad_u_manual)):.6e}")
