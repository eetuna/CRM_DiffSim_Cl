"""Debug script to check Jacobians"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import crm_equilibrium

# Load catheter parameters
cath_params_file = "../catheterdata/CatheterParameterSet_1_new.txt"
cath_config_file = "../catheterdata/CatheterSpatialConfiguration_1.txt"

params_dict = crm_equilibrium.load_default_catheter_params(
    cath_params_file, cath_config_file
)

# Test at zero currents
u = np.array([0.0, 0.0, 0.0], dtype=np.float64)
L_inserted = 50.0

# Call forward to get Jacobians
import crm_diff_py
result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

print("Forward result:")
print(f"  status: {result['status']}")
print(f"  p_tip: {result['p_tip']}")
print(f"  deltau0: {result['deltau0']}")

print("\nCached Jacobians:")
print(f"  J_p_u0 shape: {result['J_p_u0'].shape}")
J_p_u0 = result['J_p_u0'].reshape(3, 3)
print(f"  J_p_u0:\n{J_p_u0}")

print(f"\n  J_u_u0 shape: {result['J_u_u0'].shape}")
J_u_u0 = result['J_u_u0'].reshape(3, 3)
print(f"  J_u_u0:\n{J_u_u0}")

print(f"\n  K_tip shape: {result['K_tip'].shape}")
K_tip = result['K_tip'].reshape(3, 3)
print(f"  K_tip:\n{K_tip}")

print(f"\n  J_p_zc shape: {result['J_p_zc'].shape}")
J_p_zc = result['J_p_zc'].reshape(3, 3)
print(f"  J_p_zc:\n{J_p_zc}")

print(f"\n  J_u_zc shape: {result['J_u_zc'].shape}")
J_u_zc = result['J_u_zc'].reshape(3, 3)
print(f"  J_u_zc:\n{J_u_zc}")

# Compute K_tip * J_u_u0
K_J_u = K_tip @ J_u_u0
print(f"\n  K_tip * J_u_u0:\n{K_J_u}")
print(f"  rank(K_tip * J_u_u0): {np.linalg.matrix_rank(K_J_u)}")
print(f"  rank((K_tip * J_u_u0)^T): {np.linalg.matrix_rank(K_J_u.T)}")

# Check singular values
U, s, Vt = np.linalg.svd(K_J_u.T)
print(f"\n  Singular values of (K_tip * J_u_u0)^T: {s}")
