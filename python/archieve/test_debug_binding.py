import sys
import os
import numpy as np

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py

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

# Smoke case: u = [0, 0, 0], L_inserted = 50.0
u = np.array([0.0, 0.0, 0.0], dtype=np.float64)
L_inserted = 50.0

result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

print("p_tip:")
print(f"  value: {result['p_tip']}")
print(f"  shape: {result['p_tip'].shape}")
print(f"  dtype: {result['p_tip'].dtype}")
print(f"  strides: {result['p_tip'].strides}")
print(f"  flags: {result['p_tip'].flags}")
print()

print("deltau0:")
print(f"  value: {result['deltau0']}")
print(f"  shape: {result['deltau0'].shape}")
print(f"  dtype: {result['deltau0'].dtype}")
print(f"  flags: {result['deltau0'].flags}")
print()

print("J_p_u0:")
print(f"  shape: {result['J_p_u0'].shape}")
print(f"  dtype: {result['J_p_u0'].dtype}")
print(f"  flags: {result['J_p_u0'].flags}")
print(f"  value:\n{result['J_p_u0']}")
