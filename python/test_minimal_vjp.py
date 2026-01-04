"""Minimal test to isolate grad_xf corruption issue."""
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

print("Calling crm_diff_py.true_legacy_step_vjp...")
print(f"Input grad_tip_p: {grad_tip_p}")

result = crm_diff_py.true_legacy_step_vjp(
    x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p
)

print(f"\nPython received result with keys: {list(result.keys())}")

# Check if maybe the dict got corrupted
print(f"result type: {type(result)}")
print(f"result id: {id(result)}")

# Extract immediately
grad_xf_arr = result['grad_xf']
print(f"\ngrad_xf type: {type(grad_xf_arr)}")
print(f"grad_xf dtype: {grad_xf_arr.dtype}")
print(f"grad_xf shape: {grad_xf_arr.shape}")
print(f"grad_xf flags: {grad_xf_arr.flags}")
print(f"grad_xf data pointer (ctypes): {grad_xf_arr.ctypes.data:#x}")

# Print as bytes
import struct
print(f"\nFirst 3 values as raw bytes:")
for i in range(3):
    bytes_val = grad_xf_arr.tobytes()[i*8:(i+1)*8]
    float_val = struct.unpack('d', bytes_val)[0]
    print(f"  [{i}] bytes: {bytes_val.hex()}, float: {float_val}")

print(f"\ngrad_xf contents: {grad_xf_arr}")
print(f"Expected: [1.0, 0.5, 0.2, 0, 0, ..., 0]")
