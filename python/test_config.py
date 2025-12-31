"""Test configuration loading"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import crm_diff_py

# Load parameters
cath_params = crm_diff_py.load_cath_params("../catheterdata/CatheterParameterSet_1_new.txt")
cath_config = crm_diff_py.load_cath_config("../catheterdata/CatheterSpatialConfiguration_1.txt")

print(f"Loaded CathParams: {cath_params}")
print(f"Loaded CathConfig: {cath_config}")
print(f"Type of CathParams: {type(cath_params)}")
print(f"Type of CathConfig: {type(cath_config)}")

# Try to access fields (if exposed)
print(f"\nTrying to access CathConfig fields...")
print(f"dir(cath_config) = {dir(cath_config)}")
