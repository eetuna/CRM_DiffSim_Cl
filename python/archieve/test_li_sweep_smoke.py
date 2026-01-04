"""
Li Sweep Smoke Test: Compare Python binding vs C++ harness

Tests a subset of Li values to ensure Python binding matches C++ exactly.
"""
import sys
import os
import subprocess
import csv
import numpy as np

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py


def run_cpp_harness(mode="--straight-only"):
    """Run C++ harness and return results."""
    harness_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'test_li_sweep')

    if not os.path.exists(harness_path):
        raise FileNotFoundError(f"C++ harness not found: {harness_path}")

    # Run harness
    cwd = os.path.join(os.path.dirname(__file__), '..')
    result = subprocess.run(
        [harness_path, mode],
        capture_output=True,
        text=True,
        cwd=cwd
    )

    if result.returncode != 0:
        print("C++ harness stderr:", result.stderr)
        raise RuntimeError(f"C++ harness failed with return code {result.returncode}")

    # Read CSV output
    csv_file = os.path.join(cwd, "li_sweep_straight.csv" if "straight" in mode else "li_sweep_baseline.csv")

    results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Li = float(row['Li'])
            results[Li] = {
                'status': int(row['status']),
                'p_tip': np.array([
                    float(row['p_tip_x']),
                    float(row['p_tip_y']),
                    float(row['p_tip_z'])
                ]),
                'deltau0': np.array([
                    float(row['deltau0_x']),
                    float(row['deltau0_y']),
                    float(row['deltau0_z'])
                ]),
                'z_error': float(row['z_error'])
            }

    return results


def run_python_binding(Li_values, straight_mode=True):
    """Run Python binding for given Li values."""
    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"

    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    # If straight mode, we need to override ustar and gravity
    # NOTE: Python binding doesn't support modifying loaded params directly,
    # so this test only works for baseline mode OR we need to modify the files
    # For now, we'll assume the C++ harness creates the reference

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    # Actuation: u = [0, 0, 0]
    u = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    results = {}
    for Li in Li_values:
        result = crm_diff_py.equilibrium_forward(u, Li, params_dict)

        results[Li] = {
            'status': result['status'],
            'p_tip': result['p_tip'],
            'deltau0': result['deltau0']
        }

    return results


def compare_results(cpp_results, py_results, tol=1e-12):
    """Compare C++ and Python results."""
    all_pass = True

    for Li in sorted(cpp_results.keys()):
        if Li not in py_results:
            print(f"✗ Li={Li}: Missing in Python results")
            all_pass = False
            continue

        cpp = cpp_results[Li]
        py = py_results[Li]

        # Compare status
        if cpp['status'] != py['status']:
            print(f"✗ Li={Li}: status mismatch (C++={cpp['status']}, Py={py['status']})")
            all_pass = False
            continue

        # Compare p_tip
        p_tip_diff = np.abs(cpp['p_tip'] - py['p_tip'])
        p_tip_max_diff = np.max(p_tip_diff)

        if p_tip_max_diff > tol:
            print(f"✗ Li={Li}: p_tip mismatch (max_diff={p_tip_max_diff:.6e})")
            print(f"    C++: {cpp['p_tip']}")
            print(f"    Py:  {py['p_tip']}")
            all_pass = False
            continue

        # Compare deltau0
        deltau0_diff = np.abs(cpp['deltau0'] - py['deltau0'])
        deltau0_max_diff = np.max(deltau0_diff)

        if deltau0_max_diff > tol:
            print(f"✗ Li={Li}: deltau0 mismatch (max_diff={deltau0_max_diff:.6e})")
            print(f"    C++: {cpp['deltau0']}")
            print(f"    Py:  {py['deltau0']}")
            all_pass = False
            continue

        # Pass
        print(f"✓ Li={Li:6.1f}: p_tip matches (max_diff={p_tip_max_diff:.6e})")

    return all_pass


def main():
    print("="*70)
    print("Li Sweep Smoke Test: Python vs C++ Comparison")
    print("="*70)
    print()

    # Test subset of Li values
    # NOTE: We can only test baseline mode because Python binding doesn't support
    # runtime override of ustar/gravity (those are baked into CathParams/CathConfig)
    # The straight mode would require separate parameter files with ustar=0, g=0

    print("[1/3] Running C++ harness (baseline mode)...")
    cpp_results = run_cpp_harness("--baseline-only")
    print(f"  Loaded {len(cpp_results)} C++ reference points")
    print()

    # Select subset for Python test (to save time)
    test_Li_values = [0.0, 50.0, 100.0]
    print(f"[2/3] Running Python binding for Li = {test_Li_values}...")

    py_results = run_python_binding(test_Li_values, straight_mode=False)
    print(f"  Computed {len(py_results)} Python results")
    print()

    print("[3/3] Comparing results...")
    print()

    # Filter C++ results to only test_Li_values
    cpp_subset = {Li: cpp_results[Li] for Li in test_Li_values if Li in cpp_results}

    all_pass = compare_results(cpp_subset, py_results, tol=1e-12)

    print()
    print("="*70)
    print("SMOKE TEST SUMMARY")
    print("="*70)

    if all_pass:
        print("✓ PASS: Python binding matches C++ harness exactly")
        print()
        print("Tested Li values:", test_Li_values)
        print("Tolerance: 1e-12")
        return 0
    else:
        print("✗ FAIL: Python binding does NOT match C++ harness")
        return 1


if __name__ == "__main__":
    sys.exit(main())
