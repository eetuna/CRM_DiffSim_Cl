"""
CP1.8: Replay Failing Cases

Loads saved failure cases from CP1.6/CP1.7 and replays them deterministically
to help debug gradient computation issues.
"""
import sys
import os
import json
import numpy as np
import argparse

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py


def replay_cp16_case(failure_file):
    """
    Replay a CP1.6 (finite difference) failure case.
    """
    print("="*70)
    print(f"Replaying CP1.6 failure: {failure_file}")
    print("="*70)

    # Load failure data
    with open(failure_file, 'r') as f:
        data = json.load(f)

    print(f"\nOriginal failure timestamp: {data['timestamp']}")
    print(f"Test name: {data['test_name']}")
    print(f"u = {data['u']}")
    print(f"L_inserted = {data['L_inserted']}")
    print(f"Original p_tip = {data['p_tip']}")

    # Load parameters (same as in test)
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

    # Replay forward pass
    u = np.array(data['u'], dtype=np.float64)
    L_inserted = data['L_inserted']

    print("\nReplaying forward pass...")
    fwd_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

    print(f"Replayed p_tip = {fwd_result['p_tip']}")
    print(f"Replayed deltau0 = {fwd_result['deltau0']}")
    print(f"Status = {fwd_result['status']}")

    # Check if replay matches original
    p_tip_match = np.allclose(fwd_result['p_tip'], data['p_tip'], atol=1e-14)
    print(f"\nReplay matches original p_tip: {p_tip_match}")

    # Display Jacobian norms
    print("\nJacobian norms:")
    print(f"  ||J_p_u0|| = {data.get('J_p_u0_norm', 'N/A')}")
    print(f"  ||J_u_u0|| = {data.get('J_u_u0_norm', 'N/A')}")

    # Display K_tip diagonal
    if 'K_tip_diag' in data:
        print(f"\nK_tip diagonal: {data['K_tip_diag']}")

    # Display solver diagnostics
    if 'lu_rank' in data and data['lu_rank'] >= 0:
        print(f"\nSolver diagnostics:")
        print(f"  LU rank = {data['lu_rank']}")
        print(f"  Relative residual = {data.get('rel_solve_residual', 'N/A')}")

    # Display error analysis
    print("\nOriginal FD vs Analytical comparison:")
    print(f"  Max absolute error: {data['max_abs_err']:.6e}")
    print(f"  Max relative error: {data['max_rel_err']:.6e}")

    if 'J_fd' in data and 'J_auto' in data:
        J_fd = np.array(data['J_fd'])
        J_auto = np.array(data['J_auto'])

        print(f"\nFinite Difference Jacobian:")
        print(J_fd)
        print(f"\nAnalytical Jacobian:")
        print(J_auto)
        print(f"\nDifference:")
        print(J_auto - J_fd)

    # Replay backward pass for each output component
    print("\n" + "-"*70)
    print("Replaying backward passes...")
    print("-"*70)

    for i in range(3):
        grad_p_tip = np.zeros(3, dtype=np.float64)
        grad_p_tip[i] = 1.0

        bwd_result = crm_diff_py.equilibrium_backward(fwd_result, grad_p_tip)

        print(f"\nBackward pass {i} (grad_p_tip=[{grad_p_tip[0]}, {grad_p_tip[1]}, {grad_p_tip[2]}]):")
        print(f"  grad_u = {bwd_result['grad_u']}")
        print(f"  status = {bwd_result['status']}")
        print(f"  lu_rank = {bwd_result['lu_rank']}")
        print(f"  rel_residual = {bwd_result['rel_residual']:.6e}")

    print("\n" + "="*70)
    print("Replay complete")
    print("="*70)


def replay_cp17_case(failure_file):
    """
    Replay a CP1.7 (gradcheck) failure case.
    """
    print("="*70)
    print(f"Replaying CP1.7 failure: {failure_file}")
    print("="*70)

    # Load failure data
    with open(failure_file, 'r') as f:
        data = json.load(f)

    print(f"\nOriginal failure timestamp: {data['timestamp']}")
    print(f"Test name: {data['test_name']}")
    print(f"u = {data['u']}")
    print(f"L_inserted = {data['L_inserted']}")
    print(f"Original p_tip = {data['p_tip']}")
    print(f"gradcheck_passed = {data['gradcheck_passed']}")

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

    # Replay forward pass
    u = np.array(data['u'], dtype=np.float64)
    L_inserted = data['L_inserted']

    print("\nReplaying forward pass...")
    fwd_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

    print(f"Replayed p_tip = {fwd_result['p_tip']}")
    print(f"Status = {fwd_result['status']}")

    # Check if replay matches
    p_tip_match = np.allclose(fwd_result['p_tip'], data['p_tip'], atol=1e-14)
    print(f"\nReplay matches original p_tip: {p_tip_match}")

    print("\n" + "="*70)
    print("Replay complete")
    print("="*70)
    print("\nTo debug gradcheck failure, re-run test_cp17_gradcheck.py with:")
    print(f"  python test_cp17_gradcheck.py")


def list_failure_cases():
    """List all saved failure cases."""
    import glob

    cp16_files = glob.glob("failure_cp16_*.json")
    cp17_files = glob.glob("failure_cp17_*.json")

    if not cp16_files and not cp17_files:
        print("No failure cases found.")
        return

    if cp16_files:
        print("\nCP1.6 (Finite Difference) failures:")
        for f in sorted(cp16_files):
            with open(f, 'r') as fp:
                data = json.load(fp)
            print(f"  {f}")
            print(f"    Test: {data['test_name']}")
            print(f"    Time: {data['timestamp']}")
            print(f"    Max abs err: {data['max_abs_err']:.6e}")
            print(f"    Max rel err: {data['max_rel_err']:.6e}")

    if cp17_files:
        print("\nCP1.7 (Gradcheck) failures:")
        for f in sorted(cp17_files):
            with open(f, 'r') as fp:
                data = json.load(fp)
            print(f"  {f}")
            print(f"    Test: {data['test_name']}")
            print(f"    Time: {data['timestamp']}")


def main():
    parser = argparse.ArgumentParser(description="Replay failing gradient test cases")
    parser.add_argument('--file', type=str, help='Failure case JSON file to replay')
    parser.add_argument('--list', action='store_true', help='List all failure cases')

    args = parser.parse_args()

    if args.list:
        list_failure_cases()
        return 0

    if not args.file:
        print("Error: Must specify --file or --list")
        parser.print_help()
        return 1

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1

    # Determine which type of failure based on filename
    if 'cp16' in args.file:
        replay_cp16_case(args.file)
    elif 'cp17' in args.file:
        replay_cp17_case(args.file)
    else:
        print(f"Error: Unknown failure type (expected 'cp16' or 'cp17' in filename)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
