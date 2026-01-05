"""
Quick smoke test for FULLSTATE linearization and VJP.
Tests the most critical functionality without heavy dependencies.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np

def test_binding_exists():
    """Test that true_legacy_linearize binding exists."""
    print("Test 1: Checking C++ binding exists...")
    import crm_diff_py
    
    assert hasattr(crm_diff_py, 'true_legacy_linearize'), "Missing true_legacy_linearize binding"
    assert hasattr(crm_diff_py, 'true_legacy_step_forward'), "Missing true_legacy_step_forward binding"
    assert hasattr(crm_diff_py, 'true_legacy_step_vjp'), "Missing true_legacy_step_vjp binding"
    assert hasattr(crm_diff_py, 'true_legacy_step_vjp_batched'), "Missing true_legacy_step_vjp_batched binding"
    
    print("  PASS: All bindings exist")

def test_python_wrapper():
    """Test that Python wrapper imports correctly."""
    print("Test 2: Checking Python wrappers...")
    
    from control.true_legacy_step import true_legacy_linearize
    from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim
    
    # Check that method="implicit" is the default
    import inspect
    sig = inspect.signature(true_legacy_linearize)
    assert sig.parameters['method'].default == "implicit", "Default method should be 'implicit'"
    
    print("  PASS: Python wrappers import correctly with implicit as default")

def test_controller_defaults():
    """Test that controllers use jacobian_mode='implicit'."""
    print("Test 3: Checking controller defaults...")
    
    from control.ilqr import iLQRSolver
    import inspect
    
    sig = inspect.signature(iLQRSolver.__init__)
    assert sig.parameters['jacobian_mode'].default == "implicit", "iLQR should default to implicit"
    
    print("  PASS: Controllers default to jacobian_mode='implicit'")

if __name__ == "__main__":
    try:
        print("="*60)
        print("FULLSTATE Quick Smoke Tests")
        print("="*60)
        print()
        
        test_binding_exists()
        test_python_wrapper()
        test_controller_defaults()
        
        print()
        print("="*60)
        print("ALL QUICK SMOKE TESTS PASSED")
        print("="*60)
        sys.exit(0)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
