"""
CP4.0: Canonical Configuration Module

This module provides DEFAULT parameter values for reference and documentation.

IMPORTANT: Tests should NOT import defaults from this module. Instead, tests
must EXPLICITLY pass parameter values to ensure reproducibility and clarity.

This module serves as:
1. Documentation of standard/recommended values
2. Reference for what parameters mean
3. Single source of truth for typical operating ranges
"""

# ============================================================================
# Default Physical Parameters
# ============================================================================

# Default timestep for dynamics integration (seconds)
# Used in: dynamics_forward, MPC, iLQR, trajectory rollouts
# Typical range: 0.001 - 0.1 s
# Common values:
#   - 0.01 s (10 ms): Standard for control simulations
#   - 0.05 s (50 ms): Used in some NPZ datasets
DEFAULT_DT = 0.01

# Default catheter insertion length (mm)
# Used in: equilibrium_forward, dynamics_forward
# Typical range: 10.0 - 150.0 mm
# Common values:
#   - 50.0 mm: Short insertion, used in many tests
#   - 94.3 mm: Standard insertion, used in NPZ datasets
DEFAULT_L_INSERTED = 50.0

# Default integration step size for nonlinear solver
# Used in: equilibrium_forward, dynamics_forward (via params_dict)
# Typical range: 0.1 - 1.0
# Common values:
#   - 0.5: Standard, good balance of accuracy/speed
#   - 0.2: Higher accuracy, used in some NPZ datasets
DEFAULT_INTEGRATION_STEP_SIZE = 0.5

# ============================================================================
# NPZ Dataset Standard Parameters
# ============================================================================

# Parameters used in recorded NPZ datasets
# These are DIFFERENT from test defaults above
NPZ_DATASET_PARAMS = {
    'dt': 0.05,                      # 50 ms timestep
    'L_inserted': 94.3,              # Standard insertion length
    'integration_step_size': 0.2,    # Higher accuracy for dataset generation
}

# ============================================================================
# Parameter Validation Ranges
# ============================================================================

VALID_RANGES = {
    'dt': (0.0001, 1.0),                      # 0.1ms to 1s
    'L_inserted': (1.0, 200.0),               # 1mm to 200mm
    'integration_step_size': (0.01, 2.0),     # Very fine to coarse
}

# ============================================================================
# Helper Functions
# ============================================================================

def validate_params(dt=None, L_inserted=None, integration_step_size=None):
    """
    Validate that parameters are within acceptable ranges.

    Args:
        dt: Timestep (seconds), optional
        L_inserted: Insertion length (mm), optional
        integration_step_size: Integration step size, optional

    Raises:
        ValueError: If any parameter is out of range
    """
    if dt is not None:
        min_dt, max_dt = VALID_RANGES['dt']
        if not (min_dt <= dt <= max_dt):
            raise ValueError(f"dt={dt} out of valid range [{min_dt}, {max_dt}]")

    if L_inserted is not None:
        min_L, max_L = VALID_RANGES['L_inserted']
        if not (min_L <= L_inserted <= max_L):
            raise ValueError(f"L_inserted={L_inserted} out of valid range [{min_L}, {max_L}]")

    if integration_step_size is not None:
        min_h, max_h = VALID_RANGES['integration_step_size']
        if not (min_h <= integration_step_size <= max_h):
            raise ValueError(
                f"integration_step_size={integration_step_size} "
                f"out of valid range [{min_h}, {max_h}]"
            )


def get_default_params_dict(cath_params_file=None, cath_config_file=None):
    """
    Create a params_dict with default values.

    This is a convenience function for interactive use and scripts.
    Tests should create params_dict explicitly with their own values.

    Args:
        cath_params_file: Path to catheter parameter file
        cath_config_file: Path to catheter configuration file

    Returns:
        params_dict: Dict with default parameters
    """
    import crm_diff_py

    # Use standard files if not specified
    if cath_params_file is None:
        cath_params_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    if cath_config_file is None:
        cath_config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"

    cath_params = crm_diff_py.load_cath_params(cath_params_file)
    cath_config = crm_diff_py.load_cath_config(cath_config_file)

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': DEFAULT_INTEGRATION_STEP_SIZE,
        'FinalValueOnly': True,
        'L_inserted': DEFAULT_L_INSERTED,
    }

    return params_dict


def print_param_summary(dt, L_inserted, integration_step_size):
    """
    Print a one-line parameter summary.

    Args:
        dt: Timestep (seconds)
        L_inserted: Insertion length (mm)
        integration_step_size: Integration step size

    Returns:
        str: One-line summary string
    """
    return f"PARAMS: dt={dt:.4f}s, L={L_inserted:.1f}mm, h={integration_step_size:.2f}"
