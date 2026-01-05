"""
Sanity gate test: Ensure NO reduced6d (crm_diff_py.dynamics_*) imports in default stack.

This test verifies that the FULLSTATE migration is complete and no old reduced6d
dynamics functions are referenced in the default code paths.
"""

import subprocess
import sys

def test_no_reduced6d_in_default_stack():
    """Ensure NO imports of crm_diff_py.dynamics_* in default stack CODE."""

    # Search for crm_diff_py.dynamics_ references in CODE ONLY (not docs)
    result = subprocess.run(
        [
            'grep', '-r', '-n', 'crm_diff_py\\.dynamics_',
            'python/control', 'src',
            '--include=*.py', '--include=*.cpp'
        ],
        capture_output=True,
        text=True,
        cwd='/workspaces/CRM_DiffSim_Cl'
    )

    # Filter out:
    # 1. reduced6d/ subdirectory (allowed for backward compat)
    # 2. Comments
    # 3. Docstrings
    lines = []
    for l in result.stdout.split('\n'):
        if not l.strip():
            continue
        if 'reduced6d' in l:
            continue
        # Filter out comments (lines with # before the match)
        if '#' in l and l.index('#') < l.index('dynamics_'):
            continue
        lines.append(l)

    if lines:
        print("FAILED: Found crm_diff_py.dynamics_* references in default stack CODE:")
        for line in lines:
            print(f"  {line}")
        sys.exit(1)

    print("PASS: Zero hits for crm_diff_py.dynamics_* in default stack code")
    print(f"(Excluded reduced6d/ directory and documentation from check)")

if __name__ == "__main__":
    test_no_reduced6d_in_default_stack()
