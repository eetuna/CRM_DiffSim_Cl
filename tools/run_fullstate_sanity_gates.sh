#!/bin/bash
# Robust FULLSTATE Sanity Gates - NO early exit
# This script runs all gates and reports results without dying early

WORKSPACE="/workspaces/CRM_DiffSim_Cl"
cd "$WORKSPACE" || exit 1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Results tracking
GATE_RESULTS=()
ALL_PASSED=0

echo "=============================================="
echo "FULLSTATE Sanity Gates - Comprehensive Check"
echo "=============================================="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "=============================================="
echo ""

# Helper function to record gate result
record_gate() {
    local gate_num=$1
    local gate_name=$2
    local exit_code=$3

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} Gate $gate_num: $gate_name"
        GATE_RESULTS+=("PASS")
    else
        echo -e "${RED}[FAIL]${NC} Gate $gate_num: $gate_name"
        GATE_RESULTS+=("FAIL")
        ALL_PASSED=1
    fi
}

#############################################
# Gate 0 - Build
#############################################
echo "=== Gate 0: Build ==="
echo "Running: cmake -S . -B build"
cmake -S . -B build > /tmp/gate0_cmake.log 2>&1
CMAKE_EXIT=$?
echo "CMake exit code: $CMAKE_EXIT"
if [ $CMAKE_EXIT -ne 0 ]; then
    echo "CMake output:"
    cat /tmp/gate0_cmake.log
fi

if [ $CMAKE_EXIT -eq 0 ]; then
    echo "Running: cmake --build build -j"
    cmake --build build -j > /tmp/gate0_build.log 2>&1
    BUILD_EXIT=$?
    echo "Build exit code: $BUILD_EXIT"
    if [ $BUILD_EXIT -ne 0 ]; then
        echo "Build output (last 50 lines):"
        tail -n 50 /tmp/gate0_build.log
    fi
else
    BUILD_EXIT=1
fi

if [ $CMAKE_EXIT -eq 0 ] && [ $BUILD_EXIT -eq 0 ]; then
    GATE0_RESULT=0
else
    GATE0_RESULT=1
fi
record_gate 0 "Build" $GATE0_RESULT
echo ""

#############################################
# Gate 1 - ZERO removed API usage
#############################################
echo "=== Gate 1: ZERO removed API usage in default stack ==="
echo "Searching for 'crm_diff_py.dynamics_' in python/ src/ docs/"

GATE1_RESULT=0
rg --no-heading --with-filename --line-number "crm_diff_py\.dynamics_" python/ src/ docs/ > /tmp/gate1_output.txt 2>&1
RG_EXIT=$?

if [ $RG_EXIT -eq 0 ]; then
    # Found matches - this is a FAIL
    echo "Found forbidden API usage:"
    cat /tmp/gate1_output.txt
    GATE1_RESULT=1
elif [ $RG_EXIT -eq 1 ]; then
    # No matches found - this is a PASS
    echo "No forbidden API usage found (expected)"
    GATE1_RESULT=0
else
    # Error occurred
    echo "Search error (exit code $RG_EXIT):"
    cat /tmp/gate1_output.txt
    GATE1_RESULT=1
fi

record_gate 1 "ZERO removed API usage" $GATE1_RESULT
echo ""

#############################################
# Gate 2 - ZERO reduced-state leaks
#############################################
echo "=== Gate 2: ZERO reduced-state leaks in default stack ==="
echo "Searching for 'u_0', 'v_0', '6D', '9D hybrid state' outside python/control/reduced6d/"

GATE2_RESULT=0

# Search for u_0
rg --no-heading --with-filename --line-number '\bu_0\b' python/ src/ docs/ | grep -v "python/control/reduced6d/" > /tmp/gate2_u0.txt 2>&1
U0_MATCHES=$(cat /tmp/gate2_u0.txt | wc -l)

# Search for v_0
rg --no-heading --with-filename --line-number '\bv_0\b' python/ src/ docs/ | grep -v "python/control/reduced6d/" > /tmp/gate2_v0.txt 2>&1
V0_MATCHES=$(cat /tmp/gate2_v0.txt | wc -l)

# Search for 6D (but not in paths or common contexts)
rg --no-heading --with-filename --line-number '\b6D\b' python/ src/ docs/ | grep -v "python/control/reduced6d/" | grep -v "ARCHIVE" | grep -v "MIGRATION" > /tmp/gate2_6d.txt 2>&1
SIXD_MATCHES=$(cat /tmp/gate2_6d.txt | wc -l)

# Search for 9D hybrid state
rg --no-heading --with-filename --line-number '9D hybrid state' python/ src/ docs/ | grep -v "python/control/reduced6d/" | grep -v "ARCHIVE" | grep -v "MIGRATION" > /tmp/gate2_9d.txt 2>&1
NINED_MATCHES=$(cat /tmp/gate2_9d.txt | wc -l)

TOTAL_LEAKS=$((U0_MATCHES + V0_MATCHES + SIXD_MATCHES + NINED_MATCHES))

if [ $TOTAL_LEAKS -gt 0 ]; then
    echo "Found reduced-state leaks outside reduced6d/:"
    if [ $U0_MATCHES -gt 0 ]; then
        echo "u_0 matches: $U0_MATCHES"
        cat /tmp/gate2_u0.txt
    fi
    if [ $V0_MATCHES -gt 0 ]; then
        echo "v_0 matches: $V0_MATCHES"
        cat /tmp/gate2_v0.txt
    fi
    if [ $SIXD_MATCHES -gt 0 ]; then
        echo "6D matches: $SIXD_MATCHES"
        cat /tmp/gate2_6d.txt
    fi
    if [ $NINED_MATCHES -gt 0 ]; then
        echo "9D hybrid state matches: $NINED_MATCHES"
        cat /tmp/gate2_9d.txt
    fi
    GATE2_RESULT=1
else
    echo "No reduced-state leaks found (expected)"
    GATE2_RESULT=0
fi

record_gate 2 "ZERO reduced-state leaks" $GATE2_RESULT
echo ""

#############################################
# Gate 3 - FULLSTATE contract referenced
#############################################
echo "=== Gate 3: FULLSTATE contract is referenced ==="
echo "Searching for FULLSTATE indicators in python/control/ src/ python/crm_bindings.cpp"

GATE3_RESULT=1  # Default to FAIL, set to PASS if we find evidence

# Search for 18*N+15 or 18*NUM_ACT_SET+15
rg --no-heading --with-filename --line-number '18\s*\*\s*(N|NUM_ACT_SET)\s*\+\s*15' python/control/ src/ python/crm_bindings.cpp > /tmp/gate3_pattern1.txt 2>&1
PATTERN1_EXIT=$?

# Search for STATE_DIM with 18
rg --no-heading --with-filename --line-number 'STATE_DIM.*18|18.*STATE_DIM' python/control/ src/ python/crm_bindings.cpp > /tmp/gate3_pattern2.txt 2>&1
PATTERN2_EXIT=$?

# Search for FULLSTATE
rg --no-heading --with-filename --line-number 'FULLSTATE|FullState|full_state' python/control/ src/ python/crm_bindings.cpp > /tmp/gate3_pattern3.txt 2>&1
PATTERN3_EXIT=$?

if [ $PATTERN1_EXIT -eq 0 ] || [ $PATTERN2_EXIT -eq 0 ] || [ $PATTERN3_EXIT -eq 0 ]; then
    echo "Found FULLSTATE contract references:"
    if [ $PATTERN1_EXIT -eq 0 ]; then
        echo "18*N+15 / 18*NUM_ACT_SET+15 pattern:"
        cat /tmp/gate3_pattern1.txt
    fi
    if [ $PATTERN2_EXIT -eq 0 ]; then
        echo "STATE_DIM with 18 pattern:"
        cat /tmp/gate3_pattern2.txt
    fi
    if [ $PATTERN3_EXIT -eq 0 ]; then
        echo "FULLSTATE keyword:"
        cat /tmp/gate3_pattern3.txt
    fi
    GATE3_RESULT=0
else
    echo "No FULLSTATE contract references found (unexpected - FAIL)"
    GATE3_RESULT=1
fi

record_gate 3 "FULLSTATE contract referenced" $GATE3_RESULT
echo ""

#############################################
# Gate 4 - Controllers wired to FULLSTATE
#############################################
echo "=== Gate 4: Controllers wired to FULLSTATE step + linearization ==="

GATE4_RESULT=0
CONTROLLERS=("python/control/ilqr.py" "python/control/mpc.py" "python/control/lqr.py" "python/control/hybrid_controller.py")

for controller in "${CONTROLLERS[@]}"; do
    echo "Checking $controller"

    if [ ! -f "$controller" ]; then
        echo "  WARNING: $controller not found"
        GATE4_RESULT=1
        continue
    fi

    # Check for step hooks
    STEP_FOUND=0
    rg --quiet 'true_legacy_step|TrueLegacy|DynamicsBVP|DYNSolverIVP' "$controller"
    if [ $? -eq 0 ]; then
        STEP_FOUND=1
        echo "  ✓ Found step hook"
    fi

    # Check for linearization hooks
    LIN_FOUND=0
    rg --quiet 'linearize|A,\s*B|jacob|vjp|jvp' "$controller"
    if [ $? -eq 0 ]; then
        LIN_FOUND=1
        echo "  ✓ Found linearization hook"
    fi

    if [ $STEP_FOUND -eq 0 ]; then
        echo "  ✗ Missing step hook"
        GATE4_RESULT=1
    fi

    if [ $LIN_FOUND -eq 0 ]; then
        echo "  ✗ Missing linearization hook"
        GATE4_RESULT=1
    fi
done

record_gate 4 "Controllers wired to FULLSTATE" $GATE4_RESULT
echo ""

#############################################
# Gate 5 - Reduced6D is opt-in only
#############################################
echo "=== Gate 5: Reduced6D is opt-in only ==="
echo "Searching for legacy alias imports outside reduced6d/"

GATE5_RESULT=0

# Search for legacy imports in python/control/
rg --no-heading --with-filename --line-number 'from control import.*legacy|import control.*legacy' python/control/ > /tmp/gate5_all.txt 2>&1
ALL_EXIT=$?

if [ $ALL_EXIT -eq 0 ]; then
    # Found some imports, check if they're all in reduced6d/
    grep -v "python/control/reduced6d/" /tmp/gate5_all.txt > /tmp/gate5_outside.txt 2>&1
    OUTSIDE_COUNT=$(cat /tmp/gate5_outside.txt | wc -l)

    if [ $OUTSIDE_COUNT -gt 0 ]; then
        echo "Found legacy imports outside reduced6d/:"
        cat /tmp/gate5_outside.txt
        GATE5_RESULT=1
    else
        echo "All legacy imports are in reduced6d/ (expected)"
        GATE5_RESULT=0
    fi
elif [ $ALL_EXIT -eq 1 ]; then
    # No legacy imports found at all - also acceptable
    echo "No legacy imports found (acceptable)"
    GATE5_RESULT=0
else
    echo "Search error (exit code $ALL_EXIT)"
    GATE5_RESULT=1
fi

record_gate 5 "Reduced6D is opt-in only" $GATE5_RESULT
echo ""

#############################################
# Final Summary
#############################################
echo "=============================================="
echo "FINAL SUMMARY"
echo "=============================================="
echo "Gate 0 (Build):                   ${GATE_RESULTS[0]}"
echo "Gate 1 (ZERO removed API):        ${GATE_RESULTS[1]}"
echo "Gate 2 (ZERO reduced leaks):      ${GATE_RESULTS[2]}"
echo "Gate 3 (FULLSTATE contract):      ${GATE_RESULTS[3]}"
echo "Gate 4 (Controller wiring):       ${GATE_RESULTS[4]}"
echo "Gate 5 (Reduced6D opt-in):        ${GATE_RESULTS[5]}"
echo "=============================================="

if [ $ALL_PASSED -eq 0 ]; then
    echo -e "${GREEN}OVERALL VERDICT: PASS${NC}"
    echo "All sanity gates passed successfully."
    exit 0
else
    echo -e "${RED}OVERALL VERDICT: FAIL${NC}"
    echo "One or more sanity gates failed."
    exit 1
fi
