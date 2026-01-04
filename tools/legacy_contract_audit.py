#!/usr/bin/env python3
"""
Legacy Contract Audit Helper

Read-only tool to extract authoritative evidence from the legacy codebase.
Locates and prints definitions of:
  - DynamicsStepResult struct
  - dynamics_forward / dynamics_backward functions
  - EquilibriumResult struct
  - BVP solver declarations

Usage:
    python3 tools/legacy_contract_audit.py [--legacy PATH]

If --legacy is not provided, searches in the current repository.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Patterns to search for
PATTERNS = [
    (r"struct\s+DynamicsStepResult\s*\{", "DynamicsStepResult struct"),
    (r"struct\s+EquilibriumResult\s*\{", "EquilibriumResult struct"),
    (r"int\s+dynamics_forward\s*\(", "dynamics_forward function"),
    (r"int\s+dynamics_backward\s*\(", "dynamics_backward function"),
    (r"int\s+dynamics_backward_batched\s*\(", "dynamics_backward_batched function"),
    (r"int\s+equilibrium_forward\s*\(", "equilibrium_forward function"),
    (r"int\s+equilibrium_backward\s*\(", "equilibrium_backward function"),
    (r"void\s+CRMShootingMethodBVP\s*\(", "CRMShootingMethodBVP function"),
    (r"double\s+x_next\s*\[\s*6\s*\]", "x_next[6] state vector"),
    (r"double\s+p_tip\s*\[\s*3\s*\]", "p_tip[3] observable"),
]


def find_matches(
    root: Path, pattern: str, description: str
) -> List[Tuple[Path, int, str]]:
    """
    Search for pattern in all .hpp/.cpp files under root.
    Returns list of (filepath, line_number, line_content).
    """
    matches = []
    regex = re.compile(pattern)

    for ext in ("*.hpp", "*.cpp", "*.h"):
        for filepath in root.rglob(ext):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append((filepath, i, line.rstrip()))
            except (IOError, OSError):
                continue

    return matches


def print_context(filepath: Path, line_num: int, context: int = 3) -> None:
    """Print lines around the match for context."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)

        print("    Context:")
        for i in range(start, end):
            marker = ">>>" if i == line_num - 1 else "   "
            print(f"    {marker} {i+1:4d}: {lines[i].rstrip()}")
    except (IOError, OSError):
        print("    [Could not read context]")


def audit_legacy(root: Path, show_context: bool = True) -> int:
    """
    Run the audit on the given root directory.
    Returns number of patterns found.
    """
    print(f"\n{'='*60}")
    print(f"LEGACY CONTRACT AUDIT")
    print(f"Root: {root.resolve()}")
    print(f"{'='*60}\n")

    found_count = 0

    for pattern, description in PATTERNS:
        matches = find_matches(root, pattern, description)

        if matches:
            found_count += 1
            print(f"[FOUND] {description}")
            for filepath, line_num, line_content in matches:
                rel_path = filepath.relative_to(root) if filepath.is_relative_to(root) else filepath
                print(f"  File: {rel_path}:{line_num}")
                print(f"  Match: {line_content.strip()}")
                if show_context:
                    print_context(filepath, line_num)
                print()
        else:
            print(f"[NOT FOUND] {description}")
            print()

    return found_count


def print_worktree_instructions() -> None:
    """Print instructions for setting up a legacy worktree."""
    print("""
INSTRUCTIONS: Setting up a legacy worktree for read-only comparison
====================================================================

If you need to compare against the legacy main branch:

1. Create a git worktree (read-only access):

   git worktree add ../legacy-main main --detach

2. Run this audit on the worktree:

   python3 tools/legacy_contract_audit.py --legacy ../legacy-main

3. When done, remove the worktree:

   git worktree remove ../legacy-main

IMPORTANT: Do NOT modify the main branch. It is read-only ground truth.
""")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit legacy hybrid state contract definitions"
    )
    parser.add_argument(
        "--legacy",
        type=str,
        default=None,
        help="Path to legacy codebase (default: current directory)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't show surrounding code context"
    )

    args = parser.parse_args()

    # Determine root path
    if args.legacy:
        root = Path(args.legacy)
        if not root.exists():
            print(f"ERROR: Legacy path does not exist: {root}")
            print_worktree_instructions()
            return 1
    else:
        # Use current working directory or find repo root
        root = Path.cwd()
        src_dir = root / "src"
        if not src_dir.exists():
            print(f"ERROR: No 'src' directory found in {root}")
            print("Are you running from the repository root?")
            print_worktree_instructions()
            return 1
        root = src_dir.parent

    # Run audit
    found = audit_legacy(root, show_context=not args.no_context)

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY: Found {found}/{len(PATTERNS)} patterns")
    print(f"{'='*60}")

    if found < len(PATTERNS):
        print("\nSome patterns were not found. This may indicate:")
        print("  - Different file structure in legacy")
        print("  - Renamed symbols")
        print("  - Missing files")
        print("\nRun with --legacy pointing to the correct legacy codebase.")

    return 0 if found > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
