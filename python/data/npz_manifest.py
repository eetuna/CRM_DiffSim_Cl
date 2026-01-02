"""
CP4.1: NPZ Dataset Manifest and Splits

Provides centralized registry of all NPZ trajectory datasets, parameter
consistency checking, and train/val/test split generation.

Key features:
- Lists all available NPZ files
- Validates parameter consistency across datasets
- Creates train/val/test splits by trajectory type
- Filters datasets without references (no_ref)
"""

import os
import glob
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .npz_dataset import load_npz_dataset, NPZDataset


@dataclass
class DatasetManifest:
    """
    Registry of all available NPZ datasets with metadata.

    Attributes:
        all_datasets: List of all NPZDataset objects
        datasets_with_ref: List of datasets with reference trajectories
        datasets_no_ref: List of datasets without reference trajectories
        by_trajectory_type: Dict mapping trajectory type to datasets
        param_consistency: Dict with parameter consistency info
    """
    all_datasets: List[NPZDataset]
    datasets_with_ref: List[NPZDataset]
    datasets_no_ref: List[NPZDataset]
    by_trajectory_type: Dict[str, List[NPZDataset]]
    param_consistency: Dict[str, any]

    def __repr__(self):
        return (
            f"DatasetManifest(\n"
            f"  total={len(self.all_datasets)},\n"
            f"  with_ref={len(self.datasets_with_ref)},\n"
            f"  no_ref={len(self.datasets_no_ref)},\n"
            f"  types={list(self.by_trajectory_type.keys())}\n"
            f")"
        )


def discover_npz_files(data_dir: str = './data') -> List[str]:
    """
    Discover all NPZ files in data directory.

    Args:
        data_dir: Directory to search for NPZ files

    Returns:
        List of NPZ file paths (sorted)
    """
    pattern = os.path.join(data_dir, '*.npz')
    files = glob.glob(pattern)
    return sorted(files)


def classify_trajectory_type(filename: str) -> str:
    """
    Classify trajectory type from filename.

    Args:
        filename: NPZ filename

    Returns:
        str: Trajectory type ('circle', 'lemniscate', 'workspace', 'unknown')
    """
    lower = filename.lower()
    if 'circle' in lower:
        return 'circle'
    elif 'lem' in lower:  # lemniscate
        return 'lemniscate'
    elif 'workspace' in lower:
        return 'workspace'
    else:
        return 'unknown'


def check_param_consistency(datasets: List[NPZDataset]) -> Dict[str, any]:
    """
    Check parameter consistency across datasets.

    Args:
        datasets: List of NPZDataset objects

    Returns:
        Dict with consistency info:
            - consistent: bool (True if all params match)
            - dt_values: set of unique dt values
            - L_values: set of unique L_inserted values
            - h_values: set of unique integration_step_size values
            - warnings: list of warning messages
    """
    if not datasets:
        return {
            'consistent': True,
            'dt_values': set(),
            'L_values': set(),
            'h_values': set(),
            'warnings': []
        }

    dt_values = set()
    L_values = set()
    h_values = set()
    warnings = []

    for ds in datasets:
        dt_values.add(round(ds.dt, 6))  # Round to avoid float precision issues
        L_values.add(round(ds.L_inserted, 3))
        h_values.add(round(ds.integration_step_size, 3))

    # Check consistency
    consistent = True
    if len(dt_values) > 1:
        consistent = False
        warnings.append(f"Inconsistent dt values: {sorted(dt_values)}")
    if len(L_values) > 1:
        consistent = False
        warnings.append(f"Inconsistent L_inserted values: {sorted(L_values)}")
    if len(h_values) > 1:
        # h inconsistency is a warning but not a blocker
        warnings.append(f"NOTE: Multiple integration_step_size values: {sorted(h_values)}")

    return {
        'consistent': consistent,
        'dt_values': sorted(dt_values),
        'L_values': sorted(L_values),
        'h_values': sorted(h_values),
        'warnings': warnings
    }


def load_manifest(data_dir: str = './data', verbose: bool = True) -> DatasetManifest:
    """
    Load all NPZ datasets and create manifest.

    Args:
        data_dir: Directory containing NPZ files
        verbose: If True, print loading progress

    Returns:
        DatasetManifest object
    """
    if verbose:
        print(f"Loading NPZ manifest from {data_dir}...")

    # Discover files
    npz_files = discover_npz_files(data_dir)
    if verbose:
        print(f"  Found {len(npz_files)} NPZ file(s)")

    # Load datasets
    all_datasets = []
    datasets_with_ref = []
    datasets_no_ref = []
    by_trajectory_type = {}

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)

        try:
            ds = load_npz_dataset(npz_path, validate=False)  # Relaxed validation
            all_datasets.append(ds)

            # Classify by reference availability
            if ds.has_reference:
                datasets_with_ref.append(ds)
            else:
                datasets_no_ref.append(ds)

            # Classify by trajectory type
            traj_type = classify_trajectory_type(filename)
            if traj_type not in by_trajectory_type:
                by_trajectory_type[traj_type] = []
            by_trajectory_type[traj_type].append(ds)

            if verbose:
                ref_status = "with_ref" if ds.has_reference else "NO_REF"
                print(f"    ✓ {filename}: {traj_type}, {ref_status}, {ds.param_summary()}")

        except Exception as e:
            if verbose:
                print(f"    ✗ {filename}: FAILED - {e}")

    # Check parameter consistency (only for datasets with references)
    param_consistency = check_param_consistency(datasets_with_ref)

    if verbose:
        print(f"\nManifest Summary:")
        print(f"  Total datasets: {len(all_datasets)}")
        print(f"  With reference: {len(datasets_with_ref)}")
        print(f"  No reference: {len(datasets_no_ref)}")
        print(f"  Trajectory types: {list(by_trajectory_type.keys())}")

        if param_consistency['warnings']:
            print(f"\nParameter Consistency:")
            for warning in param_consistency['warnings']:
                print(f"  ⚠ {warning}")

    manifest = DatasetManifest(
        all_datasets=all_datasets,
        datasets_with_ref=datasets_with_ref,
        datasets_no_ref=datasets_no_ref,
        by_trajectory_type=by_trajectory_type,
        param_consistency=param_consistency
    )

    return manifest


def create_splits(
    manifest: DatasetManifest,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_type: bool = True,
    verbose: bool = True
) -> Tuple[List[NPZDataset], List[NPZDataset], List[NPZDataset]]:
    """
    Create train/val/test splits from manifest.

    Args:
        manifest: DatasetManifest object
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for test (default 0.15)
        seed: Random seed for reproducibility
        stratify_by_type: If True, split within each trajectory type
        verbose: If True, print split info

    Returns:
        Tuple of (train_datasets, val_datasets, test_datasets)
    """
    np.random.seed(seed)

    # Only use datasets with references for learning
    datasets = manifest.datasets_with_ref

    if not datasets:
        raise ValueError("No datasets with references available for splits")

    if stratify_by_type:
        # Split within each trajectory type
        train_all = []
        val_all = []
        test_all = []

        for traj_type, type_datasets in manifest.by_trajectory_type.items():
            # Filter for datasets with references
            type_datasets_ref = [ds for ds in type_datasets if ds.has_reference]

            if not type_datasets_ref:
                continue

            # Shuffle
            indices = np.random.permutation(len(type_datasets_ref))
            shuffled = [type_datasets_ref[i] for i in indices]

            # Compute split sizes
            n = len(shuffled)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val

            # Ensure at least 1 sample per split if possible
            if n >= 3:
                # Split normally
                test_ds = shuffled[:n_test]
                val_ds = shuffled[n_test:n_test + n_val]
                train_ds = shuffled[n_test + n_val:]
            elif n == 2:
                # 1 train, 1 test
                train_ds = [shuffled[0]]
                val_ds = []
                test_ds = [shuffled[1]]
            else:  # n == 1
                # All in train
                train_ds = shuffled
                val_ds = []
                test_ds = []

            train_all.extend(train_ds)
            val_all.extend(val_ds)
            test_all.extend(test_ds)

            if verbose:
                print(f"  {traj_type}: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    else:
        # Random split across all datasets
        indices = np.random.permutation(len(datasets))
        shuffled = [datasets[i] for i in indices]

        n = len(shuffled)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_all = shuffled[:n_test]
        val_all = shuffled[n_test:n_test + n_val]
        train_all = shuffled[n_test + n_val:]

    if verbose:
        print(f"\nFinal Splits:")
        print(f"  Train: {len(train_all)} datasets")
        print(f"  Val:   {len(val_all)} datasets")
        print(f"  Test:  {len(test_all)} datasets")

    return train_all, val_all, test_all


def print_manifest_summary(manifest: DatasetManifest):
    """
    Print detailed manifest summary.

    Args:
        manifest: DatasetManifest object
    """
    print("=" * 80)
    print("NPZ Dataset Manifest Summary")
    print("=" * 80)

    print(f"\nTotal Datasets: {len(manifest.all_datasets)}")
    print(f"  With Reference: {len(manifest.datasets_with_ref)}")
    print(f"  No Reference: {len(manifest.datasets_no_ref)}")

    print(f"\nBy Trajectory Type:")
    for traj_type, datasets in manifest.by_trajectory_type.items():
        with_ref = sum(1 for ds in datasets if ds.has_reference)
        no_ref = len(datasets) - with_ref
        print(f"  {traj_type}: {len(datasets)} total ({with_ref} with_ref, {no_ref} no_ref)")

    print(f"\nParameter Consistency:")
    pc = manifest.param_consistency
    if pc['consistent']:
        print(f"  ✓ Parameters consistent across datasets with references")
    else:
        print(f"  ✗ Parameter inconsistencies detected:")
        for warning in pc['warnings']:
            print(f"    - {warning}")

    if manifest.datasets_with_ref:
        print(f"\nDatasets with References:")
        for ds in manifest.datasets_with_ref:
            n_valid = np.sum(~ds.hold_mask)
            print(f"  • {ds.filename}: {n_valid}/{ds.n_steps} valid samples, {ds.param_summary()}")

    if manifest.datasets_no_ref:
        print(f"\nDatasets without References (excluded from learning):")
        for ds in manifest.datasets_no_ref:
            print(f"  • {ds.filename}: {ds.param_summary()}")

    print("=" * 80)
