"""
CP4.0/CP4.1: Data loading and dataset management
"""

from .npz_dataset import load_npz_dataset, NPZDataset
from .npz_manifest import (
    DatasetManifest,
    load_manifest,
    create_splits,
    print_manifest_summary
)

__all__ = [
    'load_npz_dataset',
    'NPZDataset',
    'DatasetManifest',
    'load_manifest',
    'create_splits',
    'print_manifest_summary'
]
