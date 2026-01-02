"""
CP4.0/CP4.1: NPZ Dataset Loader

Provides robust loading and validation of NPZ trajectory datasets for learning.

Key features:
- Validates required fields (currents, tip trajectories, dt, L_inserted)
- Returns structured dataset with arrays + metadata
- Handles optional fields gracefully
- Provides helpful error messages
- CP4.1: Standardized contract with tip_ref, hold_mask, param_summary()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class NPZDataset:
    """
    Structured container for NPZ trajectory dataset.

    Required fields:
        t: Time array [N]
        currents: Control inputs [N, 3]
        dt: Timestep (s)
        L_inserted: Insertion length (mm)
        integration_step_size: Integration step size

    Optional trajectory fields (at least one required):
        tip_fk: FK tip positions [N, 3]
        tip_dyn: Dynamics tip positions [N, 3]
        tip_desired: Desired tip positions [N, 3]
        tip_projected: Projected tip positions [N, 3]

    Optional metadata fields:
        dyn_converged: Dynamics convergence flags [N]
        fk_dyn_err: FK-Dyn error [N]
        deltau0: Delta u0 values [N, 3] (if available)
        hold: Hold index (int)
        segments: Segment count (int)

    Additional fields:
        metadata: Dict with any additional NPZ keys
        filename: Source filename
    """
    # Time and controls (required)
    t: np.ndarray
    currents: np.ndarray

    # Parameters (required)
    dt: float
    L_inserted: float
    integration_step_size: float

    # Trajectories (at least one required)
    tip_fk: Optional[np.ndarray] = None
    tip_dyn: Optional[np.ndarray] = None
    tip_desired: Optional[np.ndarray] = None
    tip_projected: Optional[np.ndarray] = None

    # Optional metadata
    dyn_converged: Optional[np.ndarray] = None
    fk_dyn_err: Optional[np.ndarray] = None
    deltau0: Optional[np.ndarray] = None
    hold: Optional[int] = None
    segments: Optional[int] = None

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    filename: Optional[str] = None

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the dataset."""
        return len(self.t)

    @property
    def duration(self) -> float:
        """Total duration of the trajectory (s)."""
        return self.t[-1] - self.t[0]

    @property
    def has_dynamics(self) -> bool:
        """Whether the dataset contains dynamics trajectories."""
        return self.tip_dyn is not None

    @property
    def has_fk(self) -> bool:
        """Whether the dataset contains FK trajectories."""
        return self.tip_fk is not None

    @property
    def has_reference(self) -> bool:
        """Whether the dataset contains a reference trajectory."""
        return self.tip_desired is not None or self.tip_projected is not None

    @property
    def reference_trajectory(self) -> Optional[np.ndarray]:
        """
        Get the reference trajectory (tip_projected preferred, then tip_desired).

        Returns:
            np.ndarray or None: Reference trajectory [N, 3] if available
        """
        if self.tip_projected is not None:
            return self.tip_projected
        elif self.tip_desired is not None:
            return self.tip_desired
        else:
            return None

    @property
    def tip_ref(self) -> Optional[np.ndarray]:
        """
        CP4.1: Standardized reference trajectory accessor.

        Alias for reference_trajectory. Returns tip_projected if available,
        otherwise tip_desired, otherwise None.

        Returns:
            np.ndarray or None: Reference trajectory [N, 3] if available
        """
        return self.reference_trajectory

    @property
    def hold_mask(self) -> np.ndarray:
        """
        CP4.1: Boolean mask indicating where reference is invalid/NaN.

        Returns:
            np.ndarray: Boolean array [N] where True indicates hold period
                       (reference unavailable or NaN). If no reference exists,
                       returns all True.
        """
        ref = self.tip_ref
        if ref is None:
            # No reference available, entire trajectory is "hold"
            if self.t is not None:
                return np.ones(self.n_steps, dtype=bool)
            elif self.currents is not None:
                return np.ones(len(self.currents), dtype=bool)
            else:
                # Fallback: return empty array
                return np.array([], dtype=bool)
        else:
            # Check for NaN in any dimension
            return np.isnan(ref).any(axis=1)

    def param_summary(self) -> str:
        """
        CP4.1: Print standardized parameter summary.

        Returns:
            str: Exactly "PARAMS: dt=..., L=..., h=..." or "PARAMS: INCOMPLETE" if missing
        """
        try:
            if self.dt is None or self.L_inserted is None or self.integration_step_size is None:
                return "PARAMS: INCOMPLETE (missing dt, L, or h)"
            return f"PARAMS: dt={self.dt:.4f}s, L={self.L_inserted:.1f}mm, h={self.integration_step_size:.2f}"
        except (TypeError, ValueError):
            return "PARAMS: ERROR (invalid parameter values)"

    def get_state_control_pairs(self, use_fk=False):
        """
        Get (tip_position, current) pairs for supervised learning.

        Args:
            use_fk: If True, use tip_fk; otherwise use tip_dyn

        Returns:
            tuple: (positions [N, 3], currents [N, 3])

        Raises:
            ValueError: If requested trajectory type not available
        """
        if use_fk:
            if self.tip_fk is None:
                raise ValueError("FK trajectories not available in dataset")
            return self.tip_fk, self.currents
        else:
            if self.tip_dyn is None:
                raise ValueError("Dynamics trajectories not available in dataset")
            return self.tip_dyn, self.currents

    def get_tracking_pairs(self, use_fk=False):
        """
        Get (tip_position, reference_position) pairs for tracking tasks.

        Args:
            use_fk: If True, use tip_fk; otherwise use tip_dyn

        Returns:
            tuple: (positions [N, 3], references [N, 3])

        Raises:
            ValueError: If requested trajectory or reference not available
        """
        if not self.has_reference:
            raise ValueError("No reference trajectory available in dataset")

        reference = self.reference_trajectory

        if use_fk:
            if self.tip_fk is None:
                raise ValueError("FK trajectories not available in dataset")
            return self.tip_fk, reference
        else:
            if self.tip_dyn is None:
                raise ValueError("Dynamics trajectories not available in dataset")
            return self.tip_dyn, reference

    def compute_tracking_error(self, use_fk=False):
        """
        Compute tracking error vs reference trajectory.

        Args:
            use_fk: If True, use tip_fk; otherwise use tip_dyn

        Returns:
            dict: {
                'rms': float,
                'max': float,
                'mean': float,
                'errors': np.ndarray [N]
            }

        Raises:
            ValueError: If reference or trajectory not available
        """
        positions, reference = self.get_tracking_pairs(use_fk=use_fk)
        errors = np.linalg.norm(positions - reference, axis=1)

        return {
            'rms': float(np.sqrt(np.mean(errors**2))),
            'max': float(np.max(errors)),
            'mean': float(np.mean(errors)),
            'errors': errors,
        }

    def __repr__(self):
        parts = [
            f"NPZDataset(n_steps={self.n_steps}, duration={self.duration:.3f}s",
            f"dt={self.dt:.4f}s, L={self.L_inserted:.1f}mm, h={self.integration_step_size:.2f}",
        ]

        trajs = []
        if self.has_fk:
            trajs.append("FK")
        if self.has_dynamics:
            trajs.append("Dyn")
        if self.has_reference:
            # Count valid (non-hold) samples
            n_valid = np.sum(~self.hold_mask)
            trajs.append(f"Ref({n_valid} valid)")
        else:
            trajs.append("no_ref")
        if trajs:
            parts.append(f"trajectories=[{', '.join(trajs)}]")

        if self.filename:
            parts.append(f"file={self.filename}")

        return ", ".join(parts) + ")"


def load_npz_dataset(npz_path: str, validate: bool = True) -> NPZDataset:
    """
    Load NPZ trajectory dataset with validation.

    Args:
        npz_path: Path to NPZ file
        validate: If True, validate required fields and raise errors

    Returns:
        NPZDataset: Loaded dataset

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails and validate=True
    """
    import os

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Load NPZ
    try:
        data = np.load(npz_path)
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file {npz_path}: {e}")

    errors = []

    # Extract required fields
    # Time array
    if 't' not in data:
        errors.append("Missing required field: 't' (time array)")
        t = None
    else:
        t = data['t']

    # Currents
    if 'currents' not in data:
        errors.append("Missing required field: 'currents'")
        currents = None
    else:
        currents = data['currents']

    # dt
    if 'dt' not in data:
        errors.append("Missing required field: 'dt'")
        dt = None
    else:
        dt = float(data['dt'])

    # L_inserted (may be stored as 'insertion_length')
    if 'insertion_length' in data:
        L_inserted = float(data['insertion_length'])
    elif 'L_inserted' in data:
        L_inserted = float(data['L_inserted'])
    else:
        errors.append("Missing required field: 'insertion_length' or 'L_inserted'")
        L_inserted = None

    # integration_step_size (optional, default to 0.5)
    if 'integration_step_size' in data:
        integration_step_size = float(data['integration_step_size'])
    else:
        integration_step_size = 0.5  # Default for old files
        # Don't error, just use default

    # Check for at least one trajectory
    tip_fk = data.get('tip_fk', None)
    tip_dyn = data.get('tip_dyn', None)
    tip_desired = data.get('tip_desired', None)
    tip_projected = data.get('tip_projected', None)

    has_any_traj = any(x is not None for x in [tip_fk, tip_dyn, tip_desired, tip_projected])
    if not has_any_traj:
        errors.append(
            "No trajectory found. Need at least one of: "
            "tip_fk, tip_dyn, tip_desired, tip_projected"
        )

    # If validation enabled and errors found, raise
    if validate and errors:
        error_msg = f"NPZ validation failed for {npz_path}:\n"
        error_msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    # Extract optional fields
    dyn_converged = data.get('dyn_converged', None)
    fk_dyn_err = data.get('fk_dyn_err', None)
    deltau0 = data.get('DU0', None)  # May be stored as 'DU0' in some files

    # Handle scalar metadata (may be 0-d arrays or scalars)
    hold = None
    if 'hold' in data:
        hold_val = data['hold']
        if isinstance(hold_val, np.ndarray):
            if hold_val.ndim == 0:  # 0-d array
                hold = int(hold_val.item())
            elif hold_val.size == 1:
                hold = int(hold_val.flat[0])
        else:
            hold = int(hold_val)

    segments = None
    if 'segments' in data:
        seg_val = data['segments']
        if isinstance(seg_val, np.ndarray):
            if seg_val.ndim == 0:  # 0-d array
                segments = int(seg_val.item())
            elif seg_val.size == 1:
                segments = int(seg_val.flat[0])
        else:
            segments = int(seg_val)

    # Collect additional metadata
    metadata = {}
    known_fields = {
        't', 'currents', 'dt', 'insertion_length', 'L_inserted',
        'integration_step_size', 'tip_fk', 'tip_dyn', 'tip_desired',
        'tip_projected', 'dyn_converged', 'fk_dyn_err', 'DU0',
        'hold', 'segments'
    }
    for key in data.keys():
        if key not in known_fields:
            metadata[key] = data[key]

    # Create dataset
    dataset = NPZDataset(
        t=t,
        currents=currents,
        dt=dt,
        L_inserted=L_inserted,
        integration_step_size=integration_step_size,
        tip_fk=tip_fk,
        tip_dyn=tip_dyn,
        tip_desired=tip_desired,
        tip_projected=tip_projected,
        dyn_converged=dyn_converged,
        fk_dyn_err=fk_dyn_err,
        deltau0=deltau0,
        hold=hold,
        segments=segments,
        metadata=metadata,
        filename=os.path.basename(npz_path),
    )

    return dataset


def load_multiple_datasets(npz_paths: List[str], validate: bool = True) -> List[NPZDataset]:
    """
    Load multiple NPZ datasets.

    Args:
        npz_paths: List of paths to NPZ files
        validate: If True, validate each file

    Returns:
        List[NPZDataset]: List of loaded datasets

    Raises:
        ValueError: If any file fails validation (when validate=True)
    """
    datasets = []
    for path in npz_paths:
        dataset = load_npz_dataset(path, validate=validate)
        datasets.append(dataset)
    return datasets


def concatenate_datasets(datasets: List[NPZDataset]) -> NPZDataset:
    """
    Concatenate multiple datasets into one.

    All datasets must have the same:
    - dt
    - L_inserted
    - integration_step_size
    - trajectory fields (all must have same fields available)

    Args:
        datasets: List of NPZDataset to concatenate

    Returns:
        NPZDataset: Concatenated dataset

    Raises:
        ValueError: If datasets are incompatible
    """
    if not datasets:
        raise ValueError("Cannot concatenate empty list of datasets")

    if len(datasets) == 1:
        return datasets[0]

    # Check compatibility
    ref = datasets[0]
    for i, ds in enumerate(datasets[1:], 1):
        if abs(ds.dt - ref.dt) > 1e-6:
            raise ValueError(f"Dataset {i} has different dt: {ds.dt} vs {ref.dt}")
        if abs(ds.L_inserted - ref.L_inserted) > 1e-3:
            raise ValueError(f"Dataset {i} has different L_inserted: {ds.L_inserted} vs {ref.L_inserted}")
        if abs(ds.integration_step_size - ref.integration_step_size) > 1e-6:
            raise ValueError(f"Dataset {i} has different integration_step_size")

        # Check trajectory availability
        if (ds.tip_fk is None) != (ref.tip_fk is None):
            raise ValueError(f"Dataset {i} has different tip_fk availability")
        if (ds.tip_dyn is None) != (ref.tip_dyn is None):
            raise ValueError(f"Dataset {i} has different tip_dyn availability")

    # Concatenate arrays
    t_concat = np.concatenate([ds.t for ds in datasets])
    currents_concat = np.concatenate([ds.currents for ds in datasets])

    tip_fk_concat = None
    if ref.tip_fk is not None:
        tip_fk_concat = np.concatenate([ds.tip_fk for ds in datasets])

    tip_dyn_concat = None
    if ref.tip_dyn is not None:
        tip_dyn_concat = np.concatenate([ds.tip_dyn for ds in datasets])

    tip_desired_concat = None
    if ref.tip_desired is not None:
        tip_desired_concat = np.concatenate([ds.tip_desired for ds in datasets])

    tip_projected_concat = None
    if ref.tip_projected is not None:
        tip_projected_concat = np.concatenate([ds.tip_projected for ds in datasets])

    # Create concatenated dataset
    combined = NPZDataset(
        t=t_concat,
        currents=currents_concat,
        dt=ref.dt,
        L_inserted=ref.L_inserted,
        integration_step_size=ref.integration_step_size,
        tip_fk=tip_fk_concat,
        tip_dyn=tip_dyn_concat,
        tip_desired=tip_desired_concat,
        tip_projected=tip_projected_concat,
        filename=f"concatenated_{len(datasets)}_files",
    )

    return combined
