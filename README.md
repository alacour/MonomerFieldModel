# MonomerFieldModel

Compute Raman and IR vibrational spectra of water molecules from local electric
fields. Given the electric field projected along each O–H bond (e.g. sampled from
an MD trajectory), the code solves the anharmonic vibrational Schrödinger equation
on a 3D normal-mode grid and returns Raman transition polarizabilities and IR
transition dipoles for the symmetric stretch, asymmetric stretch, and bend.

## Installation

```bash
conda env create -f environment.yml
conda activate raman-spectra-water
```

Requires Python ≥ 3.10 with NumPy, SciPy, scikit-learn, joblib, and matplotlib
(see `environment.yml`).

## Usage


```python
import numpy as np
from calculate import compute_freqs

# (N, 2) array of electric-field projections along each O–H bond, in V/Å.
# Normally these come from MD; here is a small illustrative range.
example_fields = np.zeros([10, 2])
example_fields[:, 0] = np.linspace(0, -2, 10)

compute_freqs(example_fields, "amoeba_params.json", seed=10)
# -> writes results.npz (freqs, activities, fields, eigenvalues)
```

See `examples/examples.ipynb` for a worked example showing the effect of the
field on the symmetric stretch.

### Parameters

`compute_freqs(projs, param_file, seed=10)`

- `projs` — `(N, 2)` array of electric-field projections along the two O–H bonds (V/Å).
- `param_file` — JSON file of model parameters. Two water models are provided:
  - `amoeba_params.json` — for the AMOEBA polarizable model.
  - `spce_params.json` — for the SPC/E model.
- `seed` — random seed for the bending-energy multipliers (default `10`).

Results are written to `results.npz` containing `freqs`, `activities`, `fields`,
and `eigenvalues`.

## Module layout

| File | Role |
|------|------|
| `calculate.py` | Main driver: `compute_freqs` orchestrates the full calculation. |
| `normal_modes.py` | Harmonic normal-mode analysis and eigenvalue solver. |
| `polarizability.py` | Molecular polarizability model (`onepol`). |
| `dipole.py` | One-body and two-body dipole models (`onedip`, `twodip`). |
| `potentials.py` | Potential-energy surface (`fastonebody`). |
| `basis.py` | Vibrational basis functions and numerical derivatives. |
| `geometry.py` | Coordinate transforms between internal and Cartesian frames. |
| `constants.py` | Physical constants and equilibrium geometry. |
| `data_loader.py` | Helpers for loading/sorting saved spectra. |
| `job.py` | Batch/job submission helper. |
| `pressjob.sh` | Cluster submission script. |

## Data

`data/` holds the pre-trained models loaded at import time:

- `polarizability_krr.joblib` — kernel ridge model for the polarizability.
- `one_body_ridge.joblib` — ridge model for the one-body potential.
- `one_body_shifts.npy` — one-body energy shifts.
