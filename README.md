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
from model import compute_frequencies

# (N, 2) array of electric-field projections along each O–H bond, in V/Å.
# Normally these come from MD; here is a small illustrative range.
example_fields = np.zeros([10, 2])
example_fields[:, 0] = np.linspace(0, -2, 10)

compute_frequencies(example_fields, "params/amoeba_params.json", seed=10)
# -> writes results.npz (freqs, activities, fields, eigenvalues)
```

See `examples/examples.ipynb` for a worked example showing the effect of the
field on the symmetric stretch.

### Parameters

`compute_frequencies(projs, param_file, seed=10)`

- `projs` — `(N, 2)` array of electric-field projections along the two O–H bonds (V/Å).
- `param_file` — JSON file of model parameters. Two water models are provided in `params/`:
  - `params/amoeba_params.json` — for the AMOEBA polarizable model.
  - `params/spce_params.json` — for the SPC/E model.
- `seed` — random seed for the bending-energy multipliers (default `10`).

Results are written to `results.npz` containing `freqs`, `activities`, `fields`,
and `eigenvalues`.

## Module layout

| File | Role |
|------|------|
| `model.py` | Main driver: `compute_frequencies` orchestrates the full calculation. |
| `normal_modes.py` | Harmonic normal-mode analysis and eigenvalue solver. |
| `polarizability.py` | Molecular polarizability model (`onepol`). |
| `dipole.py` | One-body and two-body dipole models (`onedip`, `twodip`). |
| `potentials.py` | Potential-energy surface (`fastonebody`). |
| `basis.py` | Vibrational basis functions and numerical derivatives. |
| `geometry.py` | Coordinate transforms between internal and Cartesian frames. |
| `params/` | Model parameters: `constants.py` (physical constants and equilibrium geometry) and the per-model JSON files (`amoeba_params.json`, `spce_params.json`). |

## Data

`data/` holds the pre-trained models loaded at import time:

- `polarizability_krr.joblib` — kernel ridge model for the polarizability.
- `one_body_ridge.joblib` — ridge model for the one-body potential.
- `one_body_shifts.npy` — one-body energy shifts.

## License

Copyright ©2026. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
