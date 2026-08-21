# Positive Tensor-Network Kahler Metrics on gCICY Threefolds

This repository contains the source code accompanying the article
**"Positive Tensor-Network Kahler Metrics on gCICY Threefolds."** It provides
the numerical geometry, positive tensor-network models, training drivers,
evaluation routines, and data-independent deterministic tests used in the
reported calculations.

The repository is intentionally code-focused. Frozen models, evaluation
samples, pointwise arrays, and claim-reconstruction scripts are archived at
[Zenodo (doi:10.5281/zenodo.21963526)](https://doi.org/10.5281/zenodo.21963526).

## Mathematical scope

The implementation covers:

- product-projective coordinates, Fubini--Study data, implicit charts, and
  tangent maps;
- generalized gCICY sections, Poincare residues, complete-fibre sampling, and
  importance weights;
- restricted section bases and positive unrestricted Hermitian metrics;
- the positive tensor-network ansatz
  `F = ||B_theta s^(tensor m)||^2 + epsilon F_ref^m` and its first and mixed
  second derivatives;
- exact degree lifts, function-preserving bond growth, and local-dictionary
  transformations;
- native normalized Monge--Ampere objectives, residual-potential controls,
  full-H controls, tail statistics, and fibre-cluster bootstrap evaluation.

The correspondence between mathematical formulas and source files is
documented in
[`docs/MATHEMATICAL_IMPLEMENTATION_GUIDE.md`](docs/MATHEMATICAL_IMPLEMENTATION_GUIDE.md).
The programs entering the reported X11, X21, and X22 calculations are listed
in [`docs/FULL_PRODUCTION_DAG.md`](docs/FULL_PRODUCTION_DAG.md).

## Installation

Python 3.12 is recommended. The core tests run on CPU; CUDA is needed only for
large training calculations.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verification

Run the deterministic mathematical-kernel and pipeline tests with:

```bash
python RUN_CORE_TESTS.py
```

The expected result is `58 passed`. The complete frozen audit, including the
data-backed tests and their fixtures, is distributed with the Zenodo archive.

The independent post-v1 GPU control plane has a separate deterministic suite:

```bash
python RUN_EXPERIMENT_TESTS.py
```

Its baseline, registered study design, manifest, and operating instructions
are in [`experiments/`](experiments/README.md).  These files create only new
external run roots and do not regenerate or replace the paper artifacts.

## Repository layout

| Path | Contents |
| --- | --- |
| `gcicy_metric/` | Geometry, section, sampling, metric, and TN library |
| `scripts/` | Portable Python programs for training, evaluation, and audits |
| `tests/` | Data-independent mathematical and model tests |
| `docs/` | Mathematical implementation guide and production-program map |
| `experiments/` | Independent post-v1 baselines, protocol, GPU manifest, and safety schema |

The Python entry points use ordinary command-line arguments and
package-relative imports. Machine-specific launchers, registered protocols,
frozen models, numerical fixtures, and the complete 148-test audit are kept
in the Zenodo archive so that this public repository remains a portable,
source-only distribution.

## Reproducing the paper numbers

This repository is sufficient for source inspection, development, and the
core deterministic tests. Reconstructing the final tables and figures requires
the frozen numerical artifacts in the
[Zenodo archive](https://doi.org/10.5281/zenodo.21963526). Follow the README in
that archive for the claim-level replay procedure.

## License

The software in this repository is released under the MIT License. Numerical
data and documentation in the Zenodo archive are separately released under
CC BY 4.0.

## Citation

Please cite the associated article and the versioned Zenodo record. Citation
metadata for this software is provided in [`CITATION.cff`](CITATION.cff).
