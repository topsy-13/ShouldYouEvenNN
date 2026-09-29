# Local compute assessment

Observed 2026-09-29 on the user's Windows PC. This is a capability check, not a
training benchmark or an experiment environment lock.

| Component | Observation |
| --- | --- |
| CPU | AMD Ryzen 7 7700X, 8 cores (reported model), 16 logical processors |
| RAM | 31.12 GiB usable; 10.61 GiB available at inspection |
| GPU | NVIDIA GeForce RTX 4060, 8 GiB VRAM |
| GPU memory use | About 1.5 GiB occupied at inspection; shared with desktop applications |
| NVIDIA driver | 591.86 |
| Workspace drive | About 269.6 GiB free on C: |

Hardware was read through NVIDIA-SMI, the CPU registry entry, Windows memory API,
and filesystem capacity. CIM queries were denied in the current execution context;
the alternative read-only checks above succeeded. Available resources fluctuate.

## Python environments

The repository `.venv` runs Python 3.11.11 with pytest 8.3.5 and Ruff 0.11.13.
It contains none of the inspected scientific packages, consistent with the
foundation-only `pyproject.toml`.

The existing `C:/Users/user/anaconda3/envs/AutoML/python.exe` runs Python 3.11.11
and has the following directly inspected packages:

| Package | Installed version |
| --- | --- |
| PyTorch | 2.6.0+cu126 |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| OpenML | 0.15.0 |
| Matplotlib | 3.9.3 |

All listed imports succeeded, including scikit-learn's
`HistGradientBoostingClassifier`. PyTorch reported CUDA available and its CUDA
runtime as 12.6. A 64-by-64 tensor multiplication and backward pass executed on
the GPU, with the expected forward value and finite gradients. `python -m pip
check` reported no broken requirements. This verifies basic execution, not the
performance or reproducibility of a future training pipeline.

XGBoost and LightGBM are absent from this environment. They are not prerequisites
if the pilot uses scikit-learn's histogram gradient boosting. pytest and Ruff are
absent here but present in the project `.venv`.

The inspected `deepgpu` environment uses Python 3.10.18, below the active project's
Python 3.11 minimum. The `GPU_machine` directory has no executable at the expected
`python.exe` path. Neither is the proposed environment for this project.

## Implication for the pilot

The machine supports a modest local tabular MLP study. Start with one GPU training
job at a time, bounded CPU parallelism, and datasets that fit within available RAM.
Do not assume the nominal 32 GB RAM or 8 GB VRAM is entirely free. Determine dataset
count, portfolio size, seeds, and endpoint from an actual development timing check;
no epoch-time or total runtime estimate has been measured yet.

The user authorized use of this PC; a numeric wall-time budget was not specified.
Before collection, declare the scientific dependencies in the project and capture
a reproducible environment with resolved versions. The working AutoML environment
provides a tested starting point; do not export its unrelated packages as project
requirements. No packages were installed or upgraded during this assessment.

## New research environment: verified after setup

Later on 2026-09-29, at the user's request, we created the separate
`shouldyouevennn-research` Conda environment from [environment.yml](../environment.yml)
and installed the active package in editable mode. Its Python is 3.11.11 and its
PyTorch is **2.12.0+cu130**, using CUDA 13.0. Other directly pinned scientific
versions match the table above. It also includes the project's pytest and Ruff.

The newer PyTorch version was chosen after the initial 2.6 wheel download stalled:
a complete locally cached 2.12 wheel was available, and its SHA256 matched the
official package index. All scientific imports and a GPU forward/backward check
passed with this version on the RTX 4060. `pip check`, both packaging tests,
Ruff linting, and formatting checks passed in the new environment. The packaging
tests include importing the installed project from outside the checkout.

The declared Python, pip, and direct package versions were checked against the
installed environment. Local installation logs and a resolved package snapshot
are ignored under `artifacts/runs/environment-setup/`. The earlier AutoML and
`.venv` observations above are retained as the pre-setup record.

Activate for subsequent work with `conda activate shouldyouevennn-research`.
