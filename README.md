# ShouldYouEvenNN

**When should training a neural candidate continue?**

This repository is the foundation for a controlled study of early stopping relative
to a classical baseline. The intended outcome is a decision rule that saves compute
while rarely rejecting candidates that would achieve a meaningful improvement.

The active package currently contains no training, forecasting, or decision
implementation. The thesis prototype and its results are historical, exploratory
material; their limitations are documented in [the archive](archive/README.md).

For local GPU research, use the pinned [Conda environment](environment.yml) and
the [research environment setup](docs/development.md#local-gpu-research-environment).

## Quick start

Use Python 3.11 or newer and a fresh virtual environment. From the repository root:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

On Windows, `py -3.11` can replace `python` when creating the environment. The
checks exercise packaging only; they do not download data or train models.

## Repository map

| Directory | Purpose |
| --- | --- |
| `src/shouldyouevennn/` | Installable, reusable research code |
| `tests/` | Automated checks for active code |
| `configs/` | Versioned experiment configurations |
| `data/raw/` | Local source datasets, treated as immutable |
| `data/processed/` | Regenerable local datasets |
| `notebooks/` | Exploration and interpretation |
| `artifacts/runs/` | Generated run records, predictions, and checkpoints |
| `reports/figures/`, `reports/tables/` | Curated figures and tables for reporting |
| `docs/` | Research protocol and development guidance |
| `archive/thesis-prototype/` | Preserved original project layout |

Start with the [research protocol](docs/research-protocol.md) and
[research log](docs/research-log.md); use the
[development guidance](docs/development.md) for setup. Experiment design choices remain open
until specified in a versioned protocol; old constants are not active defaults.

Local datasets and run outputs are ignored by Git. Curated CSV tables may be
tracked. See [archive preservation details](archive/README.md) before assuming a
clone contains every historical artifact.
