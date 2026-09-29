# Development

## Environment

Use Python 3.11+ and a fresh `.venv` as described in the root README. Install with
`python -m pip install -e ".[dev]"`. There are no scientific runtime dependencies
yet. The build backend and direct development tools are pinned in `pyproject.toml`;
this is not a complete transitive environment lock or a reconstructed thesis
environment. Capture the full resolved environment with future experiment runs.

On this Windows machine, the newly created environment initially used pip 24.0,
which failed to verify the download certificate chain. Installation succeeded with
Windows' trusted certificate store enabled, keeping HTTPS verification active:

```sh
python -m pip install --use-feature=truststore -e ".[dev]"
```

Use this variant if the same certificate-chain error occurs with pip 24.0.

## Checks

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

CI installs the regular package and runs these checks on Python 3.11 on Windows and
Linux. Tests launch isolated Python subprocesses from temporary directories, so
successful imports cannot rely on the checkout being the working directory.
Package discovery is limited to `src/shouldyouevennn/`; archive code is excluded
from package, test, and lint discovery.

To inspect a distributable without installing scientific dependencies:

```sh
python -m pip wheel --no-deps . --wheel-dir dist
```

The wheel must contain only the active package and distribution metadata. Build
outputs are ignored. Packaging does not imply that any scientific claims have
been validated.

## Working conventions

Put reusable code in the installed package. Add modules when they have a concrete
purpose; avoid placeholder model hierarchies, catch-all utilities, and duplicate
experiment implementations. New code must not import the historical prototype.

Document the research protocol before coding its decision rule. Keep scientific
changes separate from artifact migrations. Use explicit inputs and output paths;
imports must not start experiments, download data, or write outputs. Keep tests
focused on meaningful invariants and small offline examples.

Version configurations and curated reports. Preserve raw inputs, record run
provenance, and keep local generated outputs in their designated ignored folders.
Do not reinterpret or rewrite archived results as part of routine cleanup.
