# Research foundation migration verification

Verified locally on Windows with Python 3.11.11, pytest 8.3.5, Ruff 0.11.13, and
setuptools 75.6.0 for distribution builds.

## Preservation

- Source: `b52a6e3b13c2b5f19139a63f09d56cba18291d5c`, retained by `v3` and the
  annotated tag `archive/thesis-v3`.
- Destination branch: `codex/research-foundation`.
- All 1,394 inventoried local files match their original SHA-256 hashes after the
  migration, including the disposable caches that were retained locally.
- All 886 retained historical tracked files match their original Git blobs. This
  includes the root `.gitattributes`, which did not move.
- All 501 originally ignored files remain ignored at their destination paths.
  Seven formerly tracked bytecode files are now also ignored, for 508 verified
  ignored destinations. No inventoried research file was discarded.
- The archived MLP experiment filename was normalized to its original Git spelling,
  `MLP_basic_experiment.py`, resolving a Windows worktree case mismatch without
  changing its contents.
- Generated data/run paths are ignored; example curated CSV paths in `reports/`
  and `configs/` are not ignored.

See [the migration manifest](../archive/migration-manifest.json) for per-file
paths, hashes, and tracking states. Ignored artifacts require separate backup;
these preservation checks do not make them available in a clone.

## Package and checks

- Editable installation in a fresh project-local virtual environment succeeded.
- `python -m pytest`: two tests passed for both editable and regular wheel installs.
  Tests import the package from outside the checkout using isolated Python processes.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed.
- Ruff file discovery includes only `pyproject.toml` and active source/test files.
- The built wheel contains only `shouldyouevennn/__init__.py` and distribution
  metadata. The source distribution also excludes the archive and old experiments.
- Git whitespace checks passed. Historical source and notebook contents remain
  unchanged.

CI has been configured for Windows and Linux on Python 3.11, but hosted CI has not
been run because this branch has not been pushed. No benchmark was rerun and no
scientific result was validated by these structural checks.
