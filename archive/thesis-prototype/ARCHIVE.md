# Thesis prototype: historical material

This directory preserves the original project layout. Its `README.md` is the
unchanged historical README, not instructions for the new research package.

## Execution assumptions

Many scripts assume the project root is their current working directory and append
`./src` to Python's import path. For historical investigation, their corresponding
root is now this directory; individual notebooks may have different assumptions.
Some scripts contain stale imports or incompatible entry points. The original
dependency environment was not fully specified.

Execution has not been repaired or certified by this migration. Running scripts
can download data, launch expensive training, and overwrite archived outputs.
Use a separate copy and a separately managed legacy environment if investigating
historical behavior. Do not add this directory to the active package's import path.

## Known research limitations

- Candidate reconstruction can change learning rates and activation functions;
  fidelity evaluation does not consistently preserve the forecasted continuation.
- Reported probability scores and confidence intervals lack demonstrated
  calibration; pruning mixes effort representations.
- Baseline selection and evaluation partitions need reconciliation.
- Budget accounting can exceed the stated limits.
- Saved decisions, summary tables, and thesis descriptions disagree in places;
  missing outcomes have sometimes been recorded as zero.
- Results do not yet establish the missed-opportunity rate of stopping decisions.

These files are exploratory evidence. Their preservation is not endorsement of
the implementation, statistical validity, or claims in the thesis. No scientific
behavior or historical output has been corrected in this structural migration.
