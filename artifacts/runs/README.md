# Run records

Future runners must record the resolved configuration, Git revision and dirty
state, Python and dependency versions, relevant hardware, dataset version and
split identifiers, seeds, and explicit completion or failure status.

Keep metrics, predictions, checkpoints, and timing with that record. Distinguish
missing outcomes from measured zero values. Define the concrete format alongside
the first implemented experiment rather than inventing an unused schema now.
