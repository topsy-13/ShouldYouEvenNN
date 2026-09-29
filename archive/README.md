# Historical research archive

`thesis-prototype/` preserves the project at commit
`b52a6e3b13c2b5f19139a63f09d56cba18291d5c` (`v3`), tagged as
`archive/thesis-v3`. The new foundation lives on `codex/research-foundation`.

The original README, source, scripts, notebooks, document drafts, submitted thesis,
figures, and saved results retain their relative layout and file contents. See
[the archive guide](thesis-prototype/ARCHIVE.md) before using them.

## Preservation manifest

`migration-manifest.json` inventories 893 originally tracked files and 501 ignored
local files before migration. Each entry records its original and destination
paths, original and intended destination tracking status, SHA-256 hash, byte size,
and whether preservation is required. The root `.gitattributes` stays in place.

The seven tracked bytecode files become ignored local cache files. Other disposable
caches are also excluded from the preservation guarantee. Research files remain
byte-for-byte unchanged.

Ignored research files were preserved locally, not added to Git. A clone or checkout
of the archive tag does **not** recover them. The manifest identifies these files
for separate backup or transfer. The original `.gitignore` inside the archived
project preserves its local ignore behavior; tracked historical CSVs remain tracked.

The manifest records the files actually available in this checkout. It does not
establish which code revision originally produced each historical result.
