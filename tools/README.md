# Script layout

Commands are grouped by their operational role:

- `runtime/`: normal racing entry points and the Isaac launcher.
- `training/`: prior creation, training, search, and checkpoint migration.
- `evaluation/`: offline audits, evidence verification, and summaries.
- `diagnostics/`: bounded simulator and contact probes.
- `assets/`: native recording and media maintenance.
- `artifacts/`: run/legacy artifact indexing and migration helpers.
- `environment/`: local environment setup and inspection.
- `hairpin/`: legacy bounded hairpin experiments retained for reproducibility.

Run commands from the repository root. New output is isolated under `runs/<run-id>/`; `output/racing/` is the immutable legacy archive.
