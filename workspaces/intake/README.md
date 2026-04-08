# Intake Index

This directory provides a single machine-readable entrypoint for routing gray-box / white-box lines into the next system step.

Each entry exposes both:

- `id`: research-owned intake identity
- `contract_key`: system-facing join key used by `Local-API` / `Platform`

Canonical file:

- [index.json](index.json)

Boundary note:

- `index.json` is the canonical machine-readable intake directory.
- It is not a complete admitted-results directory for every method.
- Cross-track admitted results for system consumption currently live in `../implementation/artifacts/unified-attack-defense-table.json`.

Validation (from repo root):

```bash
python Project/scripts/validate_intake_index.py
```
