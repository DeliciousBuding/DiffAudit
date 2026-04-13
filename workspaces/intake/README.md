# Intake Index

This directory provides a single machine-readable entrypoint for routing gray-box / white-box lines into the next system step.

Each entry exposes both:

- `id`: research-owned intake identity
- `contract_key`: system-facing join key used by `Local-API` / `Platform`

Canonical files:

- [index.json](index.json)
- [phase-e-candidates.json](phase-e-candidates.json)

Boundary note:

- `index.json` is the canonical machine-readable intake directory.
- `index.json.entries[]` remains the promoted-contract surface that can be joined to `Local-API` / `Platform`.
- `phase-e-candidates.json` is a research-owned candidate ordering supplement only.
- `phase-e-candidates.json` is non-routable, non-admitted, non-benchmark, and must not be treated as a job release surface.
- It is not a complete admitted-results directory for every method.
- Cross-track admitted results for system consumption currently live in `../implementation/artifacts/unified-attack-defense-table.json`.

Validation (from repo root):

```bash
python Research/scripts/validate_intake_index.py
```
