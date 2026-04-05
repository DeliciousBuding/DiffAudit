# References

This directory stores the repository-level reference index and the mirrored PDF materials used during research and reproduction.

## What changed

`references/materials/` is now organized with English-only folders:

- `black-box/`
- `gray-box/`
- `white-box/`
- `survey/`
- `context/`

The old Chinese folder layout has been removed.

## Index files

- `materials/manifest.csv`: machine-readable source of truth with path, track, scores, provenance and hashes
- `materials/README.md`: human-readable layout, naming rules and scoring rubric

## Maintenance rules

- Keep PDF file names ASCII-only and kebab-case.
- Prefer one canonical copy per paper.
- Record every new PDF in `manifest.csv`.
- Use `credibility_score` and `reference_value_score` to make triage explicit instead of relying on filename intuition.

Do not treat the presence of a PDF as proof that the paper has already been reproduced.
