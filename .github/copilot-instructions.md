# DiffAudit Research Copilot Review Instructions

This repository is the research source of truth for DiffAudit.

Prioritize review comments on:

- `src/diffaudit/`, `tests/`, `scripts/`, and `configs/`
- experiment orchestration bugs
- path handling, reproducibility, provenance, and benchmark contract drift
- false claims of reproduction, paper alignment, or asset readiness
- unsafe assumptions in evaluation logic, manifests, and summary generation

Do not spend review budget on:

- style-only comments when behavior is unchanged
- generated experiment outputs unless they indicate broken provenance or wrong paths
- paper PDFs, reference mirrors, or long literature notes
- speculative feature requests unrelated to the changed code

When reviewing code in this repository:

- treat `smoke`, `preview`, and `toy` as materially different from real reproduction
- flag missing tests when behavior, manifests, or CLI contracts changed
- call out path hard-coding, machine-specific roots, and hidden local dependencies
- prefer findings about scientific validity, reproducibility, and regression risk
- be explicit when a claim is unsupported by the changed evidence

When reviewing docs and workspace notes:

- focus on factual inconsistency, overstated status, or missing blockers
- do not rewrite tone unless the wording changes technical meaning

Keep comments concise, concrete, and evidence-driven.
