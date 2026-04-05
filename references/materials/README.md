# Materials Index

`references/materials/` stores the local PDF mirror used by the DiffAudit research workflow.

## Layout

```text
materials/
  black-box/
  gray-box/
  white-box/
  survey/
  context/
  manifest.csv
  README.md
  paper-index.md
```

Rules:

- Use English folder names only.
- Use ASCII lowercase kebab-case file names.
- Prefer `year-venue-short-title.pdf`.
- Keep one canonical copy for the same paper whenever possible.

## Scoring Rubric

`credibility_score`

- `5`: peer-reviewed flagship venue or official publisher copy
- `4`: arXiv or OpenReview preprint with clear research relevance
- `3`: internal context document or local mirror with known provenance
- `2`: mirrored or partially normalized material with unclear provenance
- `1`: unclassified legacy archive item

`reference_value_score`

- `5`: core paper for the current DiffAudit implementation route
- `4`: strong secondary reference with direct design value
- `3`: useful scenario extension or supporting background
- `2`: weak or indirect reference
- `1`: not yet normalized enough for active research use

## Canonical Highlights

| Track | Canonical files |
| --- | --- |
| `black-box` | `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`, `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`, `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf` |
| `gray-box` | `2023-icml-secmi-membership-inference-diffusion-models.pdf`, `2024-iclr-pia-proximal-initialization.pdf`, `2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`, `2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf` |
| `white-box` | `2025-popets-white-box-membership-inference-diffusion-models.pdf`, `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` |
| `context` | `diffaudit-team-onboarding.pdf`, `diffaudit-product-requirements.pdf` |
| `survey` | `2025-tracing-the-roots-leveraging-temporal-dynamics-in-diffusion-trajectories-for-origin-attribution.pdf`, `2025-dp-docldm-differentially-private-document-image-generation-using-latent-diffusion-models.pdf`, `2025-privacy-preserving-low-rank-adaptation-against-membership-inference-attacks-for-latent-diffusion-models.pdf`, `2024-dual-model-defense-safeguarding-diffusion-models-from-membership-inference-attacks-through-disjoint-data-splitting.pdf`, `2025-diffence-fencing-membership-privacy-with-diffusion-models.pdf`, `2025-defending-diffusion-models-against-membership-inference-attacks-via-higher-order-langevin-dynamics.pdf`, `2026-inference-attacks-against-graph-generative-diffusion-models.pdf`, `2025-perturb-a-model-not-an-image-towards-robust-privacy-protection-via-anti-personalized-diffusion-models.pdf`, `legacy-survey-archive-index.pdf` |

## Source of Truth

`manifest.csv` is the full index.

Columns:

- `relative_path`: canonical path under `references/materials/`
- `primary_track`: primary physical category
- `track_tags`: multi-track labels when a paper spans more than one threat model
- `credibility_score`: source quality score from `1` to `5`
- `reference_value_score`: project usefulness score from `1` to `5`
- `source_url`, `accessed_at`, `sha256`, `size_bytes`, `license_note`, `notes`: provenance and maintenance fields

When adding a new PDF, update `manifest.csv` in the same commit.
