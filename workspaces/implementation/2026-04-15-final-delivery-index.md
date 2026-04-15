# 2026-04-15 Final Delivery Index

## Purpose

This file is the final handoff index for the current competition-ready DiffAudit research package.

Use it as the single entry point when you need to:

- brief judges;
- answer technical questions;
- locate the strongest current evidence;
- defend the boundary of each claim.

## 1. Primary Entry Points

### Audit View

- main audit refresh:
  - `workspaces/implementation/reports/mainline-audit-20260415-final-refresh/summary.json`
- attack-defense table:
  - `workspaces/implementation/artifacts/unified-attack-defense-table.json`

### Presentation View

- competition brief:
  - `workspaces/implementation/2026-04-15-competition-brief.md`
- competition answer pack:
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`
- judge FAQ short:
  - `workspaces/implementation/2026-04-15-judge-faq-short.md`
- slide outline and speaker notes:
  - `workspaces/implementation/2026-04-15-slide-outline-and-speaker-notes.md`
- bilingual elevator pitch and rapid answers:
  - `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`
- one-page judge cheat sheet:
  - `workspaces/implementation/2026-04-15-one-page-judge-cheat-sheet.md`
- metric glossary and claim-boundary card:
  - `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`
- slide-to-evidence map:
  - `workspaces/implementation/2026-04-15-slide-to-evidence-map.md`
- research presentation rehearsal checklist:
  - `workspaces/implementation/2026-04-15-research-presentation-rehearsal-checklist.md`
- canonical numbers and wording sheet:
  - `workspaces/implementation/2026-04-15-canonical-numbers-and-wording-sheet.md`
- defense coverage and gap note:
  - `workspaces/implementation/2026-04-15-defense-coverage-and-gap-note.md`
- presentation asset manifest:
  - `workspaces/implementation/artifacts/presentation-asset-manifest.json`
- presentation asset checksums:
  - `workspaces/implementation/artifacts/presentation-asset-checksums.json`
- research package signoff:
  - `workspaces/implementation/2026-04-15-research-package-signoff.md`
- research package signoff json:
  - `workspaces/implementation/artifacts/research-package-signoff.json`
- research to leader handoff:
  - `workspaces/implementation/2026-04-15-research-to-leader-handoff.md`
- unified evidence snapshot:
  - `workspaces/implementation/2026-04-15-unified-evidence-snapshot.md`
- attack-defense matrix:
  - `workspaces/implementation/2026-04-15-attack-defense-matrix.md`
- threat-model comparison:
  - `workspaces/implementation/2026-04-15-threat-model-comparison.md`
- final evidence manifest:
  - `workspaces/implementation/artifacts/final-evidence-manifest.json`

## 2. Strongest Current Evidence

### Black-box

- admitted headline:
  - `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
- best single-metric rung:
  - `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- CLiD corroboration:
  - `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`
  - `workspaces/black-box/runs/clid-recon-clip-partial-target100-20260415-r1/summary.json`
- CLiD cross-check note:
  - `workspaces/black-box/2026-04-15-clid-local-crosscheck-note.md`
- Recon + CLiD fusion negative result:
  - `workspaces/black-box/runs/recon-clid-fusion-partial-target100-20260415-r1/summary.json`

### Gray-box

- admitted runtime baseline:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive/summary.json`
- same-scale scale-up baseline:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- same-scale defended comparator:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
- full-split corroboration:
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- gray-box comparison note:
  - `workspaces/gray-box/2026-04-15-pia-scale-vs-secmi-note.md`

### White-box

- mainline:
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- defended comparator:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`

## 3. What We Should Say Carefully

- `Recon` is the admitted black-box headline.
- `CLiD` is currently best described as workspace-verified local corroboration, not a paper-faithful benchmark.
- `PIA` and `SecMI` are strong local gray-box runtime evidence, but they still carry provenance/protocol boundaries.
- `GSA` should be presented as the privileged-access upper bound.

## 4. Readiness Status

- black-box readiness: complete
- gray-box readiness: complete
- white-box readiness: complete
- defense comparison readiness: complete
- unified evidence / matrix / threat model: complete
- competition-facing materials: complete
- research-line closure: complete
- next owner: `Leader`

## 5. Recommended Human Reading Order

1. `workspaces/implementation/2026-04-15-competition-brief.md`
2. `workspaces/implementation/2026-04-15-competition-answer-pack.md`
3. `workspaces/implementation/2026-04-15-slide-outline-and-speaker-notes.md`
4. `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`
5. `workspaces/implementation/2026-04-15-one-page-judge-cheat-sheet.md`
6. `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`
7. `workspaces/implementation/2026-04-15-slide-to-evidence-map.md`
8. `workspaces/implementation/2026-04-15-research-presentation-rehearsal-checklist.md`
9. `workspaces/implementation/2026-04-15-canonical-numbers-and-wording-sheet.md`
10. `workspaces/implementation/2026-04-15-defense-coverage-and-gap-note.md`
11. `workspaces/implementation/artifacts/presentation-asset-manifest.json`
12. `workspaces/implementation/artifacts/presentation-asset-checksums.json`
13. `workspaces/implementation/2026-04-15-research-package-signoff.md`
14. `workspaces/implementation/artifacts/research-package-signoff.json`
15. `workspaces/implementation/2026-04-15-research-to-leader-handoff.md`
16. `workspaces/implementation/reports/mainline-audit-20260415-final-refresh/summary.json`
17. `workspaces/implementation/artifacts/unified-attack-defense-table.json`

## 6. If We Need One Backup Page

- `workspaces/implementation/2026-04-15-unified-evidence-snapshot.md`
- `workspaces/implementation/2026-04-15-judge-faq-short.md`
- `workspaces/implementation/2026-04-15-slide-outline-and-speaker-notes.md`
- `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`
- `workspaces/implementation/2026-04-15-one-page-judge-cheat-sheet.md`
- `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`
- `workspaces/implementation/2026-04-15-slide-to-evidence-map.md`
- `workspaces/implementation/2026-04-15-research-presentation-rehearsal-checklist.md`
- `workspaces/implementation/2026-04-15-canonical-numbers-and-wording-sheet.md`
- `workspaces/implementation/2026-04-15-defense-coverage-and-gap-note.md`
- `workspaces/implementation/artifacts/presentation-asset-manifest.json`
- `workspaces/implementation/artifacts/presentation-asset-checksums.json`
- `workspaces/implementation/2026-04-15-research-package-signoff.md`
- `workspaces/implementation/artifacts/research-package-signoff.json`
- `workspaces/implementation/2026-04-15-research-to-leader-handoff.md`
