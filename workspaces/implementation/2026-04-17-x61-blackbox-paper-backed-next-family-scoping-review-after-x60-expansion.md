# 2026-04-17 X-61 Black-Box Paper-Backed Next-Family Scoping Review After X-60 Expansion

## Status Panel

- `owner`: `ResearcherAgent`
- `task_type`: `cpu-first scoping review`
- `device`: `cpu`
- `verdict`: `negative but useful`

## Question

After `X-60` restored black-box as the only honest remaining non-graybox candidate surface, does the current paper-backed black-box backlog contain one genuinely new family that should now become the next live lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/paper-matrix-2024-2026.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/experiment-entrypoints.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-17-blackbox-next-family-candidate-generation-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-15-blackbox-new-family-semantic-auxiliary-classifier-feasibility.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-15-clid-paper-alignment-audit.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-08-variation-local-track.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-16-variation-asset-contract-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/kandinsky-diagnostics/README.md`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/black-box/2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models-report.md`
- `<DIFFAUDIT_ROOT>/Download/manifests/research-download-manifest.json`

## Review

### 1. Existing black-box families remain frozen where they already were

- `Recon` is still the main evidence line.
- `semantic-auxiliary-classifier` is already the landed leading challenger.
- `CLiD` is still boundary-quality corroboration.
- `variation` is still `contract-ready but blocked`.
- `Kandinsky` still sits in diagnostics only and has no experiment contract.

So `X-61` is not allowed to relabel any of these as a fresh family.

### 2. The only real paper-backed residual candidate is the face-image LDM line

The VISAPP 2025 face-image paper is the only remaining paper-backed black-box candidate not already frozen inside the current four-family wording.

But its actual shape is:

- face-domain-specific;
- dataset-level or collection-level inference;
- proxy-positive generation plus offline attack-classifier training;
- dependent on a separate face-tuning and attack-classifier pipeline.

That means it does **not** cleanly instantiate as:

- a single-sample black-box family beside `Recon`,
- a ready strict-API baseline beside `variation`,
- or a prompt-conditioned scorer beside `CLiD`.

### 3. In current repo taxonomy, the face-image line collapses below promotion

Why it does not honestly promote now:

1. It overlaps structurally with collection-level audit logic already absorbed through gray-box `CDI`.
2. Its returned-image classifier flavor also overlaps with the already-landed `semantic-auxiliary-classifier` family, just under a more domain-specific and dataset-level contract.
3. The repo does not currently own the required face-specific fine-tune + generated-positive-set + ResNet attack-classifier execution stack as a bounded local family contract.
4. No downloaded asset group or active workspace contract currently upgrades it from paper backlog to runnable candidate.

So the paper is useful for black-box taxonomy and future domain-specific risk narrative, but not as the next honest live black-box family.

## Verdict

- `x61_blackbox_paper_backed_scoping_verdict = negative but useful`
- black-box still does **not** expose one genuinely new promotable family beyond:
  - `Recon`
  - `semantic-auxiliary-classifier`
  - `CLiD`
  - `variation`
- the face-image LDM paper is best classified as:
  - `domain-specific collection-level risk note`
  - below immediate family promotion
- `active_gpu_question = none`
- `next_gpu_candidate = none`

## Reopen Rule

Black-box may reopen only if one of these becomes true:

1. a new family stays distinct from both `semantic-auxiliary-classifier` and `CDI`;
2. a runnable local contract appears for the face-image collection-level route;
3. a real asset or boundary shift upgrades `variation`, `CLiD`, or another paper-backed family into a genuinely new bounded question.

## Next Lane

- `X-62 non-graybox next-lane reselection after X-61 black-box scoping`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/black-box/plan.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- `Platform / Runtime`: no immediate handoff required
