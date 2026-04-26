# 2026-04-17 Noise-as-a-Probe Promotion-Gap Review

## Question

After `Noise as a Probe` became a strengthened bounded challenger candidate, what exact gap still blocks promotion into the packaged gray-box challenger layer, and is there any honest promotion path on the current local contract?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-protocol-asset-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-larger-rung-repeat-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-challenger-boundary-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-defended-extension-feasibility-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-summary-layer-sync-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-pia-vs-tmiadm-operating-point-comparison.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-second-defense-closure-verdict.md`

## Current Read

What the branch already has:

- a real latent-diffusion execution path on `SD1.5 + celeba_partial_target/checkpoint-25000`
- bounded `8 / 8 / 8` positive evidence
- two same-scale `16 / 16 / 16` positive rungs with one disjoint repeat
- a stable conservative-threshold story
- enough local strength to be mentionable above lane-local notes

What the branch still does **not** have:

- same-surface comparability with packaged `PIA` or `TMIA-DM`
- a defended branch
- a defended-extension design that preserves attack comparability
- a project-level operating-point comparison board
- paper-faithful parity

## Exact Promotion Blockers

### 1. The branch lives on a different contract surface

Current packaged gray-box hierarchy is built around:

- `DDPM / CIFAR-10`
- `PIA`
- `TMIA-DM`
- attack-side and defended-side operating-point comparisons

`Noise as a Probe` instead lives on:

- `latent-diffusion / SD1.5`
- `CelebA target-family LoRA`
- bounded local image-distance scoring

So the branch is not merely "smaller." It is sitting on a different contract surface from the packaged challenger board.

### 2. Promotion would currently force an apples-to-oranges hierarchy jump

`TMIA-DM` became the strongest packaged challenger because it already had:

- repeated GPU128 / GPU256 evidence
- same-family operating-point comparison against `PIA`
- defended-challenger evidence
- system-layer sync into the current gray-box board

`Noise as a Probe` has none of those comparability anchors on its own latent-diffusion contract.

### 3. Defended extension is not the missing shortcut

That path already closed as `no-go for now`.

Direct ports of:

- `stochastic-dropout(all_steps)`
- `temporal-striding(stride=2)`

would either fail structurally or redefine the attack rather than add a comparable defense.

So the promotion blocker is not "just add one defense rung."

## Honest Promotion Reading

Current honest reading is:

- `Noise as a Probe` is a real new latent-diffusion gray-box mechanism
- it is stronger than a speculative candidate
- but it still cannot honestly replace the packaged active challenger layer on the current repo-wide gray-box board

The missing piece is not another blind rung.

The missing piece is a comparable board.

## Is There An Honest Promotion Path On The Current Contract?

Current answer:

- `no, not on the current contract`

Reason:

1. current gray-box packaged hierarchy is still defined by the shared `DDPM/CIFAR10` board;
2. `Noise as a Probe` cannot currently join that board directly;
3. the branch also lacks its own defended/comparison layer on the latent-diffusion contract;
4. therefore any immediate promotion would overstate comparability rather than reveal new truth.

## Minimal Contract Shift Required Before Reopen

The smallest honest shift is **not** immediate GPU scale-up.

It is a CPU-side contract question:

- can the repo define one explicit latent-diffusion gray-box comparison contract where `Noise as a Probe` is compared against at least one other method on the same `SD1.5 + target-family` surface?

Without that shift, the branch should remain:

- `strengthened bounded challenger candidate`
- `mentionable`
- `not promotion-ready`

## Next Task

The next live lane should be:

- `Noise as a Probe contract-shift review`

It should answer:

1. whether a same-surface latent-diffusion comparison board is honest and worth building;
2. what minimum comparator or summary contract would be required;
3. whether the branch should stay indefinitely below packaged challenger status unless that board exists.

## Verdict

- `noise_as_probe_promotion_gap_verdict = negative but clarifying`
- current blocker is `comparability`, not `execution`
- current contract has `no honest direct promotion path`
- `gpu_release = none`
- `next_gpu_candidate = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `Platform/Runtime`: no immediate handoff
- `competition materials`: no immediate sync

Reason:

- this round changes internal lane selection and overclaim boundary, but not higher-layer packaging yet.
