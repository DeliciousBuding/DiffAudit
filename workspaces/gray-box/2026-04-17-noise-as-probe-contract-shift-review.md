# 2026-04-17 Noise-as-a-Probe Contract-Shift Review

## Question

After confirming that `Noise as a Probe` has no honest direct promotion path on the current repo-wide gray-box board, is it worth establishing a separate latent-diffusion same-surface comparison contract, or would that still create low-value pseudo-comparability?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-17-noise-as-probe-promotion-gap-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-protocol-asset-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-larger-rung-repeat-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-protocol-asset-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-current-contract-hold-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-final-cpu-reviewrung-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-structural-memorization-smoke-note.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-new-family-structural-memorization-feasibility.md`

## Candidate Same-Surface Board

On paper, the smallest latent-diffusion same-surface board would be:

- `Noise as a Probe`
- `MoFit`
- `structural memorization`

They are all at least partially grounded on the same local target-family direction:

- `SD1.5`
- `CelebA`-style target-family surface
- local target-family LoRA or closely related target-side contract

So the contract-shift question is real, not imaginary.

## Why The Shift Still Fails Right Now

### 1. Only one branch is actually repeat-positive

`Noise as a Probe` currently has:

- repeat-positive bounded evidence
- stable threshold behavior
- a clean “execution-real but still below promotion” reading

`MoFit` currently has:

- execution truth
- but only a tiny favorable gap under the current contract
- explicit `current-contract hold`

`structural memorization` currently has:

- execution truth
- but the local signal points in the wrong direction

So a same-surface board built now would not compare multiple active latent-diffusion contenders.

It would mostly compare:

- one real bounded contender
- one hold branch
- one negative branch

### 2. Score semantics are still too heterogeneous

The three branches do not currently emit the same kind of object:

- `Noise as a Probe` uses final image-distance after inversion + custom-noise replay
- `MoFit` uses fitted-condition / surrogate-loss style score traces
- `structural memorization` uses structure-preservation reconstruction scores

That is still useful for lane-local truth.

But it is not yet a strong enough basis for a packaged same-surface challenger board.

### 3. The board would not yet change the project-level story honestly

If this contract-shift were executed now, the likely headline would still be:

- `Noise as a Probe` is the only latent-diffusion branch with repeat-positive bounded signal
- `MoFit` is hold
- `structural memorization` is negative

That does not improve the packaged story enough to justify adding another board layer right now.

## Decision

Current decision:

- `do not build the latent-diffusion same-surface board yet`

Reason:

1. there is not yet a second same-surface branch strong enough to make the board decision-relevant;
2. current score semantics remain too heterogeneous for an honest packaged comparator board;
3. building the board now would create bookkeeping, not a stronger research truth.

## Reopen Rule

Only reopen this contract-shift if at least one of the following becomes true:

1. `MoFit` gets a materially changed contract and produces more than tiny weak-positive separation;
2. `structural memorization` flips to direction-positive under a new faithful hypothesis;
3. another latent-diffusion gray-box family lands on the same target-family surface with credible bounded signal;
4. the project explicitly needs a latent-diffusion auxiliary board for materials and that board can be labeled as non-packaged supporting evidence only.

## Next Task

The next live task should leave this latent-diffusion sub-branch and return to broader lane selection:

- `gray-box post-noise contract-shift reselection review`

That review should decide whether gray-box should now:

1. stay on `no-new-gpu-question` and pivot to a different family/box;
2. reopen one latent-diffusion branch only if a materially changed hypothesis exists;
3. or move to another system-consumable research task with higher blocker leverage.

## Verdict

- `noise_as_probe_contract_shift_verdict = negative but clarifying`
- current latent-diffusion same-surface board is `not worth building now`
- `gpu_release = none`
- `next_gpu_candidate = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `Platform/Runtime`: no immediate handoff
- `competition materials`: no immediate sync

Reason:

- this round prevents low-value contract inflation, but does not change the packaged project narrative.
