# 2026-04-15 White-Box Gradient Extraction Direct Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-1`
- `track`: `white-box`
- `canonical_run`: `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r2\target_member-gradients.pt`
- `gpu_status`: `completed`

## Question

Is the current white-box gradient extraction path actually broken, or is the blocker caused by a narrower execution-layer mismatch?

## What Was Run

Two direct upstream reproductions were used against the admitted `GSA` target-member asset line.

### Direct reproduction A

- entrypoint:
  - `workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py`
- key setting:
  - `--attack_method 2`
- extra manual flag:
  - `--dataset_name cifar10`

Observed failure:

- `KeyError: 'image'`

Interpretation:

- this failure is caused by a dataset-mode mismatch;
- the current admitted asset line is imagefolder-like, so forcing `cifar10` drives the upstream loader into the wrong schema.

### Direct reproduction B

Same direct run, but aligned to the repository wrapper contract:

- removed `--dataset_name`
- kept `--attack_method 2`
- kept the admitted target checkpoint root
- kept the admitted target-member dataset

Observed result:

- gradient extraction ran through the full `1000` sample target-member split
- the first failure was only:
  - output parent directory missing at `torch.save(...)`

After creating the output directory and replaying the same job:

- the direct run completed successfully
- `target_member-gradients.pt` was written under:
  - `workspaces/white-box/runs/gsa-gradient-direct-attackmethod2-20260415-r2/`

## Verdict

Current verdict:

- `positive`

Precise root-cause classification:

1. the current white-box gradient extraction path is runnable;
2. `attack_method = 2` is not the blocker on the admitted asset line;
3. the prior blocker collapses to execution-layer issues:
   - wrong dataset mode (`--dataset_name cifar10`)
   - missing output-directory creation before `torch.save`

## What Changes In Mainline Truth

- `WB-1` should no longer be framed as “gradient extraction may be fundamentally broken”.
- the honest current boundary is:
  - direct extraction works on the admitted asset line,
  - but the runner contract is brittle and needs explicit path hygiene.

## Next Best Move

Advance to `WB-2` from a narrower starting point:

1. reuse the direct extraction contract that just worked;
2. decide whether the second white-box verdict should be a `GSA2` comparator or a stronger no-go note for other branches;
3. if the repo wrapper is updated later, make it create parent directories before saving gradient payloads.

## Handoff Guidance

- `Platform / Runtime`: no immediate schema change required.
- `competition materials`: no wording upgrade needed; this is an unblock / execution-contract verdict, not a new admitted white-box metric.
