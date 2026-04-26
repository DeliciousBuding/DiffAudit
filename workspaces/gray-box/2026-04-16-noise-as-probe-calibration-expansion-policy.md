# 2026-04-16 Noise-as-a-Probe Calibration / Expansion Policy

## Question

After the first successful `1 + 1` interface canary, what is the smallest honest next policy for calibration and the first expansion rung?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-interface-canary-verdict.md`
- `workspaces/black-box/runs/clid-recon-clip-partial-target100-20260415-r1/datasets/member/metadata.jsonl`
- `workspaces/black-box/runs/clid-recon-clip-partial-target100-20260415-r1/datasets/nonmember/metadata.jsonl`

## Available Pool

- `member_count = 100`
- `nonmember_count = 100`

## Smallest Honest Calibration Policy

The smallest honest calibration policy is:

- reserve `8` prior non-members as calibration-only samples
- do not reuse them inside the first evaluation rung
- use them only to define a simple percentile-style threshold candidate later

Why `8`:

1. it is large enough to be more honest than `1` or `2`
2. it is still cheap enough for the first bounded expansion
3. it preserves most of the current `100` non-member pool for later runs

## First Expansion Rung

The first bounded rung after `1 + 1` should be:

- `8 members + 8 evaluation non-members`
- plus `8 calibration non-members`

Interpretation:

- this is still an interface-oriented bounded rung
- not a benchmark
- not a release packet

## Boundary

This policy still does **not** justify:

- headline attack claims
- promoted gray-box challenger status
- GPU sweep release

## Verdict

- `policy_verdict = positive`
- `smallest_calibration_set = 8 prior non-members`
- `first_expansion_rung = 8 members + 8 eval non-members + 8 calibration non-members`
- `gpu_release = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is still bounded research execution policy only.
