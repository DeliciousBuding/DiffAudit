# 2026-04-16 MoFit Real-Asset Canary Launch Gate Review

## Question

After `GB-31`, is it honest to immediately launch a fresh real local CPU canary on the admitted `MoFit` stack, or should the launch defaults be tightened first to avoid wasting CPU on an over-wide first run?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-script-level-canary-implementation-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`

## Current Launch Surface

The script is now runnable, but the default first-run shape is still relatively wide for a CPU-first launch:

- `member_limit = 1`
- `nonmember_limit = 1`
- `surrogate_steps = 8`
- `embedding_steps = 12`
- `max_timestep = 140`
- `device = cpu`

That means even the first bounded run would perform repeated gradient-bearing `UNet` calls over:

- 2 real rows
- 20 optimization steps total per row
- high-dimensional latent and text-embedding tensors on the full local target-family stack

## Decision

Current launch-gate verdict:

- do **not** auto-launch the fresh real-asset CPU canary yet

Reason:

1. the script-level orchestration is now real, but it has not yet been cost-tightened for the very first admitted local launch;
2. the current default shape is still better treated as a post-gate configuration than as the safest first CPU canary;
3. the next honest step is to tighten launch defaults or add a clearly bounded launch profile before spending real local runtime.

## Recommended Next Step

Before the first admitted local launch, implement one of:

- tighter CPU-first defaults, or
- an explicit `bounded-launch` profile with smaller optimization budgets

Minimum honest first-launch target:

- `member_limit = 1`
- `nonmember_limit = 1`
- `surrogate_steps <= 1`
- `embedding_steps <= 2`
- keep `device = cpu`

## Verdict

- `real_asset_canary_launch_gate_verdict = hold-before-launch`
- the next live task should be launch-budget tightening, not immediate real execution
- `gpu_release = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
