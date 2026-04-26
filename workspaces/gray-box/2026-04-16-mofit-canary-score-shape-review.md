# 2026-04-16 MoFit Canary Score-Shape Review

## Question

Does the first fresh `MoFit` local CPU canary show enough score directionality to justify immediate rung expansion, or is the right next move still a bounded CPU micro-rung design/review step?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\records.jsonl`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\traces\surrogate\member-000.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\traces\embedding\member-000.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\traces\surrogate\nonmember-000.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-canary-20260416-cpu-r4\traces\embedding\nonmember-000.json`

## What The Canary Actually Shows

### 1. The execution path is stable enough to optimize

Both samples show monotonic loss descent:

- member surrogate: `0.4160`
- member embedding: `0.3837273 -> 0.3834036`
- nonmember surrogate: `0.5113569`
- nonmember embedding: `0.4752950 -> 0.4749426`

So the canary is not failing because the optimization loops are completely inert.

### 2. The current score gap is directionally weak

Observed `mofit_score`:

- member: `-0.0003621`
- nonmember: `-0.0006366`

Interpretation:

- both scores are very close to zero
- both are negative
- the member/nonmember separation is extremely small at the current budget

### 3. That is not yet a scale-positive signal

The canary proves feasibility, but it does **not** yet prove:

- stable score direction
- useful separation
- immediate value in jumping to a larger rung

## Decision

Current review verdict:

- do **not** jump directly from this canary to a larger smoke rung

Best next task:

- `CPU micro-rung design / review`

Reason:

1. the execution path is now real, so there is no need to keep debating scaffolding;
2. the current budget is intentionally tiny, so the weak gap is not enough to reject the family yet;
3. the next honest question is whether a still-bounded but slightly larger CPU rung changes the score shape materially.

## Recommended Next Budget Envelope

- keep `device = cpu`
- keep `member_limit <= 2`
- keep `nonmember_limit <= 2`
- consider:
  - `surrogate_steps = 2`
  - `embedding_steps = 4`

## Verdict

- `canary_score_shape_verdict = inconclusive but still alive`
- `gpu_release = none`
- next step should be bounded CPU micro-rung design/review, not GPU escalation

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
