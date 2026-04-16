# 2026-04-16 MoFit CPU Micro-Rung Score Review

## Question

After the first valid `2x2 / 2+4 / cpu` `MoFit` micro-rung, does the score shape now justify another bounded expansion, or is the family still too weak to spend more budget on immediately?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\records.jsonl`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\traces\embedding\member-000.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\traces\embedding\nonmember-000.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-cpu-microrung-design.md`

## What The Micro-Rung Shows

### 1. Direction is now more stable than the canary

Observed `mofit_score`:

- member:
  - `-0.0018386`
  - `-0.0019949`
- nonmember:
  - `-0.0022310`
  - `-0.0023055`

Mean score:

- member mean: `-0.0019168`
- nonmember mean: `-0.0022683`
- gap: `+0.0003515`

Interpretation:

- member scores are consistently less negative than nonmember scores
- this is directionally better than the first `1x1` canary
- so the family is not execution-positive but signal-dead

### 2. Magnitude is still weak

Even after the rung increase:

- all scores remain very close to zero
- the separation is still tiny in absolute terms
- the current rung does not yet support a strong claim of practical member separation

### 3. Optimization traces still behave cleanly

Representative embedding traces remain monotonic:

- member: `0.3534219 -> 0.3518322`
- nonmember: `0.4414607 -> 0.4397250`

So the problem is not optimization collapse; the issue is weak score magnitude.

## Decision

Current review verdict:

- `weak-positive but still below promotion`

That means:

- do **not** escalate to GPU
- do **not** call this line dead
- do **not** jump straight to a large CPU smoke

## Recommended Next Step

The next honest live task should be one final bounded CPU review rung, not an open-ended ladder:

- still `device = cpu`
- small row increase only
- small step increase only
- explicit no-go if the score gap stays tiny again

## Verdict

- `cpu_microrung_score_verdict = weak-positive but still below promotion`
- `gpu_release = none`
- next step should be a final bounded CPU review rung or a direct no-go decision if that rung is rejected on cost grounds

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
