# 2026-04-16 MoFit CPU Micro-Rung Design

## Question

After the first fresh `MoFit` CPU canary landed but remained score-inconclusive, what is the smallest next CPU rung that can add signal without reopening wasteful budget behavior?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-canary-score-shape-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4/records.jsonl`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`

## Current Constraint

The current canary already proved:

- real local execution is feasible
- optimization traces move in the expected direction

But it did **not** prove:

- stable member/nonmember separation
- useful score sign or magnitude

So the next rung must add signal while staying obviously below smoke scale.

## Selected Micro-Rung

Recommended next bounded CPU rung:

- `device = cpu`
- `launch_profile = bounded-cpu-first` remains the base contract
- explicit rung override:
  - `member_limit = 2`
  - `nonmember_limit = 2`
  - `surrogate_steps = 2`
  - `embedding_steps = 4`

Why this rung:

1. it doubles rows but stays tiny enough for a local CPU pass;
2. it doubles optimization depth relative to the canary, which is the smallest honest way to test whether the near-zero gap was only a budget artifact;
3. it still stays far below a smoke-like multi-row or longer-horizon run.

## Stop Condition

This rung should stop after one pass.

No immediate follow-up rung if any of the following happen:

- member/nonmember score means remain near-zero and overlapping
- score sign remains unstable across the four rows
- wall-clock cost is already disproportionate to the information gained

## Promotion Rule

Only consider a larger CPU rung if the micro-rung shows at least one of:

- stable score direction between member and nonmember
- visibly widened separation relative to the current canary
- trace behavior that makes the score trend more interpretable

## Verdict

- `cpu_microrung_design_verdict = positive`
- the next honest live task is now implementation or execution of exactly this bounded CPU micro-rung
- `gpu_release = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
