# 2026-04-16 Subagent Paper-Scout Verdict

## Task

- `INF-3.1` test paper-scout subagent workflow

## Question

- Can a read-only paper-scout subagent materially improve next-branch candidate selection for the gray-box mainline, or does it mostly restate what the main agent already knows?

## Setup

- subagent type: read-only paper scout
- model: `gpt-5.4`
- reasoning: `high`
- write scope: none
- GPU use: none

## What The Subagent Was Asked

- Among the currently not-yet-promoted gray-box follow-up candidates, which single candidate now has the shortest credible bounded implementation path on current local assets, and why?

## Result

Subagent recommendation:

- `SimA`

Suggested bounded first experiment:

- `SimA cpu-32 timestep rescan on current CIFAR-10 DDPM asset line`

Why this was useful:

1. it explicitly argued against a premature jump to `MoFit / noise-as-a-probe / SIDe`;
2. it grounded the recommendation in current repo execution reality, not just paper novelty;
3. it reframed `SimA` from “already weak, ignore it” into a narrower question:
   - was the family weak,
   - or was the current timestep choice weak?

## Verdict

- `positive`

The paper-scout workflow created real leverage here. It did not discover a brand-new paper family, but it did sharpen candidate generation by identifying the shortest credible re-entry path that stays CPU-bounded and hypothesis-specific.

## Standardization Note

What should become standard:

1. use a read-only paper-scout when several literature candidates exist but local execution distance differs a lot;
2. require the subagent to recommend exactly one candidate and one bounded first experiment;
3. require an explicit GPU judgment so “interesting paper” does not silently become “open GPU question.”

What should stay optional:

1. using paper-scout when the next branch is already obvious from local verdicts;
2. asking the paper-scout to cover too many candidate families at once;
3. any write-capable literature delegation.

## Next Recommendation

- accept the recommendation only as a candidate-generation verdict
- if reopened, the next gray-box family question should be:
  - `SimA cpu-32 timestep rescan`

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
