# 2026-04-16 Black-Box Recon Timestep Probe Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-1.3 / timestep-selective reconstruction probe`
- `method_family`: `recon-coordinate-selective`
- `device`: `cpu`
- `decision`: `negative but useful`

## Question

Given the current local `recon DDIM public-100 step30` score artifacts, does a bounded `shadow-select -> target-eval` coordinate-selective probe find a stronger reconstruction signal than the currently used baseline coordinate?

## Executed Evidence

Primary run artifact:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/recon-timestep-probe-20260416-r1/summary.json`

Source artifacts:

- `<DIFFAUDIT_ROOT>/Research/experiments/recon-runtime-mainline-ddim-public-100-step30/score-artifacts`

## Observed Structure

- each current `recon` artifact sample contains a vector, not just one scalar
- observed vector length on the current artifact stack: `768`
- current mainline extraction still uses `feature[0]` as the effective score

## Metrics

Current bounded `shadow-select -> target-eval` probe:

- baseline coordinate:
  - `dim0`
  - target `AUC = 0.6897`
  - target `TPR@1%FPR = 0.13`
- best shadow-selected coordinate:
  - `dim732`
  - shadow `AUC = 0.6753`
  - target `AUC = 0.6472`
  - target `TPR@1%FPR = 0.14`
- target `AUC gain vs dim0 = -0.0425`

Useful upper-bound observation:

- some top-shadow coordinates have better target `AUC` than `dim0`
- but the honest `shadow-select` choice did not generalize to the target pair

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the current artifact vectors do contain more than one coordinate, so the probe itself was worth doing;
2. however, the honest `shadow-select -> target-eval` procedure does not beat the current baseline coordinate;
3. therefore the current repo should not claim that a timestep/coordinate-selective reconstruction readout has already improved the `recon` line;
4. the result is still useful because it closes the simplest selective-readout hypothesis without a new generation run.

## Decision

Current decision:

- close `BB-1.3` as `negative but useful`
- keep current `recon` headline wording unchanged
- do not replace the current baseline score extraction with a shadow-selected coordinate on the present artifact stack
- only reopen if:
  - artifact semantics are upgraded so vector dimensions can be mapped to real timestep meaning, or
  - a stronger selection rule than naive single-coordinate shadow maximization is proposed

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed; the current `recon` wording remains correct.
