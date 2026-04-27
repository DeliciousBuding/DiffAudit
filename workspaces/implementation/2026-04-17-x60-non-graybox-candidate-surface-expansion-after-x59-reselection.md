# 2026-04-17 X-60 Non-Graybox Candidate-Surface Expansion After X-59 Reselection

## Status Panel

- `owner`: `ResearcherAgent`
- `task_type`: `cpu-first candidate-surface expansion`
- `device`: `cpu`
- `verdict`: `positive`

## Question

After `X-59` already froze the current control-plane truth to "no blocked/hold branch honestly reopened" and "no fresh `I-B / I-C` successor lane exists", which fresh non-graybox candidate surface should now be restored as the next honest live lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/paper-matrix-2024-2026.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/experiment-entrypoints.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-17-blackbox-next-family-candidate-generation-refresh-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-16-post-second-signal-blackbox-next-question-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-09-blackbox-method-boundary.md`
- `<DIFFAUDIT_ROOT>/Download/manifests/research-download-manifest.json`

## Review

### 1. Existing non-graybox frozen branches still do not reopen honestly

- `I-B` already froze to `actual bounded falsifier`; same-family rescue remains forbidden.
- fresh `I-C` already froze to `negative but useful`; same-board salvage remains forbidden.
- `XB-CH-2` is still `needs-assets` on the frozen four-part reopen contract.
- white-box distinct-family import is still closed-negative.
- gray-box already yielded the next `CPU-first` slot and still has no immediate new family execution lane.

So `X-60` is not allowed to fake progress by mechanically returning to an already frozen surface.

### 2. Black-box is the only remaining honest expansion surface, but only at scouting strength

Current black-box truth remains:

- `Recon = headline`
- `semantic-auxiliary-classifier = leading challenger`
- `CLiD = evaluator-near local clip-only corroboration`
- `variation = contract-ready blocked`

But the reviewed local evidence also shows:

- the old black-box refresh review only closed **immediate promotion**, not all future scouting;
- `semantic-aux` scoring/fusion is still same-family continuation, not a new family;
- `variation` still needs real `query_image_root + endpoint + budget`;
- `CLiD` is still boundary-quality work, not a new family;
- `dataset-audit-track` exists in entrypoints, but direct promotion would currently collapse back into gray-box-owned `CDI`;
- the current download manifest exposes supplementary assets for `Recon / CLiD`, not a hidden ready-to-run new black-box family.

Therefore black-box does not contain a ready rerun or GPU-worthy question, but it **does** contain the strongest remaining CPU-only expansion room: a paper-backed next-family scouting pass.

### 3. The restored surface should be scoping-first, not promotion-first

The honest restored candidate surface is:

- `black-box paper-backed next-family scouting`

Its job is:

- inspect whether a genuinely new black-box family can be restored without collapsing into
  - same-family `semantic-aux` continuation,
  - `CLiD` boundary tightening,
  - `variation` asset unblock,
  - or gray-box-owned `CDI`;
- keep `gpu_release = none` unless a new bounded family really appears.

## Verdict

- `x60_candidate_surface_expansion_verdict = positive`
- the restored non-graybox candidate surface is `black-box paper-backed next-family scouting`
- this is a `CPU-first` scoping lane, not a GPU release and not a same-family reopen
- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `current_cpu_sidecar = I-A higher-layer boundary maintenance`

## Next Lane

- `X-61 black-box paper-backed next-family scoping review after X-60 expansion`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/black-box/plan.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- `Platform / Runtime`: no immediate handoff required
