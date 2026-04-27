# 2026-04-17 Finding NeMo Bounded Localization Observable Selection

## Question

Given the current admitted white-box assets and the already-frozen `I-B.1` protocol bridge, which single localization observable is the first honest one to trust, without smuggling in benchmark claims or paper-faithful `Finding NeMo` wording?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/signal-access-matrix.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-feature-trajectory-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/finding-nemo-observability-canary-20260410-round24/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/finding-nemo-observability-canary-20260410-round24/records.jsonl`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`

## Candidate Review

### 1. `activations`

Strengths:

- directly aligned with localization intent
- already exported on admitted `GSA` assets through a CPU-only, read-only path
- produces machine-auditable artifacts:
  - raw tensor artifact
  - `records.jsonl`
  - `summary.json`
- already frozen to one selector / one timestep / one sample-pair contract

Weaknesses:

- current export is only a canary, not a localization result
- current per-record `summary_stat` is too compressed to act as a localization score by itself

### 2. `grad_norm`

Strengths:

- still a legitimate white-box supporting signal
- already acknowledged as an optional comparator in the contract layer

Weaknesses:

- current migrated observability route does not yet emit a comparable sample-level `grad_norm` artifact packet on admitted assets
- current repository truth around `I-B` is more mature for `activations` than for `grad_norm`

## Evidence Reading

The decisive evidence is not “member beats non-member on a scalar”.

The decisive evidence is:

1. the canary export is `ready`
2. the bridge resolves:
   - admitted asset root
   - admitted checkpoint root
   - fixed member/control sample binding
   - fixed selector `mid_block.attentions.0.to_v`
   - fixed timestep `999`
3. the export emits two raw activation tensors with shape `[1, 1, 512]`
4. the export stores replayable tensor artifacts through repo-local relative artifact paths

The current `summary_stat` values for the first canary/control pair are also a useful warning:

- member mean/std:
  - `0.021409 / 0.616087`
- control mean/std:
  - `0.020849 / 0.616158`

These are extremely close.

So the honest lesson is:

- the first trustworthy localization observable is **not** the scalar `summary_stat`
- it is the raw sample-level activation tensor under the frozen selector/timestep contract

## Selected Observable

The first bounded localization observable to trust is now frozen as:

- `sample-level activation tensor`

More precise contract:

1. one admitted checkpoint root
2. one fixed selector:
   - `mid_block.attentions.0.to_v`
3. one fixed timestep:
   - `999`
4. one fixed sample-pair binding:
   - `target-member` vs `target-nonmember`
5. one raw tensor artifact per sample

And three explicit anti-overclaim rules:

- `summary_stat` is sanity metadata, not a localization score
- execution-context path strings are metadata, not the localization observable itself
- `grad_norm` remains a supporting comparator candidate, not the first trusted observable

## Verdict

- `finding_nemo_bounded_localization_observable_selection_verdict = positive but bounded`

More precise reading:

1. `I-B.2` is now satisfied:
   - the first honest localization observable is the raw sample-level activation tensor
2. this does **not** mean localization is solved:
   - no neuron ranking
   - no mechanism claim
   - no intervention claim
3. this also means the repository should **not** treat scalar export summaries as the new white-box signal family
4. `grad_norm` stays below the first-observable slot until it has a comparably fixed artifact contract

## Next Step

- `next_live_cpu_first_lane = I-B.3 bounded local intervention proposal`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no immediate sync; this is still below release-facing wording
- `Platform/Runtime`: no direct handoff; consumers must not read `summary_stat` as a released localization score
