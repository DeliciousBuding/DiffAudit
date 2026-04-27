# 2026-04-15 Black-Box Second-Signal Semantic Auxiliary Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-1`
- `family`: `semantic-auxiliary-classifier`
- `canonical_run`: `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260415-r1/summary.json`
- `gpu_status`: `completed`

## Question

Does the returned-image semantic auxiliary classifier deserve promotion from idea to a real black-box challenger beyond `Recon + CLiD`?

## Evidence Snapshot

Bounded local comparator on the current CelebA target-family stack:

- `AUC = 0.910156`
- `ASR = 0.875`
- `TPR@1%FPR = 0.3125`
- `member mean_cos = 0.564904`
- `nonmember mean_cos = 0.412804`

Scaled follow-up comparator on `2026-04-16` (`32 / 32`, same protocol):

- run anchor:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260416-r2/summary.json`
- `AUC = 0.90918`
- `ASR = 0.84375`
- `TPR@1%FPR = 0.25`
- `member mean_cos = 0.583986`
- `nonmember mean_cos = 0.430354`

The attack remains black-box honest in the current local protocol:

1. prompt comes from cached metadata or local `BLIP`;
2. attack only queries the target generation service;
3. scoring only uses final returned images plus offline features.

## Verdict

Current verdict:

- `positive challenger`

Interpretation:

1. this is no longer just a weak probe;
2. it is materially distinct from `Recon` and `CLiD`;
3. it is strong enough to keep as the current non-`Recon`, non-`CLiD` black-box challenger;
4. it still does not replace the frozen `Recon` headline story.

The `32 / 32` follow-up matters because it shows the challenger does not collapse immediately when the bounded comparator is doubled.

## Mainline Positioning

- `Recon` remains the black-box main evidence line.
- `CLiD` remains the black-box corroboration / boundary-quality line.
- `semantic-auxiliary-classifier` is now the leading new-family challenger.

## Reopen / Escalation Rule

Escalate this challenger only with a bounded new hypothesis such as:

1. a larger but still honest comparator,
2. stronger feature ablation,
3. mitigation-aware evaluation on the same returned-image protocol.

Do not escalate it by relabeling it as a `Recon` replacement.

## Handoff Guidance

- `Platform / Runtime`: no schema change required; if a black-box challenger field exists, prefer `blackbox_challenger = semantic-auxiliary-classifier`.
- `competition materials`: optional mention only as “new-family challenger landed locally”; do not let it displace `Recon` as the headline black-box evidence.
