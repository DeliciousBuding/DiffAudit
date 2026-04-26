# 2026-04-15 Black-Box Semantic Auxiliary Classifier Progress Note

## Status Panel

- `owner`: `research_leader`
- `task_scope`: `P1-BA-3 / P1-BA-4`
- `family`: `semantic-auxiliary-classifier`
- `probe_run`: `semantic-aux-classifier-probe-20260415-r1`
- `comparator_run`: `semantic-aux-classifier-comparator-20260415-r1`

## Probe Implementation

Implemented a small-sample black-box probe with the following contract:

1. take a candidate query image;
2. derive a prompt from cached metadata text, with local `BLIP` fallback;
3. query the target `SD1.5 + target LoRA` model multiple times using only that prompt;
4. compare the query image against returned images using semantic and image-level features:
   - `mean_cos`
   - `max_cos`
   - `std_cos`
   - `gap_cos`
   - `mean_ssim`
   - `max_ssim`
5. train an offline logistic auxiliary classifier on those returned-image features.

Implementation artifact:

- `D:\Code\DiffAudit\Research\scripts\run_blackbox_semantic_aux_probe.py`

## Probe Result

Completed run:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-probe-20260415-r1\summary.json`

Observed probe metrics:

- `AUC = 0.734375`
- `ASR = 0.8125`
- `TPR@1%FPR = 0.125`

Observed feature means:

- member `mean_cos = 0.544242`
- non-member `mean_cos = 0.426532`
- member `max_cos = 0.603017`
- non-member `max_cos = 0.486778`

Interpretation:

- there is real positive signal in this family;
- the signal is clearly weaker than the frozen `Recon` and local `CLiD` best rungs;
- but it is strong enough to justify one bounded comparator escalation.

## Comparator Result

Completed comparator run:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260415-r1\summary.json`

Comparator scope:

- `16 / 16` member / non-member
- same local CelebA target-family stack
- `3` returned images per query

Observed comparator metrics:

- `AUC = 0.910156`
- `ASR = 0.875`
- `TPR@1%FPR = 0.3125`

Observed feature means:

- member `mean_cos = 0.564904`
- non-member `mean_cos = 0.412804`
- member `max_cos = 0.613479`
- non-member `max_cos = 0.459950`

## Current Read

This line is now a real challenger, not just a weak probe.

Interpretation:

1. the family remains black-box honest because it uses only prompts and final returned images;
2. the comparator materially improves over the initial `8 / 8` probe;
3. on this local target-family comparator it is strong enough to stand as a serious non-`Recon`, non-`CLiD` black-box challenger.

Current positioning against existing black-box lines:

- stronger than the first bounded probe;
- still not a direct replacement for the frozen `Recon` headline story;
- more valuable as method-family diversification and challenger evidence.

## Roadmap Interpretation

- `P1-BA-3`: completed
- `P1-BA-4`: completed with positive signal

## Next Best Move

The next unchecked `P1` item is now:

- `P1-WA-1`

So the black-box new-family line should now pause in `candidate / challenger` state while the roadmap advances to the second white-box line.
