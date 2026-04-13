# 2026-04-09 Gray-Box Intake: TMIA-DM

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_paper`: `Temporal Membership Inference Attack Method for Diffusion Models (TMIA-DM)`
- `current_state`: `paper archived and threat-model judged`
- `current_judgment`: `query-based gray-box candidate, not a strict black-box mainline`

## A. Canonical Paper

- Chinese title: `面向扩散模型的时序成员推理攻击方法`
- English title: `Temporal Membership Inference Attack Method for Diffusion Models`
- journal: `计算机研究与发展`
- year: `2026`
- volume / issue / pages: `63(1):243-254`
- doi: `10.7544/issn1000-1239.202440687`
- canonical pdf:
  - `references/materials/gray-box/2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf`

## B. Threat-Model Judgment

Current judgment is:

- this is **not** a strict black-box attack in the same sense as `recon` or `variation`
- the paper explicitly positions the method as a `query-based gray-box` attack

Why:

- the method relies on diffusion-time information rather than only final image outputs
- the abstract and article metadata explicitly emphasize `noise gradient` and `temporal noise`
- that places it much closer to the current repository's gray-box family than to API-only black-box lines

So the correct repository treatment is:

- archive it under `gray-box`
- discuss it in black-box planning only as a boundary correction
- do **not** promote it into the black-box mainline stack

## C. Relation To Existing DiffAudit Lines

### Versus `recon`

- `recon` is still the main strict black-box evidence line
- `recon` uses output-side similarity and repeated generation behavior
- `TMIA-DM` instead depends on temporal / noise-side information

Conclusion:

- `TMIA-DM` does not replace `recon`

### Versus `variation / Towards`

- `variation` is API-only and fits the black-box story better
- `TMIA-DM` is stronger-access than `variation`

Conclusion:

- `TMIA-DM` should not be used to strengthen the black-box execution claim

### Versus `PIA`

- `PIA` in this repo is already framed around `epsilon-trajectory consistency`
- `TMIA-DM` also emphasizes time-dependent / noise-dependent attack information
- both papers sit on the same broad axis: member signal emerges in intermediate diffusion-time behavior, not only final output quality

Conclusion:

- `TMIA-DM` is best treated as a gray-box comparison or follow-up candidate relative to `PIA`, not as a separate black-box execution target

## D. Current Value To The Repo

Immediate value:

1. it strengthens the literature support for gray-box temporal/noise-based attacks
2. it helps explain why `PIA`-style attacks should be discussed as signal-level attacks rather than generic reconstruction heuristics
3. it gives the repo one more formally archived Chinese-language paper tied directly to diffusion MIA

Current limitation:

- no local code or official implementation has been intake-checked yet
- no executable repo path should be claimed from this paper alone

## E. Allowed Claim

The strongest allowed claim right now is:

- `TMIA-DM` is a newly archived `research-ready` gray-box candidate paper
- it supports the temporal/noise-signal direction already visible in the current `PIA` mainline
- it does not change the current black-box mainline or black-box threat-model wording

## F. Next Step

1. Keep `recon` and `variation` unchanged in black-box hierarchy.
2. Add `TMIA-DM` to the gray-box literature and execution checklist as a paper-only candidate.
3. Only consider execution after code availability or a clear reimplementation path is identified.
