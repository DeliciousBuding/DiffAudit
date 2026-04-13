# 2026-04-09 Gray-Box Note: Temporal / Noise / Condition Signal Axis

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `scope`: `gray-box literature framing for PIA defended-mainline work`
- `current_state`: `signal-axis literature note added for proposal and claim-boundary use`

## A. Why This Note Exists

Current gray-box execution still centers on `PIA`, but the next defended-mainline question is no longer "is there another attack name we can add".

The more useful framing is:

- which intermediate signal the attack depends on
- whether that signal is stable enough to be exploited
- whether a defense can break that stability without immediately collapsing utility

This note groups the currently archived gray-box papers along that axis so the repo can explain why `PIA + G-1` remains the main experimental line.

## B. Canonical Signal Axis

The current local literature now supports one broad claim:

- diffusion-model gray-box membership inference often succeeds because member and non-member samples behave differently in **intermediate time / noise / condition signals**
- these signals are not all the same, but they are close enough to justify one shared gray-box framing

Current working axis:

1. `PIA`
   - signal: `epsilon-trajectory consistency`
   - current repo role: main executable gray-box line
2. `TMIA-DM`
   - signal: temporal noise / gradient behavior
   - current repo role: literature support for time-dependent signal attacks
3. `SimA`
   - signal: single-query score / noise norm
   - current repo role: extreme low-query gray-box baseline and mechanism simplifier
4. `MoFit`
   - signal: condition mismatch under model-fitted embedding
   - current repo role: caption-free conditional gray-box extension for text-to-image models

## C. What Each Paper Adds

### `PIA`

- local source:
  - `references/materials/gray-box/2024-iclr-pia-proximal-initialization.pdf`
- strongest current repo fit:
  - it is already executable on local CIFAR-10 DDPM assets
  - its score is naturally interpretable as stepwise `epsilon` consistency
- implication for `G-1`:
  - a defense can be justified if it weakens exactly this consistency signal

### `TMIA-DM`

- local source:
  - `references/materials/gray-box/2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf`
- strongest current repo fit:
  - it reinforces that time-dependent noise information is a valid gray-box signal family
- current limit:
  - no local execution intake yet

### `SimA`

- local source:
  - `references/materials/gray-box/2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`
  - `docs/paper-reports/gray-box/2025-arxiv-sima-score-based-membership-inference-diffusion-models-report.md`
- strongest current repo fit:
  - it compresses gray-box MIA down to a single-query score norm view
  - it helps explain why "member signal" can show up in intermediate denoiser behavior even without multi-step replay
- current implication:
  - `PIA` is not the only valid gray-box path, but it is still the best local mainline because the code and assets are already live

### `MoFit`

- local source:
  - `references/materials/gray-box/2026-openreview-mofit-caption-free-membership-inference.pdf`
  - `docs/paper-reports/gray-box/2026-openreview-mofit-caption-free-membership-inference-report.md`
- strongest current repo fit:
  - it extends the same gray-box story into text-conditioned latent diffusion
  - it shows that condition signal fitting can replace true captions in some settings
- current implication:
  - this is a follow-up gray-box expansion line, not the current CIFAR-10 DDPM defended mainline

## D. Repository-Level Conclusion

Current strongest repository-safe statement is:

- `PIA`, `TMIA-DM`, `SimA`, and `MoFit` all support a shared gray-box interpretation in which membership leakage is exposed through intermediate time / noise / condition behavior
- `PIA` remains the main executable line because it already has local assets, admitted summaries, and a defense prototype
- `TMIA-DM`, `SimA`, and `MoFit` should currently be used to sharpen mechanism wording and future expansion choices, not to displace the current mainline

## E. Immediate Use In This Round

This note should be used in exactly two places:

1. proposal / PPT framing:
   - to explain why the current `G-1` work is signal-aware rather than a generic randomization trick
2. claim boundaries:
   - to justify that `PIA + G-1` is the main executable line while other papers remain literature-side support

It should not be used to imply:

- `TMIA-DM` is already executable
- `SimA` has already replaced `PIA`
- `MoFit` is already part of the current admitted CIFAR-10 DDPM table
