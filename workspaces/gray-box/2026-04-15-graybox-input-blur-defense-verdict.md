# 2026-04-15 Gray-Box Defense Candidate Verdict: Input Gaussian Blur

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-1.2 / GB-1.3`
- `selected_candidate`: `input-gaussian-blur`
- `gpu_status`: `not requested`
- `decision`: `negative`

## Question

After rejecting output-side `epsilon` defenses, does an input-side augmentation-style defense give a more promising second gray-box defended line under the current local `PIA` path?

## Candidate Definition

Selected bounded candidate:

- `input-gaussian-blur`

Operational definition:

- apply Gaussian blur directly to the query image tensor before the current `PIA` attacker consumes it;
- keep the `PIA` attack path unchanged after preprocessing;
- first smoke setting:
  - `input_gaussian_blur_sigma = 1.0`
  - `kernel_size = 3`

Why this was worth trying:

1. it is materially different from `stochastic-dropout`, because it perturbs the input image instead of the model's internal stochastic behavior;
2. it is materially different from `epsilon-precision-throttling` and `epsilon-output-noise`, because it acts before the target `epsilon` prediction exists at all;
3. it aligns with the literature-side intuition that augmentation and mild distortions can affect diffusion-model membership signals.

## Executed Evidence

Bounded smoke:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-input-blur-defense-20260415-cpu-32/summary.json`

Relevant comparison references:

- baseline:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
- current defended mainline smoke:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`
- previously rejected candidate:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-epsilon-output-noise-defense-verdict.md`

## Metrics

Observed bounded smoke metrics:

| rung | AUC | ASR | TPR@1%FPR | wall-clock(s) |
| --- | --- | --- | --- | --- |
| baseline `cpu-32` | `0.782227` | `0.765625` | `0.09375` | `98.56606` |
| stochastic-dropout `cpu-32` | `0.769531` | `0.75` | `0.09375` | `101.033055` |
| epsilon-output-noise `cpu-32` | `0.787109` | `0.765625` | `0.09375` | `154.903108` |
| input-gaussian-blur `cpu-32` | `0.867188` | `0.84375` | `0.1875` | `111.146598` |

Surrogate-quality diagnostics for `input-gaussian-blur`:

- `LPIPS = 0.17364`
- `FID = 0.007427`

## Verdict

Current verdict:

- `negative`

Reason:

1. the candidate substantially strengthened the attack instead of weakening it;
2. `AUC` moved sharply in the wrong direction (`0.782227 -> 0.867188`);
3. `ASR` also moved sharply in the wrong direction (`0.765625 -> 0.84375`);
4. `TPR@1%FPR` worsened (`0.09375 -> 0.1875`);
5. the small input distortion did not preserve the current defended story; it amplified separability.

Interpretation:

- on the current CIFAR-10 DDPM `PIA` path, mild input blur appears to make member/non-member separation easier rather than harder;
- this means the next gray-box defense candidate should not simply be “another small input-space distortion.”

## Decision

Current release decision:

- `no GPU release`

Meaning:

1. do not escalate `input-gaussian-blur` to `GPU128`;
2. do not spend budget on sigma sweeps of the same candidate before a new justification exists;
3. carry this as a rejected bounded candidate in gray-box defense wording.

## Carry-Forward Rule

1. keep `stochastic-dropout(all_steps)` as the only defended gray-box mainline;
2. keep `epsilon-precision-throttling`, `epsilon-output-noise`, and `input-gaussian-blur` as rejected bounded candidates;
3. the next `GB-1` candidate should be more structurally different than small output-side or input-side perturbations.

## Handoff Note

- No immediate `Platform` handoff is needed.
- No immediate `Runtime` handoff is needed beyond preserving the optional blur hook in `pia_adapter.py` for future bounded experiments.
- Material wording should not describe blur-like augmentations as protective under the current local `PIA` path; the recorded bounded evidence points in the opposite direction.
