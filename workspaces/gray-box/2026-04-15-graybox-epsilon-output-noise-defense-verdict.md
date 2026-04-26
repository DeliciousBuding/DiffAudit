# 2026-04-15 Gray-Box Defense Candidate Verdict: Epsilon Output Noise

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-1.1 / GB-1.2`
- `selected_candidate`: `epsilon-output-noise`
- `gpu_status`: `not requested`
- `decision`: `negative but useful`

## Question

After `stochastic-dropout(all_steps)` became the current defended gray-box mainline, is there a second low-cost gray-box defense candidate that is materially different from both dropout and deterministic precision throttling, and worth releasing onto GPU?

## Candidate Definition

Selected bounded candidate:

- `epsilon-output-noise`

Operational definition:

- keep the current `PIA` attack path unchanged;
- after each `UNet` forward pass in `RuntimeEpsGetter`, add Gaussian perturbation directly to the predicted `epsilon` tensor;
- first smoke level:
  - `epsilon_output_noise_std = 0.1`

Why this was worth trying:

1. it is materially different from `stochastic-dropout`, because it perturbs the exposed gray-box signal rather than the internal forward-pass mask pattern;
2. it is materially different from `epsilon-precision-throttling`, because it preserves continuous outputs instead of deterministically compressing them;
3. it still targets the same `epsilon-trajectory consistency` signal that the current `PIA` path exploits.

## Executed Evidence

New bounded smoke:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-eps-noise-defense-20260415-cpu-32\summary.json`

Relevant comparison references:

- baseline:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260408-cpu-32\summary.json`
- current defended mainline smoke:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32\summary.json`
- already rejected deterministic output restriction candidate:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-defense-precision-throttling-note.md`

## Metrics

Observed bounded smoke metrics:

| rung | AUC | ASR | TPR@1%FPR | wall-clock(s) |
| --- | --- | --- | --- | --- |
| baseline `cpu-32` | `0.782227` | `0.765625` | `0.09375` | `98.56606` |
| stochastic-dropout `cpu-32` | `0.769531` | `0.75` | `0.09375` | `101.033055` |
| epsilon-output-noise `cpu-32` | `0.787109` | `0.765625` | `0.09375` | `154.903108` |

Surrogate-quality diagnostics for `epsilon-output-noise`:

- `LPIPS = 0.046409`
- `FID = 0.000592`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the candidate did **not** weaken the attack at the bounded smoke scale;
2. `AUC` moved in the wrong direction (`0.782227 -> 0.787109`);
3. `ASR` did not improve;
4. `TPR@1%FPR` did not improve;
5. wall-clock cost increased substantially.

Interpretation:

- adding small Gaussian perturbation to the exposed `epsilon` tensor is not enough, by itself, to break the current `PIA` ranking signal under the local CIFAR-10 DDPM path;
- at this smoke scale it behaves more like nuisance variation than a useful privacy-preserving disruption;
- this makes it a poor immediate successor to `stochastic-dropout`.

## Decision

Current release decision:

- `no GPU release`

Meaning:

1. do not escalate this candidate to `GPU128`;
2. do not spend the current gray-box second-defense budget on larger repeats of the same knob;
3. keep it recorded as a rejected bounded candidate, not as an active defended line.

## Carry-Forward Rule

1. keep `stochastic-dropout(all_steps)` as the only defended gray-box mainline;
2. keep both `epsilon-precision-throttling` and `epsilon-output-noise` as rejected bounded candidates;
3. the next `GB-1` candidate should be more materially different from direct output perturbation, not just another nearby `epsilon` corruption knob.

## Handoff Note

- No immediate `Platform` handoff is needed.
- No immediate `Runtime` handoff is needed beyond preserving the new optional hook in `pia_adapter.py` for future bounded experiments.
- Material wording should not mention `epsilon-output-noise` as an active defense candidate; it is a recorded negative result.
