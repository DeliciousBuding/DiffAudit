# 2026-04-15 Gray-Box Defense Candidate Note: Epsilon Precision Throttling

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-15 +08:00`
- `task_scope`: `P0-GD-1 / P0-GD-2`
- `selected_candidate`: `epsilon-precision-throttling`
- `gpu_status`: `not started`

## Chosen Defense Mechanism

Selected gray-box defense mechanism:

- `epsilon-precision-throttling`

Operational definition:

- apply deterministic precision reduction to the model-predicted `epsilon` tensor before the attack consumes it
- first implementation target:
  - uniform quantization / rounding of `epsilon`
  - exposed as a runtime defense toggle in the `PIA` adapter

## Why This Is Materially Different From Stochastic Dropout

Current defended mainline:

- `stochastic-dropout`

Current effect mechanism:

- injects stochastic inference-time instability by switching the model to `train()` at chosen steps

Precision throttling is materially different because it:

- keeps the model deterministic at a fixed input;
- degrades information content rather than sampling stability;
- targets the resolution of the visible gray-box signal instead of the internal forward-pass randomness.

So this is not “another random-noise defense knob”. It is an `observable precision restriction` defense.

## Hypothesis

Current `PIA` signal depends on:

- `epsilon-trajectory consistency`

That means the attack benefits from fine-grained agreement between:

1. `eps` estimated from the original image;
2. `eps_back` estimated after re-noising across attack steps.

Hypothesis:

- if the defender only exposes a precision-throttled `epsilon` signal, the small member/non-member consistency differences should compress faster than the coarse denoising structure needed to preserve rough surrogate utility.

Expected outcome:

- attack ranking should change, not only threshold calibration;
- `AUC` and `ASR` should both weaken relative to the baseline;
- utility degradation should be qualitatively different from dropout because the outputs stay deterministic.

## Why This Candidate Is Worth A Smoke Run

This candidate is attractive for `P0-GD-3` because:

- it directly attacks the identified signal axis instead of being a generic regularizer;
- the implementation hook is small and local to `eps_getter`;
- it is plausibly realistic for an audit platform, because exposed gray-box outputs could be quantized or precision-limited at serving time;
- it gives a qualitatively different story from `stochastic-dropout`.

## Minimal Smoke Design

Smoke target:

- current CIFAR-10 `PIA` runtime path
- CPU or bounded GPU smoke with the canonical local assets

First concrete parameter proposal:

- quantize `epsilon` to a small number of bins after each forward pass
- start with one moderately aggressive level rather than a sweep

Recommended first value:

- `epsilon_quant_bins = 32`

Reason:

- strong enough to distort fine-grained score differences;
- not so extreme that the smoke becomes an obviously broken-path artifact.

## Promotion Criterion

Promote this candidate to a mainline comparator only if:

1. it lowers attack metrics versus baseline in a repeatable way;
2. the effect is not merely identical to `stochastic-dropout` at the ranking level;
3. the surrogate-quality diagnostics stay interpretable rather than collapsing into obvious garbage.

Otherwise label it:

- `failed`
- `no-go`
- or `negative but useful`

## Next Action

1. add a small runtime defense hook in `pia_adapter.py`;
2. run one bounded smoke;
3. compare against baseline and current dropout defended reading.

## Executed Smoke Result

Smoke run executed:

- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-eps-precision-defense-20260415-cpu-32`

Configuration:

- dataset: CIFAR-10
- method: `PIA`
- device: `cpu`
- sample count per split: `32`
- defense: `epsilon_precision_bins = 32`

Observed result:

| rung | AUC | ASR | TPR@1%FPR | wall-clock(s) |
|---|---|---|---|---|
| baseline `cpu-32` | `0.782227` | `0.765625` | `0.09375` | `98.56606` |
| stochastic-dropout `cpu-32` | `0.769531` | `0.75` | `0.09375` | `101.033055` |
| epsilon-precision-throttling `cpu-32` | `0.808594` | `0.78125` | `0.09375` | `191.069539` |

Surrogate-quality diagnostics for precision throttling:

- `LPIPS`: `0.015008`
- `FID`: `0.000821`

## Verdict

Current verdict:

- `negative but useful`

Reason:

- this candidate did **not** weaken the attack on the bounded smoke;
- it increased both `AUC` and `ASR` relative to the paired `cpu-32` baseline;
- it is therefore not a promising immediate successor to `stochastic-dropout`.

Interpretation:

- coarse deterministic precision compression does not automatically destroy the `epsilon-trajectory consistency` signal;
- at this smoke scale it may even remove nuisance variance faster than it removes member/non-member separation;
- this makes it a poor next mainline defense candidate under the current local `PIA` path.

## Roadmap Consequence

- `P0-GD-3`: satisfied by the bounded smoke run
- `P0-GD-4`: no promotion to mainline comparator; smoke was not promising
- `P0-GD-5`: negative verdict should be carried into gray-box defense comparison wording as a rejected `G-2` candidate
