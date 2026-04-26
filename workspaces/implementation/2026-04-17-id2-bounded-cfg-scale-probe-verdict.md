# 2026-04-17 I-D.2 Bounded CFG-Scale Probe Verdict

## Question

On the frozen `I-D.1` local conditional contract, does one bounded `CFG` scale change produce a real, reviewable shift in membership-facing score behavior, or does the signal stay effectively unchanged?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-id1-honest-conditional-target-contract.md`
- `D:\Code\DiffAudit\Research\scripts\run_noise_as_probe_interface_canary.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-interface-canary-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-expansion-rung-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-expansion-repeat-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-threshold-hardening-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g35\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g75\summary.json`

## Frozen Probe Contract

One family only:

- `Stable Diffusion v1.5 base + celeba_partial_target/checkpoint-25000`

One packet only:

- `8 members + 8 evaluation non-members + 8 calibration non-members`
- offsets:
  - `member = 64..71`
  - `calibration non-member = 72..79`
  - `evaluation non-member = 80..87`

One execution surface only:

- `DDIM inversion + DDIM replay`
- `inversion_steps = 10`
- `generation_steps = 10`
- `inversion_guidance_scale = 1.0`
- `distance_metric = MSE`

One bounded variable only:

- `generation_guidance_scale = 3.5` vs `7.5`

Budget:

- one single GPU
- two sequential micro-runs
- no family change
- no scheduler change
- no defense/randomization yet

## Run Anchors

- `g = 3.5`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g35\summary.json`
- `g = 7.5`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g75\summary.json`

## Result

Both GPU micro-runs completed on the same frozen packet.

### `g = 3.5`

- member mean `MSE = 1027.1457`
- eval non-member mean `MSE = 2108.8940`
- separation `= 1081.7483`
- threshold `= 1730.5812`
- `accuracy = 0.875`
- `TPR = 1.0`
- `FPR = 0.25`

### `g = 7.5`

- member mean `MSE = 2642.3119`
- eval non-member mean `MSE = 4222.7693`
- separation `= 1580.4574`
- threshold `= 3323.9035`
- `accuracy = 0.875`
- `TPR = 0.875`
- `FPR = 0.125`

### Cross-scale threshold portability

Reusing the `g = 3.5` threshold on the `g = 7.5` packet:

- `TPR = 0.125`
- `FPR = 0.0`

Reusing the `g = 7.5` threshold on the `g = 3.5` packet:

- `TPR = 1.0`
- `FPR = 1.0`

## Interpretation

This is enough to claim:

1. `CFG` scale is a real conditional-surface variable on the frozen contract;
2. moving from `3.5` to `7.5` materially changes score geometry rather than leaving the packet unchanged;
3. higher `CFG` widened raw member-vs-nonmember separation on this packet;
4. same-run threshold behavior also changed, with lower `FPR` but slightly lower `TPR` at `7.5`.

This is **not** enough to claim:

- a monotonic or globally better `CFG` rule;
- portable thresholding across scales;
- low-FPR maturity;
- attack-family promotion beyond the frozen local contract;
- conditional-diffusion coverage beyond this `SD1.5 + local LoRA` surface.

The most important negative boundary from this packet is:

- `CFG` changes are not just a score-rescaling nuisance;
- but fixed thresholds are also not portable across scales on the current micro-probe.

## Verdict

- `I-D.2 verdict = positive but bounded`
- `cfg_scale_effect = real and reviewable`
- `best same-run operating point on this packet = g=7.5`
- `cross-scale threshold portability = negative`
- `active_gpu_question = none`
- `next live CPU-first lane = I-D.3 bounded CFG-randomization defense idea`
- `next_gpu_candidate = I-D.3 SD1.5 hidden-guidance-jitter micro-probe (pending adaptive-review contract freeze)`
- `CPU sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this creates the first real conditional packet verdict, but it still lives below admitted metrics and does not yet change consumer fields.
