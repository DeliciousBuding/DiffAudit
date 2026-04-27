# 2026-04-17 I-D.3 Bounded CFG-Randomization Defense Idea Verdict

## Question

After `I-D.2` showed that fixed `CFG` scale changes materially reshape score geometry and break threshold portability, is there one honest bounded defense idea on the same local conditional contract?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-id2-bounded-cfg-scale-probe-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/noise-as-probe-cfg-microprobe-20260417-gpu-r1-g35/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/noise-as-probe-cfg-microprobe-20260417-gpu-r1-g75/summary.json`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_noise_as_probe_interface_canary.py`

## Candidate Defense Idea

The first bounded `I-D.3` idea is:

- `hidden-guidance jitter`

Meaning:

- keep the same local `SD1.5 + celeba_partial_target/checkpoint-25000` contract;
- keep the same `DDIM 10 / 10` inversion/replay surface;
- keep the same `8 member + 8 eval non-member + 8 calibration non-member` packet;
- but let the effective replay-side `generation_guidance_scale` vary per sample under a hidden defender-side rule rather than staying globally fixed.

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/id3-hidden-guidance-jitter-20260417-r1/summary.json`

This anchor is a CPU-side recombination sweep over two already executed real GPU packets:

- fixed `g = 3.5`
- fixed `g = 7.5`

The mixed packet uses:

- per-sample hidden choice from `{3.5, 7.5}`
- `p = 0.5`
- seeds `0..9`
- calibration-only `15th percentile` thresholding on each mixed packet

## Fixed-Scale References

On the frozen packet:

- fixed `g = 3.5`
  - `accuracy = 0.875`
  - `TPR = 1.0`
  - `FPR = 0.25`
  - separation `MSE = 1081.7483`
- fixed `g = 7.5`
  - `accuracy = 0.875`
  - `TPR = 0.875`
  - `FPR = 0.125`
  - separation `MSE = 1580.4575`

## Mixed Hidden-Jitter Read

Across seeds `0..9`:

- mean `accuracy = 0.675`
- min / max `accuracy = 0.5 / 0.875`
- mean `TPR = 0.6375`
- mean `FPR = 0.2875`
- min / max `FPR = 0.0 / 0.5`
- mean separation `MSE = 1401.7527`

Additional read:

- no mixed seed beats either fixed packet on accuracy
- only `1/10` seed beats fixed `g = 7.5` on `FPR`
- only `2/10` seeds beat fixed `g = 3.5` on `FPR`

## Interpretation

This is enough to claim:

1. hidden `CFG`-scale variation is a plausible conditional-surface defense idea;
2. the idea can materially destabilize attack-side calibration on the frozen packet;
3. mixed hidden-guidance packets usually reduce attack accuracy relative to both fixed-scale references;
4. the defense idea is therefore worth keeping as a bounded follow-up rather than discarding immediately.

This is **not** enough to claim:

- a released defense;
- low-FPR reliability;
- adaptive-attacker robustness;
- runner-level deployment truth;
- a project-level admitted mitigation.

The most important current boundary is:

- the idea looks promising as an attack-disrupting randomization;
- but the current evidence is still seed-sensitive and below release-grade.

## Verdict

- `I-D.3 verdict = positive but bounded`
- `defense_idea = hidden-guidance jitter on the frozen local conditional contract`
- `attack_disruption_read = positive on average, unstable across seeds`
- `low_fpr_read = not established`
- `adaptive_read = not established`
- `active_gpu_question = none`
- `next live CPU-first lane = X-14 cross-box / system-consumable sync after first bounded I-D packet pair`
- `next_gpu_candidate = I-D.3 actual runner-level hidden-guidance-jitter rerun on the frozen 8/8/8 packet (pending adaptive-review contract freeze)`
- `CPU sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes research-side direction and GPU candidate quality, but it still does not change admitted fields or competition headline claims.
