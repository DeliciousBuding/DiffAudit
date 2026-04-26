# 2026-04-17 I-D.4 Actual Runner-Level Hidden-Guidance-Jitter Verdict

## Question

If the first `I-D.3` hidden-guidance jitter idea is executed as a real runner-level packet, does it still look like a plausible bounded defense idea, or does the honest deterministic contract collapse once the mixed packet is run for real?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\scripts\run_noise_as_probe_interface_canary.py`
- `D:\Code\DiffAudit\Research\tests\test_noise_as_probe_interface_canary.py`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-id2-bounded-cfg-scale-probe-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-id3-bounded-cfg-randomization-defense-idea-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\runs\id3-hidden-guidance-jitter-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g35\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-cfg-microprobe-20260417-gpu-r1-g75\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-hidden-guidance-jitter-20260417-gpu-r2-s0\summary.json`

## Frozen Contract

Same family only:

- `Stable Diffusion v1.5 base + celeba_partial_target/checkpoint-25000`

Same packet only:

- `8 members + 8 evaluation non-members + 8 calibration non-members`
- offsets:
  - `member = 64..71`
  - `calibration non-member = 72..79`
  - `evaluation non-member = 80..87`

Same replay surface only:

- `DDIM inversion + DDIM replay`
- `inversion_steps = 10`
- `generation_steps = 10`
- `inversion_guidance_scale = 1.0`
- `distance_metric = MSE`

One deterministic hidden-jitter rule only:

- `generation_guidance_mode = hidden-jitter`
- candidate set = `{3.5, 7.5}`
- `generation_guidance_seed = 0`
- selection rule:
  - deterministic hidden per-sample choice keyed by `seed + split + file_name`
- calibration threshold rule:
  - `percentile(15)` on the jittered calibration packet only

Implementation hardening added before execution:

- `subset_limit = 0` now truly yields zero rows
- evaluation non-member and calibration rows must be disjoint by `file_name`
- each record now writes its effective generation guidance scale
- summary now exposes mode, candidate scales, seed, selection rule, and split-level scale counts

## Run Anchor

- Actual runner-level packet:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\noise-as-probe-hidden-guidance-jitter-20260417-gpu-r2-s0\summary.json`

## Result

The real runner-level hidden-jitter packet completed successfully on one GPU.

Effective guidance usage:

- member split:
  - `7.5 x 7`
  - `3.5 x 1`
- evaluation non-member split:
  - `3.5 x 6`
  - `7.5 x 2`
- calibration non-member split:
  - `3.5 x 4`
  - `7.5 x 4`

Observed packet behavior:

- member mean `MSE = 2537.7371`
- eval non-member mean `MSE = 2584.9976`
- separation `= 47.2605`
- threshold `= 2535.1431`
- `accuracy = 0.5`
- `TPR = 0.625`
- `FPR = 0.625`

Cross-check against the earlier CPU-side idea:

- the new deterministic runner contract does **not** reproduce the earlier optimistic `seed = 0` CPU recombination readout:
  - previous CPU recombination `seed 0`:
    - `accuracy = 0.875`
    - `TPR = 0.75`
    - `FPR = 0.0`
- instead, the real runner packet collapses to:
  - `accuracy = 0.5`
  - `TPR = 0.625`
  - `FPR = 0.625`

Execution-audit cross-check:

- under the new deterministic selection rule, the actual runner packet matches the corresponding per-sample `g=3.5 / g=7.5` fixed-packet records exactly
- so the failure is not an implementation mismatch between mixed execution and summary writing
- the failure is the honest packet-level behavior of this deterministic hidden-jitter contract

## Interpretation

This packet is enough to claim:

1. runner-level hidden guidance execution is now real, auditable, and reproducible on the frozen local contract;
2. the earlier CPU-side mixed-seed idea was too weak as a promotion surface unless the exact selection rule is frozen first;
3. once the contract is made honest and deterministic, this seed does not preserve useful separation.

This packet is **not** enough to claim:

- a working release-grade defense;
- low-FPR maturity;
- adaptive robustness;
- a reusable `I-D` promotion path;
- any broader conditional-diffusion defense claim.

The most important truth from this run is:

- the first actual runner-level deterministic hidden-guidance-jitter packet is not merely "bounded";
- on the honest frozen packet it is `negative but useful`, because the attack signal collapses together with calibration specificity.

## Verdict

- `I-D.4 verdict = negative but useful`
- `runner_level_hidden_guidance_jitter = real but not defensible on the frozen packet`
- `low_fpr_boundary = failed`
- `adaptive_release = no-go`
- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `next live CPU-first lane = I-A truth-hardening refresh after negative actual I-D rerun`
- `CPU sidecar = higher-layer PIA provenance / boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/reproduction-status.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes the next GPU-candidate truth and closes the currently prepared `I-D` runner-level follow-up, but it still does not change admitted tables, exported schema, or competition-facing claims.
