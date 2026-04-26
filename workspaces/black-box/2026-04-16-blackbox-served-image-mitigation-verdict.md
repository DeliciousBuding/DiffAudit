# 2026-04-16 Black-Box Served-Image Mitigation Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-4 / mitigation-aware black-box evaluation`
- `selected_mitigation`: `served-image-sanitization`
- `probe_target`: `CLiD clip local bridge`
- `device`: `cpu`
- `decision`: `negative but useful no-go`

## Question

If the service applies a mild deployment-side served-image sanitization before returning images to a strict black-box attacker, does the current local `CLiD` black-box signal weaken enough to justify this line as the first realistic black-box mitigation story?

## Executed Evidence

Primary mitigation probe:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-served-image-sanitization-probe-20260415-r1\summary.json`

Frozen local baseline:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-recon-clip-target100-20260415-r1\summary.json`

Supporting preparation script:

- `D:\Code\DiffAudit\Research\scripts\prepare_clid_sanitized_probe.py`

## Probe Configuration

Chosen mitigation:

- deterministic served-image sanitization
- `JPEG quality = 70`
- `resize 512 -> 448 -> 512`

Probe scope:

- `32` member samples
- `32` non-member samples
- same staged `SD1.5` base
- same target-side Recon LoRA
- same local `CLiD clip` finalization logic

## Metrics

Sanitized probe:

- `AUC = 1.0`
- `ASR = 1.0`
- `TPR@1%FPR = 1.0`

Frozen local baseline:

- `AUC = 1.0`
- `ASR = 1.0`
- `TPR@1%FPR = 1.0`

Utility sanity check on the member subset:

- `mean PSNR = 38.286 dB`
- `min PSNR = 30.82 dB`
- `mean MAE = 1.879`

Readout:

- the mitigation is real but mild;
- image structure is not catastrophically destroyed;
- attack metrics did not move at all on this bounded probe.

## Verdict

Current verdict:

- `negative but useful no-go`

Reason:

1. the selected deployment-side mitigation did not reduce the current local `CLiD` attack metrics;
2. the null result is still informative because image utility remained high enough that “quality collapse” is not the main explanation;
3. this means the repo now has one honest black-box mitigation candidate that was actually tried and failed;
4. stronger black-box mitigation claims should pivot to a materially different mechanism instead of more JPEG-only tuning on the same probe.

## Decision

Current decision:

- `keep black-box defense as not-yet-landed`
- `record served-image-sanitization as first bounded no-go`
- `do not escalate this exact JPEG/resize route into a larger comparator`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if black-box defenses are mentioned, the honest wording is that the first realistic deployment-side served-image mitigation was tried on the local `CLiD` bridge and failed to degrade attack metrics, so black-box defense remains an open backlog rather than a landed result.
