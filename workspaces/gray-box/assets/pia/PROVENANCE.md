# PIA Provenance Record

更新时间：`2026-04-17`

这份记录用于固定当前 gray-box `PIA` 主线的 canonical 资产来源、哈希和状态口径。

## Canonical Assets

### Checkpoint

- path: `workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm/checkpoint.pt`
- size_bytes: `574274030`
- last_modified: `2026-04-06 16:49:00 +08:00`
- source bundle: `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
- extracted from: `DDPM/ckpt_cifar10.pt`
- sha256: `55C3F4545BC345522C402EF9C84627BB87BA53049C9541DDF46A6FEB5B0B283E`
- ownership: `gray-box / PIA`
- current status: `workspace-verified`

### Dataset Archive

- path: `workspaces/gray-box/assets/pia/sources/cifar-10-python.tar.gz`
- size_bytes: `170498071`
- last_modified: `2026-04-07 00:40:47 +08:00`
- sha256: `6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE`
- extracted root: `workspaces/gray-box/assets/pia/datasets/cifar10`
- ownership: `gray-box / PIA`
- current status: `workspace-verified`

### Source Bundle

- path: `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
- size_bytes: `1387062124`
- last_modified: `2026-04-07 00:55:01 +08:00`
- sha256: `B69310B92AF98B5AFF2D046747DA6650A915C0FC6A79291C999192D43CEF98E5`
- ownership: `gray-box / PIA`
- note: retained as the original imported source bundle for the current checkpoint

### External Split Dependency

- path: `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`
- sha256: `ACA922ECEE25EF00DC6B6377EBAF7875DFCC77C2CDFE27C873B26A65134AA0C0`
- ownership: `upstream external dependency`
- current status: `retained external dependency`

## Runtime Evidence Anchor

Current admitted runtime evidence is anchored to:

- baseline summary:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512/summary.json`
- defense summary:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512/summary.json`
- same-scale repeat:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-rerun1/summary.json`
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-rerun1/summary.json`

Current interpretation:

- the asset line is now `workspace-verified`
- the runtime evidence is now both multi-scale and same-scale repeat-supported
- this is enough for `provisional G-1`
- this is still not enough for `paper-aligned`

## Next-Run Gate

Machine-readable next-run evidence now exists at:

- relaxed gate:
  - `workspaces/gray-box/assets/pia/next-run-20260409/manifest.json`
  - `workspaces/gray-box/assets/pia/next-run-20260409/provenance.json`
- strict gate:
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/manifest.json`
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/provenance.json`

Current gate interpretation:

- the relaxed gate proved the current config and member split inputs could be hashed and recorded
- the strict gate passed after `external/PIA` returned to a clean git state
- this is enough to promote the local asset line to `workspace-verified`
- this still does not justify promotion to `paper-aligned`

## Current Gate Judgment

- `runtime readiness`: `single-machine ready`
- `asset grade`: `single-machine-real-asset`
- `provenance status`: `workspace-verified`
- `repeat confirmation`: `GPU512 repeat-confirmed`
- `paper alignment`: `not confirmed`

## Frozen Paper-Alignment Boundary

Current blocker tuple:

1. `release/source identity unresolved`
2. `CIFAR10 random-four-split / four-model tau-transfer protocol mismatch vs current single fixed ratio0.5 split`

Current hygiene caveat:

- `2026-04-09` strict next-run gate passed under a historical clean snapshot
- this does **not** prove that the current repo state is still clean
- this does **not** prove immutable upstream release identity

## Allowed Claims

- the current checkpoint, dataset root and member split can be loaded by the Research-side `PIA` integration
- the current line is acceptable as a local research mainline input
- the current line is not yet acceptable as `paper-aligned` or `benchmark-ready`

## Required Carry-Forward Fields

Any new `PIA` summary or system-level record must carry:

- `track = gray-box`
- `method = pia`
- `contract_stage = target`
- `asset_grade = single-machine-real-asset`
- `provenance_status = workspace-verified`
- `evidence_level` matching the actual command that produced the result

## Consumer Rule

Downstream intake or display consumers may safely read this line as:

- `system-intake-ready`
- `workspace-verified`

They must **not** collapse it into:

- `paper-aligned`
- `benchmark-ready`

## Reopen Trigger

This provenance blocker should reopen only if at least one of the following changes materially:

1. a stable upstream release identity / immutable source-bundle proof appears
2. protocol-aligned random-four-split / four-model assets become available
3. a new strict review is re-run under a newly cleaned and explicitly re-checked upstream state
