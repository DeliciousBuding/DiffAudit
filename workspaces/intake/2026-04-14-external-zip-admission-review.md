# 2026-04-14 External Zip Admission Review

## Scope

This memo records the Research-side quarantine judgment for two external zip inputs reviewed during the submission freeze window:

- `Stable_Diffusion-周洛玄.zip`
- `ckpt-step0-徐驰.zip`

These assets were reviewed under the current hard boundary:

- admitted mainline remains `recon / PIA + stochastic-dropout(all_steps) / GSA + W-1`
- no new GPU question
- no narrative drift caused by private snapshots, large checkpoints, or provenance-unclear bundles

## Verdict Summary

### 1. `Stable_Diffusion-周洛玄.zip`

- `current_verdict = reference-only`
- `recommended_shape = docs-only intake memo; do not import raw bundle`
- `reason`:
  - mixed private experiment bundle rather than clean research code drop
  - contains large copied datasets, backups, run snapshots, and helper scripts
  - no release-level provenance or reproducible contract
  - overlaps with gray-box themes but does not strengthen the current admitted `PIA` line
  - would introduce `SecMI`/Stable-Diffusion-side noise into the 4C mainline

### 2. `ckpt-step0-徐驰.zip`

- `current_verdict = reject`
- `recommended_shape = keep out of repo-tracked mainline assets`
- `reason`:
  - single checkpoint-only payload with no source bundle, config, flagfile, or manifest
  - inspection showed `step = 0`, so it is not a mature runtime artifact for the current mainline
  - does not unblock `SecMI`
  - does not improve `PIA paper-aligned` provenance
  - would only add a large opaque file to the repo without decision-grade value

## Hard Rule Going Forward

For the remaining submission window, any new external zip / checkpoint / private experiment directory must be judged into exactly one of:

- `go docs-only`
- `reference-only`
- `archive-only`
- `reject`

It must not be copied into:

- `Research/src/`
- canonical `Research/workspaces/**/assets/**`
- admitted `Research/experiments/**`
- any file consumed as the 4C main evidence chain

## Current Outcome

- no mainline promotion
- no asset-root mutation
- no GPU release
- no 4C narrative change
