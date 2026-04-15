# 2026-04-14 Baseline vs SMP-LoRA vs W-1 Comparator Admission Packet

## Decision stub
- `candidate_key`: `comparator/baseline-smp-lora-w1`
- `status`: `planning / readiness gate`
- `release_scope`: `analysis-only; no new GPU question release until comparator verdict`
- `active_gpu_question_if_launched`: `none`

## Hypothesis
当前的 SMP-LoRA optimizer/lr frontier 已判定 `closed-mixed-no-go`（`AdamW` run 在 `AUC=0.5923`、`SGD(momentum)` 回退至 `AUC=0.4211`），而 `batch14 throughput` 的 best evidence band 依旧落在 baseline Adam anchor 下。既然单一 optimizer/lr 并未给出 decisive rescue 信号，下一步的唯一有意义的 GPU action是分析 `baseline`、`SMP-LoRA`（最强 candidate）和 `W-1`（strong defended comparator）之间的差异：如果 comparator 仍然无法压过 W-1 defensive wall 或进一步拉开距离，就没有理由在 SMP-LoRA 上释放新 question；反之，在 comparator 中如果 SMP-LoRA 发生实质性超越，则可打通资源门控。

## Assets / Requirements
- `Research/outputs/smp-lora-t06-rank1-lambda01-ep10-batch14-throughput-adamw-lowlr-20260413-201437`
- `Research/outputs/smp-lora-t06-rank1-lambda01-ep10-batch14-throughput-sgd-matchedlr-20260413-202327`
- `Research/workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408`
- `baseline Adam anchor (batch14 throughput / legacy) metadata`
- `latest `unified-attack-defense-table.json` entry for W-1 comparator`
- `intake manifest for GSA/W-1 to align defense metrics`
- `hypothesis doc for `PIA + GSA/W-1` matured mainline to keep closed gating`

## Compute budget & orchestration
- `budget_total`: `<= 12 GPUh` (fits within one RTX 4070 session)
- `release_mode`: `serial comparator runs only (no parallelized grid)`
- `suggested rungs`:
  1. `baseline Adam / batch14 throughput (reference output already recorded)`
  2. `SMP-LoRA best candidate (lambda=0.1, epoch=10, batch=14, throughput mode, AdamW + low lr, SGD momentum)`
  3. `W-1 strong-v3 full-scale defended comparator (batch32 same-protocol diagnostics / DPDM strong-v3)`
- `checkpoint_policy`: `save_every=500`, final checkpoint pinned for evaluation
- `compute_owner`: `Researcher triage + LocalOps schedule only if comparator verdict warrants GPU release`

## Stop conditions
- SMP-LoRA comparator run fails to beat either W-1 defensive AUC/ASR pairs by more than 0.02 AUC in repeated evaluation → `no-GPU-release; document no-go`
- SMP-LoRA run inferior to baseline anchor on both AUC and ASR → pivot to `comparator/no-go` narrative, no further optimizer tuning
- W-1 comparator run exposes unexpected drift (e.g., ASR jumps due to scheduling) → pause and root-cause on DPDM artifacts before any SMP line
- Evidence chain (evaluation.json, checkpoint pointer, logs) missing `boundary` fields → stop and hold GPU until metadata is complete

## Expected artifact
- `evaluation.json` for each comparator rung (baseline, SMP-LoRA, W-1) with `track`, `attack`, `defense`, `AUC`, `ASR`, `note`, `boundary`
- `stdout.log / stderr.log` with runtime metadata, GPU configuration, training cost, and hyperparameter summary
- `config.json` capturing optimizer, lr, throughput_mode, permit_tf32, cudnn flags
- `checkpoint` pointers (milestone + final) for auditing, plus `snapshot.json` summarizing comparator ordering
- `comparator narrative brief` describing whether SMP-LoRA beats W-1 or defaults to no-go, to be merged into `mainline-narrative` and `competition-evidence-pack`

## Current hold rationale
1. `SMP-LoRA T06` optimizer/lr frontier already judged `closed-mixed-no-go` (`AdamW`/`SGD(momentum)` runs fail to improve over base), so there is no new GPU question to release.
2. Active GPU question is `none`; `nvidia-smi` shows only desktop/C+G workloads, and latest outputs date `2026-04-13`.
3. Future release must be built on a comparator that proves SMP-LoRA can breach the W-1 defense wall or clarifies that cavitation is tangential to comparator design.
4. Leader / Platform audiences already expect `PIA + GSA/W-1` as mainline; releasing SMP-LoRA without comparator evidence would break that narrative.

## Next gating conditions to release GPU question
- comparator packet must deliver a clear verdict: either SMP-LoRA wins a head-to-head vs W-1 (>=0.02 AUC uplift while defending ASR drop), or comparator documents a stable no-go.
- `baseline vs SMP-LoRA vs W-1` evaluation artifacts are promoted into `Research/workspaces/intake/phase-e-candidates.json` with `phase=Comparator`.
- CPU-side provenance check for `PIA` remains in blocker state; only when comparator verdict aligns with that gate can we close `PIA` blocker and reconsider GPU release.
- A new `admitted summary` entry referencing comparator results must be ready for `Local-API` before any further GPU question release.
