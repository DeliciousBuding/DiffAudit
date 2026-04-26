# 2026-04-14 Baseline vs SMP-LoRA vs W-1 Comparator Admission Packet

## Decision stub
- `candidate_key`: `comparator/baseline-smp-lora-w1`
- `status`: `bounded local-board review / no auto-release`
- `release_scope`: `analysis-only; no new GPU question release until comparator verdict`
- `active_gpu_question_if_launched`: `none`

## Hypothesis
当前 packet 的职责已不再是“决定要不要先做 comparator”，因为这一步已经完成。当前职责是冻结完成后的 bounded local-board truth：`baseline local63`、frozen `SMP-LoRA local63`、以及 refreshed `W-1 strong-v3 local63` 已经构成第一张 honest same-asset local comparator board。该板在共享主指标 `AUC` 上的当前排序是 `SMP-LoRA (0.34375) < W-1 local63 (0.474175) < baseline (0.5565217391304348)`。这说明 successor lane 已经获得一个 bounded local comparator win，但这仍然只是 local-board verdict，不是 admitted white-box upgrade，也不会自动释放新的 GPU 问题。

## Assets / Requirements
- `Research/outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `Research/outputs/smp-lora-sweep/sweep_results.json`
- `Research/workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-local63-20260416/summary.json`
- `Research/workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`
- `Research/workspaces/white-box/2026-04-16-dplora-comparator-release-review-refresh.md`

## Compute budget & orchestration
- `budget_total`: `none new`
- `release_mode`: `analysis-only / packet refresh`
- `suggested rungs`:
  1. `baseline local63`
  2. `SMP-LoRA local63 frozen candidate`
  3. `W-1 strong-v3 local63 refresh`
- `checkpoint_policy`: `frozen existing artifacts only`
- `compute_owner`: `Researcher triage`

## Stop conditions
- if anyone tries to read the local-board win as:
  - admitted white-box upgrade
  - full-scale benchmark replacement
  - automatic GPU release trigger
  then this packet should force `no`
- if the next proposed task is only another optimizer/lr rescue repeat, hold it as stale
- if a future proposal cannot name a bounded question beyond the completed local board, hold GPU at `none`

## Expected artifact
- a refreshed comparator intake packet with:
  - the three local rows
  - current local ordering
  - explicit boundary language
  - explicit `gpu_release = none`
- optional future companion:
  - a secondary-metric harmonization note, if a new bounded question is selected

## Current hold rationale
1. the old packet was written before the completed local comparator board existed, so its framing is now stale
2. the new local board already proves one bounded local comparator win over refreshed `W-1`
3. but that win remains local-board-only, not full-scale and not metric-complete
4. therefore the honest release gate stays `none` until a new bounded question is written explicitly

## Next gating conditions to release GPU question
- a future GPU ask must name one bounded question that is not already answered by the local board
- that ask must also explain why CPU-side packet refresh or metric harmonization is insufficient
- until then:
  - `phase = comparator / bounded local-board win`
  - `gpu_release = none`
