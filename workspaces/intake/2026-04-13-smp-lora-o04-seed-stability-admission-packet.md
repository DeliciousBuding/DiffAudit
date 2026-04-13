# 2026-04-13 SMP-LoRA O04 Seed Stability Admission Packet

## Decision Stub

- `candidate_key`: `O04/batch14-throughput-seed-stability-mini`
- `status`: `admitted-for-seed7-run1`
- `release_scope`: `one active GPU run only`
- `active_gpu_question_if_launched`: `rank1 / lambda=0.1 / epochs=10 / batch14 / throughput / seed=7`

## Hypothesis

`batch14 throughput` 当前最大的未决问题不是再换一个 runtime knob，而是它在显式 seed 下到底能否保持在收益带内。若预注册 seed 仍频繁掉出收益带，则这条线只能继续保留 `variance note`，不能再朝模板方向推进。

## Asset Requirement

- 固定 `local_model / member_dir / nonmember_dir`
- 固定 `rank=1 / lambda=0.1 / epochs=10 / batch_size=14 / num_workers=8`
- 固定 `throughput_mode=true / allow_tf32=true / cudnn_benchmark=true / non_blocking_transfers=true`
- 使用修复后的 `--seed` 合同
- 预注册 seed shortlist：`7 / 11 / 29`

## Compute Budget

- `budget_total`: `<= 3 GPUh`
- `current_release`: `seed=7 run1 only`
- `followup_release_rule`: 只有 `seed=7` 完成并评估后，才决定是否继续 `seed=11`

## Stop Conditions

- 当前 seed 结果回退到 `AUC >= 0.50`
- 训练卡住、checkpoint 不写、或运行态不再可比
- 两个已完成 seed 中失败率达到 `>= 50%`
- 新样本只重复已知“高方差且不可模板化”的结论

## Expected Artifact

- `seed=7` 输出目录
- `config.json / stdout.log / stderr.log / final checkpoint / evaluation.json`
- 单样本 seed verdict
- 若继续下一 seed，再累计成 stability matrix

## Why This Candidate Wins

1. 它直接回答当前 blocker：
   - `batch14 throughput` 的模板化问题本质上是稳定性问题
2. 它比 `O03` 便宜：
   - 不需要 16-24h 长训
3. 它比 `O08` 更少前置依赖：
   - 不需要额外 comparator 合同

## Current Todolist

- `done`: `no-TF32` 三样本复验并收口
- `done`: 恢复默认 unseeded 合同
- `done`: 为 `O04` 缩小版写清 hypothesis/assets/budget/stop conditions
- `in_progress`: 放行 `seed=7 run1`
- `todo`: 完成评估并决定是否继续 `seed=11`
- `blocked`: `O03 / O08 / 第二条 GPU 长任务`
