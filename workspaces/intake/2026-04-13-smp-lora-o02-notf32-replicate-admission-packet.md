# 2026-04-13 SMP-LoRA O02 no-TF32 Replicate Admission Packet

## Decision Stub

- `candidate_key`: `O02/batch14-throughput/no-TF32-rerun2`
- `status`: `closed after rerun3`
- `release_scope`: `one active GPU run only`
- `active_gpu_question_if_launched`: `none`

## Hypothesis

在恢复历史 `unseeded` 行为之后，`no-TF32` 可能仍能保留 `batch14 throughput` 的收益带，并降低一部分数值波动；如果它再次落在收益带内，它就能升级成真正值得继续复验的 next-question 候选。

## Asset Requirement

- `scripts/train_smp_lora.py` 已恢复“默认不显式 seed”
- 固定 `local_model / member_dir / nonmember_dir`
- 固定 `rank=1 / lambda=0.1 / epochs=10 / batch_size=14 / num_workers=8`
- 固定 `throughput_mode=true / cudnn_benchmark=true / non_blocking_transfers=true / allow_tf32=false`
- 输出目录单独隔离

## Compute Budget

- `budget`: `<= 2 GPUh`
- `single_run_expectation`: `~1 GPUh` 级别
- `checkpoint_policy`: `save_every=500`

## Stop Conditions

- 训练在早期出现卡死、OOM 或日志长时间不前进
- 结果回退到 `AUC >= 0.50`
- 运行态与既有 `batch14 throughput` 合同不再可比
- 新结果只复制“单点模糊正样本”而不增加信息密度

## Expected Artifact

- 新的输出目录
- `config.json / stdout.log / stderr.log / checkpoints`
- 若训练完成，再补 `evaluation.json`
- 单次 run verdict：`promote to replicate-worthy candidate` 或 `close as no-go`

## Why This Candidate Wins Over Others

1. 它比 seeded 复验更直接：
   - seeded 路径已经被 `123` 和 `42` 证明会显著退化
2. 它比 `O03/O04/O08` 更便宜：
   - 仍在当前最强候选附近，只改一个运行时变量
3. 它比继续盲目 `batch14 throughput` 复跑更有信息增量：
   - 当前问题不是再堆一个同配置样本，而是验证 `TF32` 是否是波动来源之一

## Current Evidence Update

- `no-TF32 rerun1`: `AUC = 0.3957`
- `no-TF32 rerun2`: `AUC = 0.3838`
- `no-TF32 rerun3`: `AUC = 0.5306`
- 当前判定：
  - `no-TF32` 先升到 `replicate-worthy candidate`，但第三个样本失效后被降回 `mixed / not a stabilization answer`
  - 当前不能继续写成 `default template`
  - 当前没有 active run

## Current Todolist

- `done`: 恢复历史 unseeded 合同
- `done`: 用测试锁定 `--seed` 默认不显式注入
- `done`: 明确这条题的 hypothesis / asset / budget / stop conditions / artifact
- `in_progress`: 启动单条 no-TF32 rerun2
- `todo`: 监控日志与 checkpoint cadence
- `todo`: 完成后评估并写回 verdict
- `blocked`: 第二条 GPU 长任务
