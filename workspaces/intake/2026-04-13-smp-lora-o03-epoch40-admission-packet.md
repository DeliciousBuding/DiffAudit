# 2026-04-13 SMP-LoRA O03 Epoch-40 Admission Packet

## Decision Stub

- `candidate_key`: `O03/batch14-throughput/epoch40-mini`
- `status`: `admitted-for-epoch40-run1`
- `release_scope`: `one active GPU run only`
- `active_gpu_question_if_launched`: `rank1 / lambda=0.1 / epochs=40 / batch14 / throughput / unseeded`

## Hypothesis

当前 `10 epochs` 也许只是窄窗口最优点。若在相同 `rank1 / lambda0.1 / batch14 / throughput` 合同下把训练延长到 `40 epochs` 后仍留在收益带，则 `O03` 会变成比 runtime/seed tweak 更有价值的下一条主线；若它明显回退，就可以把“短训优于长训”写得更硬。

## Asset Requirement

- 固定 `local_model / member_dir / nonmember_dir`
- 固定 `rank=1 / lambda=0.1 / batch_size=14 / num_workers=8`
- 固定 `throughput_mode=true / allow_tf32=true / cudnn_benchmark=true / non_blocking_transfers=true`
- 使用修复后的历史 `unseeded` 默认行为
- 输出目录单独隔离

## Compute Budget

- `budget_total`: `<= 4 GPUh`
- `current_release`: `epoch40 run1 only`
- `checkpoint_policy`: `save_every=500`

## Stop Conditions

- 指标明显回退到 `AUC >= 0.50`
- 训练卡住、checkpoint 不写、或运行态不再可比
- 中途 checkpoint 已经反复显示只会继续恶化
- 长训只是复制“10 epochs 更优”的已知结论而无新增信息

## Expected Artifact

- `epoch40` 输出目录
- `config.json / stdout.log / stderr.log / milestone checkpoints / final checkpoint / evaluation.json`
- `epoch40 vs epoch10` 的单条长训 verdict

## Why This Candidate Wins

1. 它直接回答“10 epochs 是否只是窄窗口最优”这个核心问题
2. 它比 `O08` 少依赖 comparator 合同
3. 它比继续玩 runtime/seed tweak 更接近真正的训练长度 frontier

## Current Todolist

- `done`: `no-TF32` 三样本复验并收口
- `done`: `O04 seed7` 单样本 no-go 收口
- `done`: 为 `O03 epoch40` 写清 hypothesis/assets/budget/stop conditions
- `in_progress`: 放行 `epoch40 run1`
- `todo`: 完成评估并决定是否需要 `epoch100` 或直接封口
- `blocked`: `O08 / 第二条 GPU 长任务`
