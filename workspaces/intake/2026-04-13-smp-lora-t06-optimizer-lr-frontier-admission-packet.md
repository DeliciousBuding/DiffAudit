# 2026-04-13 SMP-LoRA T06 Optimizer/LR Frontier Admission Packet

## Decision Stub

- `candidate_key`: `T06/batch14-throughput/optimizer-lr-frontier`
- `status`: `closed-mixed-no-go`
- `release_scope`: `one GPU question only; up to 4 serial runs`
- `active_gpu_question_if_launched`: `none`

## Hypothesis

`batch14 throughput` 当前的波动可能不是方法 ceiling 已到，而是优化器与学习率配置错位导致的训练不稳定。若在固定 `rank1 / lambda=0.1 / epochs=10 / batch14 / throughput` 的合同下，只改变 `optimizer + lr` 就能把结果重新拉回收益带，则 `SMP-LoRA` 仍值得继续进入下一轮 `capacity / trainable-block frontier`；若这些组合仍普遍失败，就应停止继续救活并转向 comparator/no-go。

## Asset Requirement

- 固定 `local_model / member_dir / nonmember_dir`
- 固定 `rank=1 / lambda=0.1 / epochs=10 / batch_size=14 / num_workers=8`
- 固定 `throughput_mode=true / allow_tf32=true / cudnn_benchmark=true / non_blocking_transfers=true`
- 固定当前攻击器、评估 schema、输出 schema、checkpoint pointer 格式
- 保持历史 `unseeded` 默认行为：
  - 不显式传 `--seed`
- 仅使用仓内现有 optimizer 实现：
  - `adam`
  - `adamw`
  - `sgd(momentum)`

## Compute Budget

- `budget_total`: `<= 16 GPUh`
- `release_mode`: `serial only / no parallel release`
- `suggested_rungs`:
  1. `historical anchor only: adam + current lr pair`
  2. `released now: adamw + conservative lr pair (lora_lr=5e-5, proxy_lr=5e-4)`
  3. `next if needed: sgd(momentum=0.9) + matched lr pair`
  4. `optional fourth rung only if one of the released runs enters the worth-continuing band`
- `checkpoint_policy`: `save_every=500`

## Stop Conditions

- 全部组合都落在当前失败区间附近
- 没有任何组合接近或超过 `batch14 legacy / seed42` 这类高点
- comparability 因实现差异被破坏
- 任一 run 出现训练发散、checkpoint 缺失、或总时长超预算 2 倍

## Expected Artifact

- `3-4` 个独立输出目录
- 每个目录都包含：
  - `config.json`
  - `stdout.log / stderr.log`
  - `milestone checkpoints`
  - `final checkpoint`
  - `evaluation.json`
- 一张统一 optimizer/LR ablation 表：
  - `optimizer`
  - `lora_lr`
  - `proxy_lr`
  - `wall-clock`
  - `AUC`
  - `Accuracy`
  - `checkpoint pointer`
- 一条明确裁决：
  - `continue to capacity frontier`
  - `pivot to comparator`
  - `direct no-go`

## Current Results

| rung | optimizer | lora_lr | proxy_lr | AUC | Accuracy | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| `historical anchor` | `adam` | `1e-4` | `1e-3` | `0.3708 / 0.4188 / 0.2971 / 0.4083 / 0.5250 / 0.4929 / 0.4431` | mixed | current best evidence band, but still high-variance |
| `run1` | `adamw` | `5e-5` | `5e-4` | `0.5923` | `0.5789` | worse than baseline; negative signal |
| `run2` | `sgd(momentum=0.9)` | `1e-4` | `1e-3` | `0.4211` | `0.3684` | returns to the historical Adam band, but does not beat it |

## Current Verdict

- `AdamW + lower lr` 明显恶化到 `AUC=0.5923`，不支持 optimizer rescue
- `SGD(momentum)` 回到 `AUC=0.4211`，只说明“换 optimizer 仍可回到历史 Adam 收益带”，但没有带来超越既有 `Adam` anchor 的新增信息
- 因此 `T06` 当前结论是：
  - `mixed`
  - `no optimizer-rescue evidence beyond the historical Adam anchor`
  - `do not continue to capacity frontier from optimizer evidence alone`
  - `next recommended move = pivot to baseline vs SMP-LoRA vs W-1 comparator`

## Why This Candidate Wins

1. 它直接回答当前唯一还值得占 GPU 的问题：
   - `batch14 throughput` 的波动究竟来自 optimizer/lr 错位，还是方法 ceiling 已到
2. 它比继续做 seed/epoch/runtime tweak 更有信息密度
3. 它比立即转 comparator 更便宜，也更接近现有资产
4. 仓内现已具备 `adam / adamw / sgd(momentum)` 最小实现，因此 packet 可执行

## Current Todolist

- `done`: `O03 / O04 / no-TF32` 三条稳定化失败路径收口
- `done`: 双主线口径落盘
- `done`: 为 `T06` 写清 hypothesis/assets/budget/stop conditions
- `done`: 为训练链补最小 optimizer 支持
- `done`: 将历史 `Adam + current lr` 结果固定为 T06 anchor，不再重复占 GPU
- `done`: 完成 `AdamW + lower lr run1` 并补评估，结果 `AUC=0.5923`
- `done`: 完成 `SGD(momentum) + matched lr run2` 并补评估，结果 `AUC=0.4211`
- `done`: 生成统一 optimizer/LR ablation 表
- `done`: 裁决 `pivot to comparator`
- `todo`: 若要继续 `SMP-LoRA`，下一步应改写为 comparator packet，而不是继续 optimizer/lr frontier
- `blocked`: `Finding NeMo / SecMI / TMIA-DM / 第二条 GPU 长任务`
