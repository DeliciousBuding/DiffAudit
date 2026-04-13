# 2026-04-13 SMP-LoRA O02 Template Decision Packet

## Decision

- `decision_status`: `sealed`
- `active_gpu_question`: `none`
- `template_verdict`: `batch14 throughput is the strongest candidate with variance note; it is not the default template`
- `release_effect`: `no new GPU release`
- `scope`: `rank=1 / lambda=0.1 / epochs=10 local DDPM/CIFAR10 protocol`

## Why This Packet Exists

`O02` 在本轮已经收集到足够多的同协议证据。继续为同类 `rerun` 烧 GPU，不再提供足够高的信息密度。当前更有价值的是把结论写清，固定边界，并把下一条 GPU 问题的准入条件独立出来。

## Evidence Table

### Rank1 Core Rungs

- `batch12 throughput`: `0.4056 / 0.4264 / 0.4773`
- `batch12 legacy`: `0.4224`
- `batch13 throughput`: `0.4482`
- `batch14 throughput`: `0.3708 / 0.4188 / 0.2971 / 0.4083 / 0.5250 / 0.4929 / 0.4431`
- `batch14 legacy`: `0.6485`
- `batch15 throughput`: `0.3139 / 0.6250`
- `batch16 throughput`: `0.5554`

### Throughput Side Probes

- `batch14 workers4`: `0.65625`
- `batch14 workers6`: `0.5321969696969697`
- `batch14 no-cudnn-benchmark`: `0.6388888888888888`
- `batch14 no-TF32`: `0.39570607028753994`
- `batch14 seed123`: `0.6222222222222221`
- `batch14 seed42`: `0.6624649859943978`

### Supporting Capacity Comparators

- `rank2 batch12 throughput`: `0.4847`
- `rank4 batch12 throughput`: `0.5532`
- `rank4 batch8 throughput`: `0.6159`
- `rank4 batch8 legacy`: `0.4872 / 0.4944`

## Decision Logic

1. `batch13` 不能升级：
   - 它没有超出 `batch12` 的带宽，只能算 `no-gain`
2. `batch15` 不能升级：
   - 第二次观测直接回到 `0.6250`
   - 单次好结果不再成立
3. `batch16` 不能升级：
   - 结果已经回退到明显退化区
4. `batch14 legacy` 直接否定“只是 batch14 更优”：
   - `0.6485` 说明收益依赖 `throughput_mode`
5. `batch14 throughput` 仍是最强候选：
   - 它拥有当前最强单次点
   - 七次 workers8 无 seed 结果的均值约 `0.4223`
6. `batch14 throughput` 仍不能升成默认模板：
   - 七次结果的波动仍高
   - 同一合同下还不足以写成“稳定默认”
7. 当前稳定化 side probes 都不成立：
   - `workers4/6` 失败
   - `no-cudnn-benchmark` 失败
   - `seed123/42` 失败
8. `no-TF32` 仍只有一次样本：
   - 只够当 future candidate
   - 不够当 released next question

## Hard Wording

- 可以写：
  - `batch14 throughput is the strongest candidate with variance note`
  - `the gain depends on throughput_mode rather than batch14 alone`
  - `O02 has enough evidence for packet closure`
- 不可以写：
  - `batch14 throughput is the stable default template`
  - `batch12 is the low-variance narrow-band anchor`
  - `seeded execution solved the variance problem`
  - `workers4/6 or no-bench stabilized throughput`

## Effective GPU Utilization Verdict

- `verdict`: `do not continue blind reruns`
- 原因：
  - 当前信息瓶颈不再是“再补一个近似同分布样本”
  - 当前更缺的是下一题合同，而不是更多同题噪声
  - 同类复跑会继续占用 GPU，但不一定降低决策不确定性

## Deferred Candidate Rungs

这些不是 released item，只是如果未来要重新申请 GPU，最值得先写 packet 的候选。

### Candidate A: `batch14 throughput no-TF32 replicate`

- `hypothesis`: 关闭 TF32 可能保留 throughput 收益并降低一部分数值波动
- `asset_requirement`: 现有 `rank1/lambda0.1/epochs10/batch14` 配置、同评估协议、至少 2 次新样本
- `compute_budget`: `<= 2 GPUh`
- `stop_conditions`: 任一新样本回退到 `>= 0.50`；无法保持同协议；新样本只复制已有模糊结论
- `expected_artifact`: `no-TF32 x3 packet with mean/std and release recommendation`

### Candidate B: `seeded reproducibility cross-check`

- `hypothesis`: 当前 seed123/42 的失败可能说明某些 seed 区域极差，但需要更严格的 seed 合同才知道是否值得保留
- `asset_requirement`: 显式 seed contract、固定 workers8/throughput_mode、至少 3 个预注册 seed
- `compute_budget`: `<= 3 GPUh`
- `stop_conditions`: 再次出现显著退化；无法给出比 unseeded 更清楚的结论；只是在把 seed 敏感性显式化
- `expected_artifact`: `seed sensitivity packet, not a template promotion`

### Candidate C: `O03/O04/O08 separate next-question review`

- `hypothesis`: 更高信息增量可能已经不在 `O02` 模板内，而在新的 frontier 或 comparator 设计
- `asset_requirement`: 单独的 `hypothesis / asset requirement / compute budget / stop conditions / expected artifact`
- `compute_budget`: `TBD by separate packet`
- `stop_conditions`: 仍然只是重跑旧结论；没有比 `O02` 封口更高的信息密度
- `expected_artifact`: `next-question admission packet`

## Current Todolist

- `done`: 固定 `batch14 throughput` 结论
- `done`: 固定 `batch13 / batch15 / batch16` 边界
- `done`: 固定 `workers4/6 / no-bench / seeded` 的失败结论
- `done`: 固定 `no-TF32` 为单样本候选
- `done`: 将 `active_gpu_question` 置为 `none`
- `todo`: 单独起草下一题 packet 前，不重新申请 GPU
- `blocked`: `O03/O04/O08`
- `blocked`: `SecMI / TMIA-DM / Finding NeMo`
