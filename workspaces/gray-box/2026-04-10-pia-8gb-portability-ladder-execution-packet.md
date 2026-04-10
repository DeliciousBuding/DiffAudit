# 2026-04-10 PIA 8GB Portability Ladder Execution Packet

## Status Panel

- `owner`: `boss + research_leader + gpu_scheduler_leader`
- `track`: `gray-box`
- `method`: `PIA`
- `packet_type`: `release-free current-mainline-support`
- `hardware_target`: `RTX4070 laptop 8GB`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `current_blocker`: `checkpoint/source provenance`
- `active_gpu_question`: `none`
- `admission_effect`: `none`
- `updated_at`: `2026-04-10 21:36 +08:00`

## 目的

这份 packet 只回答一个问题：

- 在 `RTX4070 laptop 8GB` 现实下，当前 `PIA + provisional G-1(all_steps)` 是否还能以 `release-free supporting run` 方式稳定保留同向 `adaptive` 信号与可恢复执行链。

它不回答：

- `PIA` 是否升级为 `paper-aligned`
- `PIA` 是否形成新的 admitted 主档
- `stochastic-dropout` 是否已成为 validated privacy win
- `Phase E` 是否因此打开

## Hard Boundary

本 packet 固定继承以下边界：

- `PIA` strongest claim 继续保持：
  - `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`
- 任何新 rung 都只算：
  - `current-mainline-support`
  - `supporting evidence`
  - `8GB portability / cost frontier evidence`
- 任何新 rung 都不允许写成：
  - `paper-aligned confirmation`
  - `admitted replacement`
  - `canonical summary refresh`
  - `validated privacy win`
  - `release passed`

## Evidence Anchors

### 已有锚点

1. `CPU32 adaptive-reviewed pair`
   - baseline:
     - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-cpu-32-adaptive/summary.json`
   - defense:
     - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-cpu-32-allsteps-adaptive/summary.json`
2. `GPU128 / GPU256 / GPU512` baseline + defense pairs
3. `GPU512 adaptive-reviewed baseline + all_steps + late_steps_only`
   - `workspaces/gray-box/2026-04-09-pia-gpu512-adaptive-ablation.md`
4. `PIA provenance dossier`
   - `workspaces/gray-box/2026-04-09-pia-provenance-dossier.md`

### 当前固定解释

- 当前攻击信号仍解释为：
  - `epsilon-trajectory consistency`
- 当前 defended mainline 仍解释为：
  - `stochastic-dropout(all_steps)` 打散该一致性信号
- 新 rung 的价值只在于：
  - 证明单卡 8GB 下可移植、可恢复、方向不漂移

## Review Gate

每一条 rung 都必须满足：

1. `admission_effect = none`
2. 只回答 portability / cost / adaptive direction，不偷带 admitted 升级
3. 产物至少包括：
   - `summary.json`
   - 完整命令
   - `workspace`
   - `batch_size / max_samples / adaptive_query_repeats`
   - baseline / defense 配对关系
4. 如果结果只是在重复旧结论，不继续扩档
5. 失败也必须能留下可审查 artifact，而不是只留下口头判断

## Execution Ladder

### Stage 0: Request + Probe + Preview

#### 目标

- 在真正占用主 GPU 前，确认：
  - `PIA` 真资产路径仍健康
  - `cuda:0` 上的 member/non-member preview 可运行

#### 命令

```powershell
conda run -n diffaudit-research python -m diffaudit runtime-probe-pia `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0
```

```powershell
conda run -n diffaudit-research python -m diffaudit runtime-preview-pia `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --preview-batch-size 2
```

#### 通过条件

- `runtime-probe-pia` 返回 `ready`
- `runtime-preview-pia` 在 `cuda:0` 成功完成，不报 `OOM`

#### 停止条件

- preview 失败或 `OOM`
- preview 期间观察到明显资源不稳定

#### 预期产物

- 控制台 JSON 结果
- round report 中写清 probe / preview 结论

### Stage 1: Anchor Reuse

#### 当前处理

- 直接复用：
  - `CPU32 adaptive-reviewed pair`
- 不重跑 CPU32

#### 目的

- 把 `CPU32` 固定为 non-GPU 锚点，避免为 portability 机械重跑已有小样本档

### Stage 2: Rung-B `GPU128 adaptive pair`

#### 问题

- `GPU128 + adaptive repeats` 是否仍保留 baseline > defense 的同向关系

#### 假设

- 在 `workspace-verified` 资产线上，`all_steps` defended setting 在 `GPU128` 下仍能压低 `adaptive AUC / ASR`

#### 命令

baseline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 128 `
  --batch-size 8 `
  --adaptive-query-repeats 3 `
  --provenance-status workspace-verified
```

defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 128 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --dropout-activation-schedule all_steps `
  --adaptive-query-repeats 3 `
  --provenance-status workspace-verified
```

#### 预算

- `2` runs
- 单卡串行
- 默认 `batch_size = 8`

#### 停止条件

- 任一 run `OOM`
- defense 不再低于 baseline
- 单边 wall-clock 超过历史 `GPU128` 非 adaptive 结果约 `2x`

#### 通过条件

- paired `summary.json` 完整落盘
- `adaptive` 指标方向不翻转

#### 预期产物

- baseline `summary.json`
- defense `summary.json`
- 一条 `GPU128 adaptive portability` 对照结论

### Stage 3: Rung-C `GPU256 adaptive pair`

#### 问题

- `GPU256` 是否能成为单卡 8GB 的主 decision rung

#### 前置条件

- `GPU128 adaptive pair` 已完成
- 方向稳定
- 没有出现 `OOM / runtime > 2x / 明显热不稳定`

#### 命令模板

与 `GPU128` 相同，仅替换：

- `workspace`
- `--max-samples 256`

#### 预算

- `2` runs
- 单卡串行
- 默认 `batch_size = 8`

#### 停止条件

- 只重复 `GPU128` 旧结论，无新增信息
- defense 方向翻转或明显弱化
- 运行不稳定

#### 预期产物

- paired `summary.json`
- 若成功，则给出：
  - `decision rung = GPU256`
- 若失败，则给出：
  - `8GB do-not-expand-beyond-GPU128`

### Stage 4: Rung-D `GPU512 ceiling check`

#### 默认策略

- 默认不重跑
- 直接复用现有 `GPU512 adaptive-reviewed` 结果作为 ceiling anchor

#### 只有在以下情况下才允许重开

- `GPU256` 与历史 `GPU512 adaptive` 出现不可解释冲突
- 该冲突会影响 portability frontier 判断
- 已单独写清：
  - `hypothesis`
  - `protocol delta`
  - `compute budget`
  - `stop condition`
  - `expected artifact`

#### admission effect

- `none`

## OOM / Recovery Chain

统一恢复顺序：

1. `batch_size 8 -> 4`
2. `batch_size 4 -> 2`
3. `max_samples 256 -> 128 -> 64 -> 32`
4. 回到 `Stage 0 preview`
5. 若 preview 都不稳定，则本轮停止，不横向改题

恢复纪律：

- 不做半程续跑
- 只从最近一个成功阶段重新发下一条命令
- 每个阶段单独使用独立 `workspace`

## Allowed Claim

本 packet 允许写：

- `release-free gray-box supporting run`
- `8GB portability evidence`
- `direction preserved under bounded 8GB budget`
- `current-mainline-support only`

本 packet 禁止写：

- `new PIA mainline`
- `paper-aligned confirmation`
- `admitted update`
- `system default update`
- `canonical best summary`

## Immediate Round-26 Decision

本轮唯一放行动作固定为：

1. `Stage 0 probe + preview`
2. 若成功，执行 `Stage 2 GPU128 adaptive pair`
3. 若 `GPU128 adaptive pair` 方向稳定且运行健康，允许继续到 `Stage 3 GPU256 adaptive pair`

本轮明确不做：

1. 不直接跳到 `GPU256`
2. 不重跑 `GPU512`
3. 不把结果写成 `paper-aligned`
4. 不把 supporting rung 写成新的 admitted 主档

## Round-26 Execution Outcome

### Stage 0

- `runtime-probe-pia` on `cuda:0` = `ready`
- `runtime-preview-pia` on `cuda:0` = `ready`

### Stage 2 `GPU128 adaptive pair`

- baseline:
  - `adaptive AUC = 0.817444`
  - `adaptive ASR = 0.765625`
  - `wall_clock = 49.544877s`
- defense `all_steps`:
  - `adaptive AUC = 0.806274`
  - `adaptive ASR = 0.761719`
  - `wall_clock = 50.667101s`

结论：

- `GPU128` 在 `RTX4070 8GB` 下稳定可跑
- `all_steps` 继续保持同向 privacy-drop
- 该 rung 通过

### Stage 3 `GPU256 adaptive pair`

- baseline:
  - `adaptive AUC = 0.841293`
  - `adaptive ASR = 0.78125`
  - `wall_clock = 95.934721s`
- defense `all_steps`:
  - `adaptive AUC = 0.829559`
  - `adaptive ASR = 0.763672`
  - `wall_clock = 213.126361s`

结论：

- `GPU256` 仍保持同向 privacy-drop
- 但 defense 侧 wall-clock 已显著恶化
- 当前最安全的 round-26 决策是：
  - `stop before GPU512 rerun`
  - `retain GPU128 as the quickest portable pair`
  - `retain GPU256 as the decision rung with cost warning`

配套 frontier note：

- `workspaces/gray-box/2026-04-10-pia-8gb-supporting-frontier-note.md`
