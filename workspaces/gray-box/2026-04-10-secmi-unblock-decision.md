# 2026-04-10 SecMI Unblock Decision

## 目的

这份文档只回答一个问题：

- `SecMI` 当前是否已经解除 `blocked baseline`，足以重新进入 `Phase E` intake review。

它不是：

- run 授权
- GPU 申请单
- baseline 已恢复的公告
- admitted 升级文档

## 当前固定前提

- `PIA = gray-box mainline`
- `PIA paper-aligned confirmation = document-layer conditional rank 1, execution-layer no-go`
- `Finding NeMo = zero-GPU hold`
- `DP-LoRA = comparability / intake hardening only`
- `SecMI = intake review priority #3`

## 当前已知事实

来自现有 probe 与状态页的可复核事实只有这些：

1. `SecMI` 的 synthetic smoke 已存在
2. 当前真实 asset probe 仍报错：
   - `Missing SecMI flagfile: REPLACE_WITH_SECMI_MODEL_DIR\\flagfile.txt`
3. 当前阻塞类型已明确为：
   - `asset/layout blocker`
   - 不是 `GPU blocker`
   - 不是 `algorithm blocker`

## 当前 blocker

当前唯一硬 blocker 继续固定为：

- `real flagfile.txt + matching checkpoint root / model root layout not yet available`

这意味着当前不能写成：

- `asset-ready`
- `execution-ready`
- `baseline restored`

## 当前正式裁决

- `decision_grade = decision-grade`
- `current_verdict = not-yet`
- `selected_baseline = SecMI`
- `current_state = blocked baseline`
- `execution_release = none`
- `gpu_release = none`

## 为什么当前只能是 not-yet

1. 当前仍缺真实 `flagfile.txt`
2. 当前仍缺与 `flagfile` 相匹配的 checkpoint root / model root layout
3. 在这两个条件未满足前，任何 `probe / dry-run / runtime` 都仍然只会重复同一阻塞，而不是产生新的决策信息

## Unblock Gate

只有当下面条件同时满足时，才允许把当前 verdict 从 `not-yet` 重审为 `go / no-go`：

1. 拿到真实 `flagfile.txt`
2. 拿到可解析的真实 checkpoint root / model root layout
3. 二者路径关系已写成可复核 layout contract
4. 用最小 `runtime-probe-secmi` 重新验证一次，并记录输出

## 预期产物

如果未来真的满足 unblock gate，下一步最小产物也仍然只能是：

- 一份新的 `SecMI unblock memo`
- 一次最小 `runtime-probe` 结果
- `go / no-go` 执行裁决

不是：

- 新 GPU 任务
- 新 admitted baseline
- 当前可直接与 `PIA` 争主线

## 队列影响

当前这份 decision 只说明：

- `SecMI` 继续保留在 intake review priority #3

它不说明：

- `SecMI` 已越过 `Finding NeMo` 或 `DP-LoRA`
- `SecMI` 已可申请 GPU
- `SecMI` 已恢复为可执行 baseline
- `SecMI` 已进入 released queue item

## 当前明确不做

- 不改 `PIA` 主讲线
- 不改 `Finding NeMo zero-GPU hold`
- 不把 `SecMI` 写成 asset-ready
- 不把 `SecMI` 写成 execution-ready
- 不为 `SecMI` 新开 GPU

## 下一步

在当前 decision 固定后，下一条非 GPU 槽位建议切到：

- `TMIA-DM intake` 的协议与资产拆解

而不是继续反复重写 `SecMI` 的同一 blocker。
