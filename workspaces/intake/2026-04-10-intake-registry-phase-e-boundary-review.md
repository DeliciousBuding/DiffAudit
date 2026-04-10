# 2026-04-10 Intake Registry Phase E Boundary Review

## Status Panel

- `owner`: `research_leader`
- `artifact_type`: `decision-grade boundary / indexing review`
- `decision_scope`: `writing + schema-only, non-GPU`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `active_gpu_question`: `none`
- `bridge_status`: `closed-frozen`

## 目的

这份文档只回答一个问题：

- `Project/workspaces/intake/index.json` 是否应该从当前的 promoted contract 目录扩展到包含 `Phase E` 候选索引。

它不是：

- run 授权
- GPU 申请单
- admitted 升级文档
- 新执行主线公告

## 当前事实

1. `index.json.entries[]` 当前承载的是 `PIA` 与 `GSA` 的 promoted intake contracts。
2. 这些记录都带有：
   - `contract_key`
   - `manifest`
   - `compatibility.commands`
3. 这套结构与 `Local-API` 的 promoted asset / job routing 契约直接相关。
4. 当前 `Phase E` 候选则仍停留在：
   - `Finding NeMo = adapter-complete zero-GPU hold`
   - `DP-LoRA = comparability / intake hardening only`
   - `SecMI = blocked baseline`
   - `TMIA-DM = protocol-and-asset decomposition intake only`
5. `PIA paper-aligned confirmation` 仍是：
   - `document-layer conditional rank 1`
   - `execution-layer no-go`

## 风险判断

如果把 `Phase E` 候选直接混进 `entries[]`，会同时引入三类漂移：

1. 把候选池误写成可被 `Local-API` 路由的 contract surface
2. 把 `intake / candidate / blocked baseline / zero-GPU hold` 偷换成 admitted 或 execution-ready
3. 把文档层条件排序和执行层默认顺位重新混写

## 正式决策

- `decision_grade = decision-grade`
- `entries[] = keep promoted/system-intake-ready contracts only`
- `phase-e-candidates.json = add as research-owned sibling file`

换言之：

1. 保持 `index.json` 只服务 promoted intake contracts
2. 不扩展 `entries[]`
3. 新增一个并列 sibling file：`phase-e-candidates.json`

## 结构约束

`phase-e-candidates.json` 只允许承载：

- 文档层条件排序
- 执行层默认候选顺位
- 当前 verdict / shape / release-state 快照
- source doc 指针

`phase-e-candidates.json` 明确禁止承载：

- `contract_key`
- `manifest`
- `compatibility.commands`
- `admission`
- 任何 admitted / benchmark-ready / execution-ready 声称

## 当前写回形态

### 文档层条件排序

只保留：

1. `PIA paper-aligned confirmation`
   - `current_verdict = blocked`
   - `current_shape = document-layer conditional rank 1`
   - `execution_layer_status = no-go`

### 执行层默认顺位

1. `Finding NeMo`
   - `current_verdict = not-yet`
   - `current_shape = adapter-complete zero-GPU hold`
   - `gpu_release = none`
   - `queue_state = not-requestable`
2. `DP-LoRA`
   - `current_verdict = not-yet`
   - `current_shape = comparability / intake hardening only`
3. `SecMI unblock`
   - `current_verdict = not-yet`
   - `current_shape = blocked baseline`
4. `TMIA-DM intake`
   - `current_verdict = not-yet`
   - `current_shape = protocol-and-asset decomposition intake only`

## 为什么这次值得做

1. `Finding NeMo` 已经进入 `zero-GPU hold`，候选池不能继续只靠口头排序维持秩序。
2. `DP-LoRA / SecMI / TMIA-DM` 都已各自形成 decision-grade intake 文档，缺的是统一 machine-readable indexing。
3. 当前不存在 active GPU question，这轮最值得推进的是结构性治理，而不是强行开新 run。

## 当前明确不做

- 不改变 admitted hierarchy
- 不释放任何 GPU
- 不把 `Phase E` 候选写成 job-ready / benchmark-ready
- 不重开 `white-box same-protocol bridge`
- 不把 `PIA paper-aligned confirmation` 从文档层条件排序提前放行到执行层
