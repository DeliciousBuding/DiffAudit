# 2026-04-10 Phase E Candidate Surface Audit

## Status Panel

- `owner`: `research_leader`
- `artifact_type`: `decision-grade boundary audit`
- `decision_scope`: `writing-only, non-GPU`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `active_gpu_question`: `none`
- `bridge_status`: `closed-frozen`

## 目的

这份文档只回答一个问题：

- 在 round-17 / round-18 完成 `phase-e-candidates.json` 分层后，当前剩余高层治理与研究叙事文档里，是否还存在把 `phase-e-candidates.json` 误写成 `index.json.entries[]`、`Local-API` promoted contract surface、run release surface 或 admitted/benchmark surface 的表述。

它不是：

- run 授权
- GPU 申请单
- admitted 升级文档
- 新执行线启动公告

## 审计范围

本轮固定审计以下高层文档：

1. `ROADMAP.md`
2. `Agents/GLOBAL_TASK_BOARD.md`
3. `Project/ROADMAP.md`
4. `Project/docs/comprehensive-progress.md`
5. `Project/docs/reproduction-status.md`
6. `Project/docs/mainline-narrative.md`
7. `Project/docs/future-phase-e-intake.md`

## 当前硬边界

当前必须持续保持：

- `Project/workspaces/intake/index.json.entries[] = promoted / system-intake-ready contracts only`
- `Project/workspaces/intake/phase-e-candidates.json = candidate ordering / status only`
- `phase-e-candidates.json` 不构成：
  - `Local-API` contract surface
  - promoted contract surface
  - run release surface
  - admitted / benchmark registry

当前候选状态也必须持续保持不变：

- `Finding NeMo = adapter-complete zero-GPU hold`
- `DP-LoRA = comparability / intake hardening only`
- `SecMI = blocked baseline`
- `TMIA-DM = protocol-and-asset decomposition intake only`

## 审计结果

### 1. 高层治理文档

- `ROADMAP.md`：已明确 `phase-e-candidates.json` 独立于 promoted intake contracts，未把 candidate surface 写成 `Local-API` contract surface。
- `Agents/GLOBAL_TASK_BOARD.md`：已明确 `index.json.entries[]` 与 `phase-e-candidates.json` 分层，未把 candidate ordering 写成当前 release surface。

### 2. 研究路线图与总览文档

- `Project/ROADMAP.md`：已明确 `phase-e-candidates.json` 是唯一 machine-readable candidate ordering mirror，且 `index.json.entries[]` 继续只承载 promoted contracts。
- `Project/docs/comprehensive-progress.md`：已把当前最值得推进的唯一目标收口到 candidate registry 的高层同步，没有把候选写成 execution-ready。
- `Project/docs/reproduction-status.md`：已明确 `phase-e-candidates.json` 从 `index.json.entries[]` 中剥离，且不构成 `Local-API` promoted contract surface。
- `Project/docs/mainline-narrative.md`：已把 admitted 面与候选治理面拆开叙述，没有把 candidate metadata 混进系统读链。
- `Project/docs/future-phase-e-intake.md`：已明确 `phase-e-candidates.json` 只是排序与状态快照，不是 run release，也不是 admitted 升级入口。

## 正式裁决

- `decision_grade = decision-grade`
- `audit_verdict = hold-for-semantic-downgrade`
- `admitted_drift = none found`
- `operational-semantic-risk = present`
- `gpu_impact = none`
- `required_follow-up = rename candidate surface away from release / queue wording`

## Same-Round Resolution

本轮已按这份 HOLD 裁决完成 same-round rework：

1. `phase-e-candidates.json` 已从 `v1` 升级为 `v2`
2. 顶层候选列表已从 `execution_layer_default_order` 改为 `intake_review_priority_order`
3. 候选记录已移除：
   - `execution_release`
   - `gpu_release`
   - `queue_state`
4. 候选记录改为只保留：
   - `current_verdict`
   - `current_shape`
   - `current_boundary`
5. 相关叙事已把“默认放行顺序 / 最可放行候选”降级为：
   - `准入验证优先顺序`
   - `最先审查候选`

因此，这份审计的最终处置状态为：

- `hold resolved in the same round`
- `current_registry_posture = candidate-only review snapshot`

## 为什么这轮值得做

1. round-17 / round-18 已经完成结构分离与高层同步，但如果没有单独的 audit artifact，这个边界仍可能在后续文档更新中被重新混写。
2. 当前没有 active 主 GPU 问题，本轮最值得推进的是把 candidate surface 的“放行面”语义降级为“准入验证优先顺序 / 边界快照”，而不是强行开启新 intake 或新实验。
3. 这份审计结论可以作为后续新增候选、修改高层文档时的 review gate 基线。

## 下一次必须重开审计的触发条件

只要出现以下任一情况，就必须重新开一轮 candidate-surface audit：

1. 新增或调整 `Phase E` 候选记录
2. 任何文档重新描述 `Project -> Local-API -> Platform` 的系统读链
3. 任何文档开始把 `phase-e-candidates.json` 与 `Local-API`、`job routing`、`contract registry` 同句绑定
4. 任一候选状态从 `not-yet / blocked / hold` 进入 release review

## 当前明确不做

- 不改变 admitted hierarchy
- 不释放任何 GPU
- 不新增任何 `Phase E` 候选记录
- 不把 candidate metadata 回流到 `index.json.entries[]`
- 不把 `Finding NeMo / DP-LoRA / SecMI / TMIA-DM` 改写成 execution-ready
