# 2026-04-10 Phase E Candidate Surface Remaining Sweep

## Status Panel

- `owner`: `research_leader`
- `artifact_type`: `decision-grade remaining-surface sweep`
- `decision_scope`: `writing-only, non-GPU`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `active_gpu_question`: `none`
- `bridge_status`: `closed-frozen`

## 目的

这份文档只回答一个问题：

- 在 round-19 完成 `phase-e-candidates.json` 的 semantic downgrade 之后，仓库内是否还存在“剩余未审计文档”把 candidate surface 重新讲成 release / queue-control surface。

它不是：

- run 授权
- GPU 申请单
- admitted 升级文档
- 新执行线启动公告

## 审计范围

本轮额外检查了：

1. `Research/docs/local-api.md`
2. `Research/docs/mia-defense-research-index.md`
3. `Research/docs` 与 `Research/workspaces` 内所有含下列关键词的文档命中：
   - `phase-e-candidates`
   - `execution_layer_default_order`
   - `intake_review_priority_order`
   - `execution_release`
   - `gpu_release`
   - `queue_state`
   - `run release`
   - `execution-ready`
   - `benchmark-ready`

同时复核了：

- `ROADMAP.md`
- `Agents/GLOBAL_TASK_BOARD.md`
- `Research/ROADMAP.md`
- `Research/docs/comprehensive-progress.md`
- `Research/docs/reproduction-status.md`
- `Research/docs/mainline-narrative.md`
- `Research/docs/future-phase-e-intake.md`
- `Research/workspaces/intake/phase-e-candidates.json`

## 当前硬边界

当前必须持续保持：

- `Research/workspaces/intake/index.json.entries[] = promoted / system-intake-ready contracts only`
- `Research/workspaces/intake/phase-e-candidates.json = candidate-only intake-review snapshot`
- `phase-e-candidates.json` 不构成：
  - `Local-API` contract surface
  - promoted contract surface
  - release-control / queue-control table
  - admitted / benchmark registry

## 剩余命中分类

### A. 已清理的高层 candidate surface

以下高层文档当前未再把 candidate surface 讲成 release / queue-control 面：

- `ROADMAP.md`
- `Agents/GLOBAL_TASK_BOARD.md`
- `Research/ROADMAP.md`
- `Research/docs/comprehensive-progress.md`
- `Research/docs/reproduction-status.md`
- `Research/docs/mainline-narrative.md`
- `Research/docs/future-phase-e-intake.md`
- `Research/docs/local-api.md`
- `Research/docs/mia-defense-research-index.md`

### B. 仍包含 release / queue 词汇，但属于局部 decision/intake 文档

剩余命中主要集中在以下文档：

- `Research/workspaces/intake/2026-04-10-dplora-comparability-intake.md`
- `Research/workspaces/gray-box/2026-04-10-secmi-unblock-decision.md`
- `Research/workspaces/gray-box/2026-04-10-tmia-dm-intake-decomposition.md`
- `Research/workspaces/white-box/2026-04-10-finding-nemo-*`
- `Research/workspaces/black-box/2026-04-10-recon-explanation-layer.md`

这些命中的共同特征是：

1. 它们描述的是各条线自己的 `no release / no go / blocked / hold` 边界
2. 它们没有把 `phase-e-candidates.json` 当作 `Local-API` contract surface
3. 它们没有把 candidate metadata 回流到 `index.json.entries[]`

因此，这些命中当前应归类为：

- `line-local boundary wording`

而不是：

- `candidate-surface relapse`

## 正式裁决

- `decision_grade = decision-grade`
- `remaining_sweep_verdict = clean`
- `high_level_relapse = none found`
- `line_local_release_wording = acceptable when scoped to local decision docs`
- `gpu_impact = none`

## 为什么这轮值得做

1. round-19 已完成 semantic downgrade，但如果不做一次 repo-wide remaining sweep，后续仍很难区分“局部 decision 文档中的 release 词汇”和“高层 candidate surface 复燃”。
2. 当前没有 active 主 GPU 问题，本轮最值得推进的是把“还有没有遗漏回流点”做成可审查结论，而不是继续扩写新候选。
3. 这份文档把下一轮注意力从“继续反复查已清理的高层文档”切换到“只盯未来新增文档是否再次引入 release-like wording”。

## 下一次必须重开 sweep 的触发条件

以下任一条件出现，就必须再做一次 remaining sweep：

1. 新增任何引用 `phase-e-candidates.json` 的高层文档
2. `Research/docs/local-api.md` 或其他系统读链文档开始提到 `Phase E` candidate ordering
3. `phase-e-candidates.json` 再次新增 release / queue-like 字段
4. 任一候选从 `not-yet / blocked / hold` 进入 release review

## 当前明确不做

- 不改变 admitted hierarchy
- 不释放任何 GPU
- 不新增任何 `Phase E` 候选记录
- 不修改各条线已有 local decision docs 的 `no release / hold / blocked` 语句
- 不把 line-local release wording 错当成 candidate-surface relapse
