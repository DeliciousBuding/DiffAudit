# 2026-04-10 Phase E Finding NeMo Intake Hold Decision

## 目的

这份文档只做一件事：

- 把 `Finding NeMo + local memorization + FB-Mem` 正式收口成一份 `Phase E intake governance` 决策。

它不是：

- run 授权
- GPU 申请单
- release review
- benchmark 结果
- admitted 升级

## Status Panel

- `selected_candidate`: `Finding NeMo + local memorization + FB-Mem`
- `decision_grade`: `decision-grade`
- `current_verdict`: `not-yet`
- `current_shape`: `adapter-complete zero-GPU hold`
- `gpu_release`: `none`
- `queue_state`: `not-requestable`
- `admitted_change`: `none`
- `bridge_change`: `none`
- `next_reconsideration_gate`: `separate hypothesis/budget review only`

## 当前固定前提

- `Current Mainline = PIA + provisional G-1(all_steps)`
- `Current Risk Evidence = recon DDIM public-100 step30`
- `Current Depth Line = GSA + W-1 strong-v3 full-scale`
- `PIA provenance dossier = remain long-term blocker`
- `white-box same-protocol bridge = closed-frozen`
- `active_gpu_question = none`

## 汇总输入

当前这份 hold decision 只汇总以下既有工件，不新增实验事实：

- `workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md`
- `workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
- `workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md`
- `workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`

## 为什么当前值得收口成 hold decision

1. 它已经是当前 `Phase E` 准入验证优先第 1 位，但还没有一份 boss-level 的单一决策工件把“最佳候选但继续 hold”钉死。
2. 当前技术边界已经清楚：
   - paper-faithful 原始 `Stable Diffusion v1.4 / cross-attention value layers` 路线不适配当前 admitted 白盒资产
   - 当前只剩 migrated DDPM observability route 的受限 code path
3. 当前实现状态已经足够支撑决策，但仍不足以支撑放行：
   - `activation export adapter = implemented`
   - `gpu_release = none`
   - `queue_state = not-requestable`

## 当前正式裁决

当前正式结论固定为：

- `Finding NeMo + local memorization + FB-Mem` 继续保留为当前最完整的 `Phase E` intake dossier
- 但它仍只是 `adapter-complete zero-GPU hold`
- 它不构成当前 queue item
- 它不构成当前或近期待申请的 `validation-smoke`
- 它不构成新的 active GPU question

## 为什么当前仍是 not-yet

1. 当前 admitted 白盒资产与原始 `Finding NeMo` 论文协议面仍结构性不兼容。
2. 当前 adapter code path 只证明：
   - `read-only contract-probe implemented`
   - `cpu-only activation-export adapter implemented`
   - `code path available`
3. 当前并没有新的 hypothesis/budget review 去说明为什么值得占用一次未来注意力。
4. 任何把当前状态改写成“下一步就该申请最小 smoke”的表述，都会制造 queue/release 漂移。

## 未来唯一允许的重审入口

只有在下面条件同时满足后，才允许进入一次未来单独的重审：

1. 有单独的 `hypothesis / budget review`
2. 该 review 写清：
   - hypothesis
   - compute budget
   - expected artifact
   - stop conditions
3. 该 review 明确保持：
   - `gpu_release = none` 直到 review 通过
   - `queue_state = not-requestable` 直到单独裁决完成
   - 不触发新的 benchmark / admitted 叙事

## 当前明确不做

- 不把这份 hold decision 写成 release review
- 不把 `Finding NeMo` 写成当前最小 smoke 可申请项
- 不把 `Finding NeMo` 写成 execution-ready
- 不把 `Finding NeMo` 写成 benchmark-ready
- 不把当前 adapter code path 写成新 GPU 问题
- 不把这条线写成 `bridge` 的续跑

## 输出影响

这份文档只改变一个层面的治理清晰度：

- `Finding NeMo` 现在拥有单一的 `intake-hold decision` 锚点

它不改变：

- admitted hierarchy
- mainline
- bridge decision
- GPU queue
- `index.json.entries[]`
