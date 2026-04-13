# 2026-04-10 Finding NeMo Mechanism Intake

## 目的

这份文档用于把 `Finding NeMo + local memorization + FB-Mem` 从“高优先级候选”收口成一份 `decision-grade intake/eligibility artifact`。

它不是执行许可证，也不是新的 benchmark 结果。

固定字段：

- `owner`: `research_leader`
- `track`: `white-box`
- `candidate`: `Finding NeMo + local memorization + FB-Mem`
- `decision`: `intake-admit / execution-no-go`
- `current_rank`: `Phase E intake review priority #1`
- `depends_on`: `Phase D closed-frozen stays unchanged`
- `admitted_change`: `none`

当前允许写出的结论只有三类：

1. `eligible for one future separate release-review reconsideration`
2. `not-yet: intake incomplete`
3. `no-go: remain intake`

在这份 intake 未完成前：

- 当前 `active_gpu_question = none`
- 下一条 GPU 问题尚未 released
- 不得把 `Finding NeMo` 写成当前执行主线
- 不得把 `FB-Mem` 或 local memorization 写成已经解释了当前 `W-1` defended gap

## 当前状态快照

- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `current_risk_evidence`: `recon DDIM public-100 step30`
- `current_depth_line`: `GSA + W-1 strong-v3 full-scale`
- `bridge_status`: `white-box same-protocol bridge = closed-frozen`
- `active_gpu_question`: `none`
- `admitted_boundary`:
  - `GSA epoch300 rerun1` 仍是 admitted 白盒攻击主结果
  - `W-1 strong-v3 full-scale` 仍是 admitted defended 主 rung
- `diagnostic_boundary`:
  - `batch32 diagnostic comparator = runtime-smoke / diagnostic`
  - `admitted_change = none`

## 候选定位

`Finding NeMo + local memorization + FB-Mem` 当前之所以位于 intake review priority #1，不是因为它已经具备执行许可证，而是因为：

1. `PIA paper-aligned confirmation` 已因 provenance 长期 blocker 退回执行层 `no-go`
2. white-box bridge 已完成 `closed-frozen` 收口，不再占用主 GPU 注意力
3. 当前白盒最自然的下一问是：局部记忆信号是否可被定位，且是否与现有成员分离信号存在可复述关系
4. 如果 admitted 白盒资产与 `Finding NeMo` 的 `Stable Diffusion v1.4 / cross-attention value layers` 观测面不兼容，本轮应优先形成 `asset + protocol compatibility memo`，而不是强行开跑

这里的 `decision-grade` 只指 intake/eligibility 决策质量，不指 admitted 证据级别，不指 benchmark-ready，也不指 GPU 已放行。

## 参考材料

- 主论文：`Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models`
- 扩展解释层：
  - `Exploring Local Memorization in Diffusion Models via Bright Ending Attention`
  - `Demystifying Foreground-Background Memorization in Diffusion Models`
- 当前仓库内直接承接材料：
  - `docs/paper-reports/white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models-report.md`
  - `workspaces/white-box/signal-access-matrix.md`
  - `docs/claude-report-4-9-review.md`
  - `workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
  - `workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md`

## 当前待检验假设

当前只允许把下面内容写成 `hypothesis`，不能写成既成事实：

1. 在当前 admitted 白盒资产上，部分 member / non-member 分离信号可能伴随可复述的局部激活差异。
2. 这类局部差异未必需要一开始就进入 `cross-attention neuron intervention`，可以先在 `gradient + activations` 层面形成最小机制诊断。
3. `Finding NeMo` 提供的是定位框架，`local memorization / FB-Mem` 提供的是边界解释与升级方向；它们当前都不能被写成已经解释了 `W-1` 的 defended gap。

## Protocol Surface

当前建议按 6 个阶段理解这条候选，但本轮只允许推进到 `Stage 0-1` 的准入判断：

1. `Stage 0: protocol reconciliation`
   - 确认当前 admitted 白盒资产是否真能承接 `Stable Diffusion v1.4 / cross-attention` 观测面
   - 若不能，必须明确写成“观测面迁移”还是“结构性不兼容 no-go”
2. `Stage 1: observability smoke`
   - 用 `1 prompt x 1 seed x 1 layer` 证明最小 hook 链能稳定导出
3. `Stage 2: memorization intake`
   - 形成最小 `memorized / non-memorized` prompt 对照包
4. `Stage 3: localization probe`
   - 跑最小版 `NEMO` 初筛，输出候选神经元集合
5. `Stage 4: local-memory probe`
   - 检查是否存在区域级集中信号
6. `Stage 5: eligibility decision`
   - 只回答是否值得升级为正式 execution workspace

当前 round 的允许范围只覆盖：

- `Stage 0: protocol reconciliation`
- `Stage 1: observability smoke planning`

## Why Now

当前值得推进这条 intake，而不是继续围绕旧问题打转，原因只有三点：

1. `PIA provenance dossier` 已 closed 为 `remain long-term blocker`，`PIA paper-aligned confirmation` 当前不具备执行层放行条件。
2. white-box bridge 已 `closed-frozen`，再继续扩大不具备比机制 intake 更高的研究收益。
3. 这条线可以先落在非 benchmark、非 admitted、低成本的机制验证层，不会直接冲掉现有主线。

## 资产需求

进入任何 GPU validation 前，至少需要补齐以下资产或映射：

1. 一个固定的 admitted 白盒攻击资产根：
   - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
2. 一个固定的 defended 参考 rung：
   - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`
3. 样本绑定材料：
   - member / non-member 小样本清单
   - 对应 `sample_id` 规则
   - 与 admitted 协议一致的最小数据切片
4. 模型内部访问材料：
   - checkpoint root
   - 层级命名映射
   - 最小 hook 入口
5. 结果 schema：
   - `sample_id`
   - `timestep`
   - `loss`
   - `grad_norm`
   - `layer_id`
   - `neuron_id`
6. 如果后续要纳入 `local memorization / FB-Mem` 升级项，还需要：
   - 区域级可视化或 mask/saliency 产物接口
   - 前景/背景局部证据导出接口

## 首批允许的 Hook 点

首批信号访问只允许落在：

1. `gradient`
2. `activations`

当前明确不把 `cross-attention` 作为首跑硬前置。

原因：

1. 当前仓库的白盒信号访问矩阵已经明确建议首批实现先围绕 `gradient + activations`
2. 如果一开始就把 `cross-attention neuron intervention` 当成准入硬前置，会把 intake 直接推成重型机制线
3. `Finding NeMo` 的 value-layer 干预是后续升级项，不是当前最小验证形状

## 最小允许首跑形状

如果未来经过单独 release review 重审，当前最多只允许保留一轮最小 `validation-smoke` 作为上限设想：

1. 固定单一 admitted 白盒攻击资产和单一已冻结 defended rung
2. 只验证一个最小 hook 链是否可用：
   - 优先 `activations`
   - 补一个 `gradient/grad_norm` 基线
3. 只抽取一小批 member / non-member 样本
4. 只生成样本级机制记录
5. 不做 `cross-attention` 干预
6. 不做额外训练
7. 不做 full-dataset sweep
8. 不做 benchmark 冲榜

## Compute Budget 与 Stop Conditions

没有预声明预算、停止条件和期望产物，不得申请 GPU。

当前建议固定为：

1. `budget`:
   - `1 次单卡短时 validation-smoke`
   - 只覆盖 `1 个 hook 路径 + 1 组小样本 + 1 次可复述输出`
2. `stop_conditions`:
   - 若最小预算内拿不到稳定样本级 hook 输出，则立即停止并判定 `intake incomplete`
   - 若输出只能证明“代码能跑”，但不能回答“局部记忆是否可定位、是否与当前成员分离信号一致”，则停止，不升级
   - 若需要引入 `cross-attention` 干预、额外训练、协议改写或第二轮 GPU 才能解释结果，则停止并回到 intake 补件
   - 若结果与 admitted / diagnostic 边界冲突，则先回文档裁决，不得继续烧 GPU

## Expected Artifact

当前允许承诺的产物只有 intake/eligibility 层：

1. 一份 `white-box mechanism intake note`
2. 一份 `asset + protocol compatibility memo`
3. 一份最小 future release-review packet
4. 一条明确裁决：
   - `eligible for one future separate release-review reconsideration`
   - `remain intake / not-yet`
   - `no-go`

当前不允许把下面内容写成已交付产物：

1. 已成立的机制证据
2. 已完成的局部记忆诊断报告
3. 已定位的 memorization neurons
4. 已解释的 `W-1` defended gap

## Eligibility Gate

`Finding NeMo + local memorization + FB-Mem` 只有在以下条件同时满足后，才允许从候选升级为“可进入一次未来单独 release review 的重审范围”：

1. `Phase D` 已按 `closed-frozen` 收口，bridge 不再占用主 GPU 注意力
2. `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 admitted 口径继续冻结
3. `batch32 diagnostic comparator` 继续保持 `runtime-smoke / diagnostic`
4. intake 文档已补齐：
   - `hook 点`
   - `资产需求`
   - `局部记忆诊断目标`
   - `退出条件`
5. 运行入口 portable，且不依赖 `LocalOps/paper-resource-scheduler`
6. compute budget / stop conditions / expected artifact 已提前写死
7. 目标被定义为 `new-question intake validation`，而不是 bridge 变相续跑

## Reconsider / No-Go / Not-Yet 规则

### Reconsider

只允许放行到：

- `one future separate release-review reconsideration`

不是：

- 长 GPU run
- 新 benchmark
- 新 admitted 结果

### Not-Yet

出现下面任一情况就维持 `not-yet`：

1. hook 层级命名仍未固定
2. member / non-member 小样本绑定还不稳定
3. portable 入口还依赖 scheduler
4. 输出 schema 还不足以复审

### No-Go

出现下面任一情况就维持 `no-go`：

1. 首跑必须依赖 `cross-attention` 干预或新增训练才能成立
2. 机制问题与当前 admitted 协议没有稳定绑定
3. 该路线需要重写主协议才能产出可解释结果

## 明确不做

1. 不把 `Finding NeMo` 写成当前执行主线
2. 不把这份 intake 写成 benchmark-ready 或 execution-ready
3. 不把 `FB-Mem` 写成已经验证的原因解释
4. 不把任何 diagnostic、candidate 或 bridge 状态升级成 admitted 主结果
5. 不在没有新的 decisive artifact 前重开 white-box bridge
6. 不在 intake 未闭环时并行第二条白盒长 GPU 线

## 当前裁决

- `decision_date`: `2026-04-10`
- `decision_status`: `review-ready`
- `candidate_state`: `intake/eligibility only`
- `gpu_release`: `none`
- `next_required_proof`: `portable observability smoke + asset/protocol reconciliation`
- `best_next_move`: `finish intake completeness and hold for boss review`

当前结论固定为：

`Finding NeMo + local memorization + FB-Mem` 现在只获得了“未来单独 release review 可重审一次最小 validation-smoke 上限”的定义，还没有获得任何新长 GPU run 的授权。
