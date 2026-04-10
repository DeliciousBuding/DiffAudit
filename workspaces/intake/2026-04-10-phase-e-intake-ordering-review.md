# 2026-04-10 Phase E Intake Ordering Review

## 目的

这份文档只回答一个问题：

- 在 `Phase D = closed-frozen` 且当前没有 active 主 GPU 问题的前提下，`Phase E` 候选池的排序应如何被正式固定，避免文档层条件排序与执行层默认放行顺序继续混写。

它不是：

- GPU 申请单
- run 授权
- admitted 升级文档
- 新研究线启动公告

## 当前固定前提

- `Current Mainline = PIA + provisional G-1(all_steps)`
- `Current Risk Evidence = recon DDIM public-100 step30`
- `Current Depth Line = GSA + W-1 strong-v3 full-scale`
- `white-box same-protocol bridge = closed-frozen`
- `active_gpu_question = none`
- `PIA provenance dossier = remain long-term blocker`
- `Finding NeMo activation-export adapter = zero-GPU hold`

## 决策结构

当前必须把 `Phase E` 排序拆成两层，而不是继续写成单一列表：

### 1. 文档层条件排序

这层回答：

- 如果 blocker 被解除，哪条线在论文叙事或主讲线升级上条件性优先。

当前固定为：

1. `PIA paper-aligned confirmation`

原因：

- 它不是“新题”，而是当前主讲线的升级确认。
- 一旦 `checkpoint/source provenance` 被真正解除，它的叙事收益高于任何新候选。

但当前同时固定：

- `execution-layer status = no-go`
- 在 provenance 条件未变化前，它不得进入执行层。

### 2. 执行层默认放行顺序

这层回答：

- 在当前 blocker 现实下，未来真正进入准入验证时，候选默认应按什么顺序被看。

当前正式固定为：

1. `Finding NeMo + local memorization + FB-Mem`
2. `DP-LoRA`
3. `SecMI unblock`
4. `TMIA-DM intake`

## 各候选裁决

### A. Finding NeMo + local memorization + FB-Mem

- `current_status = not-yet`
- `current_shape = adapter-complete zero-GPU hold`
- `gpu_release = none`
- `queue_state = not-requestable`

保留在执行层第 1 顺位的原因：

1. intake 已最完整
2. contract / sketch / adapter / hold review 已形成闭环
3. 一旦未来要重审一次最小 smoke，这条线的准备度最高

当前不允许的误写：

- execution-ready
- benchmark-ready
- mechanism evidence ready
- new GPU question released

### B. DP-LoRA

- `current_status = research-ready candidate`
- `current_shape = defended-track successor intake`

保留在执行层第 2 顺位的原因：

1. 与 `W-1 = DPDM` 的 defended 叙事天然相邻
2. 其价值主要在未来 defended comparator 扩展，而不是当前主讲线升级
3. 当前仍缺一份更清晰的 comparability / protocol-delta 收口文档

### C. SecMI unblock

- `current_status = blocked baseline`
- `current_shape = asset-unblock dependent`

保留在执行层第 3 顺位的原因：

1. 价值首先是“恢复 baseline 可执行”，不是立即产生新主讲线
2. 当前仍缺真实 `flagfile + checkpoint root`
3. 在资产未到位前，不值得抢前两位注意力

### D. TMIA-DM intake

- `current_status = literature-stage intake`
- `current_shape = protocol-and-asset decomposition pending`

保留在执行层第 4 顺位的原因：

1. 当前仍主要是文献轴候选，不是已收口的执行候选
2. 需要额外拆协议、资产和最小 smoke 入口
3. 当前解释收益高于短期执行收益

## 正式推荐结论

- `decision_grade = decision-grade`
- `document_layer_conditional_rank_1 = PIA paper-aligned confirmation`
- `execution_layer_default_order = Finding NeMo > DP-LoRA > SecMI unblock > TMIA-DM intake`
- `current_execution_release = none`

## 为什么现在值得写这份文档

1. `Finding NeMo` 已被固定为 `zero-GPU hold`，不能再靠“它是第一候选”这句模糊话术维持秩序。
2. `PIA paper-aligned confirmation` 继续保留文档层第 1 顺位，但若不明确说明 `execution-layer no-go`，后续极易漂移。
3. `DP-LoRA / SecMI / TMIA-DM` 需要被明确写成后续候选，而不是被动堆在列表尾部。

## 当前明确不做

- 不改变 admitted hierarchy
- 不释放任何 GPU
- 不把任何候选写成已进入执行态
- 不把 explanation / boundary / intake 写成 benchmark
- 不把 `Finding NeMo zero-GPU hold` 偷换成 smoke ready

## 下一步

在这份排序评审写回后，下一条非 GPU 槽位建议固定为：

- `intake registry / candidate indexing` 的结构边界判断与 machine-readable 写回

而不是：

- 新黑盒 run
- `Finding NeMo` release review
- `PIA paper-aligned` 提前放行
