# Future Phase E Intake

这份文档用于固定 `Phase E` 候选池的排序、进入条件、退出条件与预期产物。

它不是执行许可证。

在 `Phase D: Same-Protocol Benchmark Bridge` 正式收口前，任何候选都只能停留在 intake，不进入执行态。

当前配套排序评审见：

- `workspaces/intake/2026-04-10-phase-e-intake-ordering-review.md`
- `workspaces/intake/2026-04-10-intake-registry-phase-e-boundary-review.md`

当前 machine-readable 候选镜像见：

- `workspaces/intake/phase-e-candidates.json`

边界约束：

- `workspaces/intake/index.json.entries[]` 继续只承载 `promoted / system-intake-ready` contracts
- `phase-e-candidates.json` 只承载 `Phase E` 候选排序与状态快照
- `phase-e-candidates.json` 不是 run release，也不是 admitted 升级入口

## 当前固定排序

1. `PIA paper-aligned confirmation`
2. `Finding NeMo + local memorization + FB-Mem`
3. `DP-LoRA`
4. `SecMI unblock`
5. `TMIA-DM intake`

## 当前正式解释

上面的单一列表只保留文档层可读性；真正执行时，必须按两层理解：

1. `PIA paper-aligned confirmation`
   - 文档层条件性第 1 顺位
   - 当前执行层 `no-go`
2. `Finding NeMo + local memorization + FB-Mem`
3. `DP-LoRA`
4. `SecMI unblock`
5. `TMIA-DM intake`

其中 2-5 才构成当前准入验证优先顺序。

## 1. PIA paper-aligned confirmation

### 当前定位

如果 `PIA provenance` 被解除，这不是“新题”，而是当前主讲线的升级确认。

### 进入条件

- `Phase D` 已正式关闭
- `checkpoint/source provenance` 已被单独核准
- 有明确 delta hypothesis
- 目标是确认 `paper-aligned`，而不是重复已冻结结论

### 退出条件

- 形成 `paper-aligned confirm / no-confirm` 的正式裁决
- 不允许无限延长成“再跑一轮看看”

### 预期产物

- 一份 `paper-aligned` 确认结论，或
- 一份明确 `no-go` 决策

## 2. Finding NeMo + local memorization + FB-Mem

### 当前定位

这是白盒最自然的下一问，但当前只允许把它推进成 `decision-grade intake/eligibility artifact`。

它当前回答的问题不是“已经证明了什么机制”，而是：

- 是否值得申请一次最小 `validation-smoke`
- 当前最小 hook 面、资产需求和退出条件是否已经足够清楚

详细 intake 见：

- `workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md`
- `workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
- `workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md`
- `workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`

当前固定结论补充：

- `activation export adapter = implemented`
- `current verdict = zero-GPU hold`
- `current boundary = non-GPU only; separate release review required before any validation-smoke discussion`
- 这仍不是 `go`

### 进入条件

- `Phase D` 已收口
- 当前 `GSA / W-1` 协议面不再漂移
- 已有完整 intake，至少包括：
  - hook 点
  - 资产需求
  - 局部记忆诊断目标
  - 退出条件
- 已预声明：
  - compute budget
  - stop conditions
  - expected artifact
- 首批信号访问限定在 `gradient + activations`
- `cross-attention` 不作为首跑硬前置

### 退出条件

- 形成一次 intake 级 `go / not-yet / no-go` 裁决
- 若 `go`，最多只允许申请一次最小 `validation-smoke`
- 若 `not-yet / no-go`，则继续停留在 intake，不进入执行态

当前已落地裁决：

- `not-yet`
- 具体形态：`adapter-complete zero-GPU hold`

### 预期产物

- white-box mechanism intake note
- one minimal `validation-smoke` plan
- `go / not-yet / no-go` 裁决

## 3. DP-LoRA

### 当前定位

这是下一代轻量防御候选，不替换当前 `W-1 = DPDM`。

当前 intake 评审见：

- `workspaces/intake/2026-04-10-dplora-comparability-intake.md`

当前固定结论补充：

- `current_verdict = not-yet`
- `current boundary = comparability-note only; no release review until protocol overlap is decision-grade`
- 当前只允许推进 `comparability / intake hardening`

### 进入条件

- `W-1` 当前 defended rung 已稳定
- admitted 主口径不再改动
- 已能定义与 `GSA + DDPM + CIFAR-10` 足够接近的比较协议

### 退出条件

- 形成“是否能成为可比较 defended candidate”的 yes/no 决策

当前已落地裁决：

- `not-yet`
- 具体形态：`comparability / intake hardening only`

### 预期产物

- `W-2` 级候选 intake
- comparability note
- 是否值得进入 admitted 备选表的判断

## 4. SecMI unblock

### 当前定位

它不是当前主讲线，也不是 bridge 后默认第一优先。

当前 unblock 决策见：

- `workspaces/gray-box/2026-04-10-secmi-unblock-decision.md`

当前固定结论补充：

- `current_verdict = not-yet`
- `current_state = blocked baseline`
- `current boundary = asset-blocked; no release review until real flagfile + checkpoint root arrive`

### 进入条件

- 真实 `flagfile + checkpoint root` 已到位
- 已完成 probe / dry-run
- 能证明这是解除 baseline 阻塞，而不是在资产未齐时烧 GPU

### 退出条件

- 恢复为可执行 baseline，或
- 正式继续维持 `blocked baseline`

当前已落地裁决：

- `not-yet`
- 具体形态：`remain blocked baseline until real flagfile + checkpoint root arrive`

### 预期产物

- unblock memo
- 最小可执行入口
- yes/no 执行裁决

## 5. TMIA-DM intake

### 当前定位

当前只作为灰盒候选池文献，不进入黑盒执行面。

当前 intake 拆解见：

- `workspaces/gray-box/2026-04-10-tmia-dm-intake-decomposition.md`

当前固定结论补充：

- `current_verdict = not-yet`
- `current_shape = protocol-and-asset decomposition intake`
- `current boundary = paper-and-asset decomposition only; no release review until a minimal executable path exists`

### 进入条件

- 已完成协议拆解
- 已形成资产清单
- 已定义最小 smoke 入口与退出条件

### 退出条件

- 给出是否值得进入执行面的排序判断

当前已落地裁决：

- `not-yet`
- 具体形态：`gray-box protocol / asset decomposition intake only`

### 预期产物

- intake 文档
- 资产需求清单
- 与 `PIA` 的比较定位

## 当前默认规则

- 只要 `PIA provenance` 在窗口内解除，`PIA paper-aligned confirmation` 就保持第一优先
- 当前 `PIA provenance dossier` 已 closed 为 `remain long-term blocker`，因此执行层默认优先推动 `Finding NeMo + local memorization + FB-Mem` 从 intake 升级
- 这里的 `decision-grade` 只指 intake/eligibility 决策质量，不指 admitted 升级，不指 benchmark-ready，也不指 GPU 已放行
- `DP-LoRA` 不抢第一条新问题
- `SecMI` 与 `TMIA-DM` 继续后排
- `Finding NeMo` 当前虽然位于准入验证优先第 1 位，但状态已固定为 `zero-GPU hold`
- `Finding NeMo` 当前虽然位于准入验证优先第 1 位，但状态已固定为 `zero-GPU hold`

## 当前准入验证优先顺序

在真正进入下一条 GPU 题的准入验证时，当前候选审查顺序为：

1. `Finding NeMo + local memorization + FB-Mem`
2. `DP-LoRA`
3. `SecMI unblock`
4. `TMIA-DM intake`

补充说明：

- `PIA paper-aligned confirmation` 继续保留文档层条件性第 1 顺位
- 但在 provenance 条件未变化前，不进入执行层
- `Finding NeMo + local memorization + FB-Mem` 当前只是准入验证优先第 1 候选，不构成当前 GPU release
- 这份“准入验证优先顺序”不等于现在立刻执行；它只定义未来 intake review 的顺序
- 这份“准入验证优先顺序”不等于现在立刻执行；它只定义未来 intake review 的顺序

## 当前明确不做

- 不在 `Phase D` 未关闭时放行任何候选进入主 GPU 队列
- 不把 intake 文档直接写成执行完成
- 不把 `On the Edge` 和 `MIDST` 并入执行候选
