# 2026-04-10 DP-LoRA Comparability Intake

## 目的

这份文档只回答一个问题：

- `DP-LoRA` 当前是否已经足够进入更深一层 `Phase E` intake review，还是仍应停留在 `comparability / intake hardening`。

它不是：

- run 授权
- GPU 申请单
- admitted defense upgrade
- `W-1` 替代宣告

## 当前固定前提

- `Current Mainline = PIA + provisional G-1(all_steps)`
- `Current Risk Evidence = recon DDIM public-100 step30`
- `Current Depth Line = GSA + W-1 strong-v3 full-scale`
- `white-box same-protocol bridge = closed-frozen`
- `Finding NeMo = zero-GPU hold`
- `PIA paper-aligned confirmation = document-layer conditional rank 1, execution-layer no-go`

## 当前定位

`DP-LoRA` 当前只能写成：

- `defended-track successor candidate`
- `W-2 style comparability / intake candidate`

不能写成：

- `W-1 replacement`
- `benchmark-ready`
- `execution-ready`
- `next GPU question released`

## 为什么现在值得推进

1. `Finding NeMo` 已固定为 `zero-GPU hold`，当前 intake review priority #2 候选需要更清晰的 comparability 定义。
2. `DP-LoRA` 的工程吸引力明确存在：它代表从全模型 DP 训练转向参数高效适配的轻量防御方向。
3. 当前最缺的不是“再找一个白盒防御名字”，而是写清它与 `W-1 = DPDM` 到底是否可比、如何可比、何时值得进入准入验证。

## Comparability Question

当前真正要回答的核心问题不是：

- `DP-LoRA` 好不好

而是：

- 它与当前 admitted 白盒线 `GSA + DDPM + CIFAR-10 + W-1 strong-v3 full-scale` 是否能建立足够可辩护的比较关系。

## Protocol Delta

当前已知的关键 protocol delta 至少包括：

1. `model family`
   - 当前 admitted 白盒线：`DDPM / CIFAR-10`
   - `DP-LoRA` 论文语境：`latent diffusion / LoRA adaptation`
2. `defense surface`
   - 当前 `W-1`：全模型 DP-style defended comparator
   - `DP-LoRA`：LoRA 参数高效适配上的隐私保护
3. `comparability target`
   - 当前 admitted 白盒线强调的是与 `GSA rerun1` 的近协议 defended comparator
   - `DP-LoRA` 更像下一代轻量 defended candidate，而不是同一条 defended rung 的直接续写

因此当前允许写成：

- `DP-LoRA is a successor defense candidate with partial protocol overlap, not a direct replacement for W-1.`

不允许写成：

- `DP-LoRA is already comparable to W-1 under the current admitted protocol.`

## Intake Checklist

在真正进入下一步之前，至少应补齐：

1. `paper surface memo`
   - 明确论文采用的模型族、数据域、LoRA 训练面、攻击面
2. `protocol comparability note`
   - 明确哪些维度可比较，哪些维度不可比较
3. `asset checklist`
   - 本地最小可验证资产需要什么
4. `expected artifact`
   - 将来若进入准入验证，想看到什么 artifact 才算值
5. `stop conditions`
   - 在什么情况下直接判 `not-yet / no-go`

## 最小资产清单

当前至少需要能写清：

1. 目标模型面：
   - 是否仍沿用当前 `DDPM/CIFAR-10`
   - 还是必须转向 latent diffusion 面
2. 训练接口面：
   - LoRA 适配训练需要的最小入口
3. defense evaluation 面：
   - 与当前 `W-1 strong-v3 full-scale` 比较时，采用什么统一指标和统一 threat model

## Expected Artifact

如果未来要进入下一步，当前最合理的预期产物不是 run，而是：

- `DP-LoRA comparability note`
- `asset checklist`
- `go / not-yet / no-go` intake verdict

## 当前正式裁决

- `decision_grade = decision-grade`
- `current_verdict = not-yet`
- `execution_release = none`
- `gpu_release = none`
- `queue_state = unchanged / not-requestable`

### 理由

1. 当前只知道它值得保留为候选，但 comparability 仍未写清。
2. 当前没有足够证据把它写成 `W-1` 的直接替代。
3. 当前也没有足够资产信息把它推进到执行态。

## 对队列的影响

当前这份文档只说明：

- `DP-LoRA` 是 intake review priority #2

它不说明：

- `DP-LoRA` 已获得 run release
- `DP-LoRA` 应抢在 `Finding NeMo` 前面
- `DP-LoRA` 已经值得占用 GPU
- `DP-LoRA` 已进入 released queue item

## 下一步

在这份 intake 文档固定后，下一步仍应继续保持：

- `Finding NeMo = zero-GPU hold`
- `DP-LoRA = comparability/intake hardening only`
- `gpu_release = none`
