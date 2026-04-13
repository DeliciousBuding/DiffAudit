# 2026-04-10 PIA 8GB Supporting Frontier Note

## Status Panel

- `owner`: `research_leader`
- `track`: `gray-box`
- `artifact_type`: `decision-grade supporting frontier note`
- `execution_scope`: `release-free current-mainline-support`
- `hardware`: `RTX4070 laptop 8GB`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `provenance_status`: `workspace-verified`
- `admission_effect`: `none`
- `updated_at`: `2026-04-10 21:36 +08:00`

## 问题

在 `RTX4070 laptop 8GB` 现实下，当前 `PIA + stochastic-dropout(all_steps)` 是否还能以 bounded GPU rung 保持同向 `adaptive` 指标下降，并形成可复述的 portability / cost frontier。

## Probe / Preview

- `runtime-probe-pia --device cuda:0`：`ready`
- `runtime-preview-pia --device cuda:0 --preview-batch-size 2`：`ready`

这说明：

- 真资产路径仍健康
- `cuda:0` 上的 member/non-member preview forward 正常
- 本轮可以进入 bounded supporting runs

## Rung Results

| rung | baseline adaptive AUC | defense adaptive AUC | baseline adaptive ASR | defense adaptive ASR | baseline wall-clock(s) | defense wall-clock(s) |
| --- | --- | --- | --- | --- | --- | --- |
| `GPU128` | `0.817444` | `0.806274` | `0.765625` | `0.761719` | `49.544877` | `50.667101` |
| `GPU256` | `0.841293` | `0.829559` | `0.78125` | `0.763672` | `95.934721` | `213.126361` |

对应 artifact：

- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-256-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-256-allsteps-adaptive/summary.json`

## 观察

### 1. `GPU128` 是最干净的便携 supporting pair

- 两边 wall-clock 基本对称
- `all_steps` 继续压低 `adaptive AUC / ASR`
- 这是当前最适合在 8GB 机器上快速复述的 bounded supporting pair

### 2. `GPU256` 仍是有效 decision rung，但 cost 已显著变差

- `all_steps` 仍然压低 `adaptive AUC / ASR`
- 但 defense 侧 wall-clock 上升到 `213.126361s`
- 相比 baseline 的 `95.934721s`，成本比约为 `2.22x`

这意味着：

- `GPU256` 仍值得保留为 decision rung
- 但它已经不适合作为默认快速复核档

### 3. baseline adaptive path 仍是 deterministic anchor

- baseline 在 `adaptive_query_repeats = 3` 下仍给出：
  - `member_std = 0.0`
  - `nonmember_std = 0.0`
- defense 则出现非零 `score_std`

这与当前机理叙事一致：

- baseline 维持稳定的 `epsilon-trajectory consistency`
- `all_steps` defense 在 repeated-query 下引入额外不稳定性

## Round-26 Decision

当前最合理的收口不是继续冲 `GPU512`，而是：

1. `retain GPU128 as the quickest portable pair`
2. `retain GPU256 as the decision rung with cost warning`
3. `stop before any GPU512 rerun`

理由：

1. 当前 portability 问题已经被 `GPU128 + GPU256` 回答
2. `GPU256 defense` 成本已经超过旧 `GPU256` 非 adaptive defense 的 `2x` 阈值
3. 再上 `GPU512` 只是在重复支持旧上限，而不是回答新问题

## Allowed Claim

当前只允许写：

- `8GB portability/supporting evidence confirmed at GPU128 and GPU256`
- `all_steps keeps the same privacy-drop direction under bounded 8GB runs`
- `GPU128 is the cleanest quick portable pair`
- `GPU256 remains useful but carries a cost warning`

当前不允许写：

- `new admitted gray-box mainline`
- `paper-aligned confirmation`
- `recommended canonical defense`
- `validated privacy win`

## Next Step

1. 把本 note 写回 `reproduction-status` 与灰盒计划
2. 保持 `PIA` strongest claim 不变
3. 若未来继续做 `PIA defense-cost frontier`，优先从 `GPU128` 级 quick pair 出发，而不是直接重开 `GPU512`
