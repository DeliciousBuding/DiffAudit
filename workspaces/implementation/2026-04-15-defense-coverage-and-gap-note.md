# 2026-04-15 Defense Coverage And Gap Note

## 为什么要有这张说明

当前 `attack-defense matrix` 里有几处 `not yet run`。如果不解释，评委很容易把它理解成“这部分还没做完”。这张说明的作用，就是把这些空位解释成“当前比较覆盖范围”，而不是“隐藏的未完成工作”。

## 当前已经纳入主讲的 defense coverage

### Gray-box defended headline

- 主线：`PIA 1024 / 1024`
- defended comparator：stochastic dropout (`all_steps`)
- 结论：`0.83863 -> 0.825966`
- 作用：这是当前最清楚、最适合公开主讲的灰盒攻防对比

### White-box defended comparator

- 主线：`GSA 0.998192`
- defended comparator：`DPDM W-1 0.488783`
- 作用：展示特权访问上界与防御后对照之间的强反差

## 为什么有些方法没有 defended row

### `Recon`

- 当前角色：admitted black-box headline
- 原因：当前最终包里没有一条同口径、同资产、已 admitted 的 black-box defense ladder

### `CLiD`

- 当前角色：black-box corroboration
- 原因：当前 `CLiD` 承担独立 scorer corroboration，不承担 defense benchmark 角色

### `SecMI`

- 当前角色：gray-box alternate scorer corroboration
- 原因：当前 final package 里没有匹配的 defended `SecMI` rung；它的职责是证明灰盒信号不依赖单一 scorer

## 如果评委问“是不是还没做完”

建议回答：

不是。这里的 `not yet run` 指的是“当前最终答辩包没有纳入一条同口径 defended comparator”，不是说主线研究还停在未完成状态。当前 defense 主讲线已经固定为 `PIA + stochastic dropout`，其余方法在最终包里的职责是 headline、corroboration 或 upper bound。

## 如果评委问“为什么不把所有方法都补防御”

建议回答：

因为答辩主讲需要的是清晰、可复述、边界明确的证据，而不是把每个方法都机械补成不完全可比的矩阵。宁可把 defense coverage 说清楚，也不把口径不齐的比较强行塞进主叙事。

## 一句话版本

当前 defense 证据不是“每条方法都补一遍”，而是“选择最清晰的可比 defended comparator 作为主讲线，其余方法承担 corroboration 或 upper-bound 角色”。
