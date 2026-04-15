# 2026-04-15 Metric Glossary And Claim-Boundary Card

## 这页用来解决什么问题

评委经常会把指标词和结论词混在一起听，比如把 `AUC` 理解成“实际攻击成功率”，或者把 `admitted` 理解成“绝对最强结果”。这页的用途，就是把这些高频误读提前拆开。

## 核心指标怎么解释

### `AUC`

- 含义：成员和非成员分数分离能力
- 直觉：越接近 `1.0`，说明排序区分越强
- 不要说成：真实世界必然攻击成功率

### `ASR`

- 含义：在当前阈值设定下的攻击成功比例
- 直觉：反映当前 scoring rule 下的整体判别效果
- 不要说成：任何部署环境下都能稳定达到的固定命中率

### `TPR@1% FPR`

- 含义：把误报率压到很低时，还能保留多少真阳性
- 直觉：更接近评委关心的“严格误报约束下还剩多少攻击力”
- 不要说成：完整产品环境的最终业务指标

## 常见结论词怎么解释

### `admitted`

- 含义：当前最适合公开主讲、边界最清楚、证据链最稳的一条结果
- 不等于：数值绝对最高

### `candidate`

- 含义：结果成立，但主讲时还需要更谨慎表述，或更适合作 corroboration
- 不等于：结果无效

### `corroboration`

- 含义：独立方法或独立 scorer 对主线结果的交叉支持
- 不等于：替代主线叙事

### `upper bound`

- 含义：在更强权限假设下观察到的风险上界
- 不等于：普通用户场景下默认就有这么高的风险

## 本项目里这几个词分别对应什么

### `Recon`

- 角色：admitted black-box headline
- 该怎么说：黑盒已能观察到 membership signal

### `CLiD`

- 角色：black-box corroboration
- 该怎么说：workspace-verified local corroboration
- 不该怎么说：full paper-faithful benchmark

### `PIA`

- 角色：gray-box main controlled line
- 该怎么说：有限模型侧访问下，信号在 scale-up 后仍稳定

### `SecMI`

- 角色：gray-box alternate scorer corroboration
- 该怎么说：灰盒信号不依赖单一攻击目标函数

### `GSA`

- 角色：white-box privileged upper bound
- 该怎么说：权限提升后，membership risk 接近饱和
- 不该怎么说：普通产品 KPI

## 4 个最容易被误读的点

1. `AUC` 高，不等于“线上一定会按同样比例被打穿”。
2. `admitted` 不等于“数值最强”，而是“最适合公开主讲”。
3. `CLiD` 现在是 corroboration，不是 paper-faithful benchmark claim。
4. `GSA` 是上界，不是默认产品场景风险值。

## 如果评委问“你们到底能不能落到一句话”

可以：扩散模型的 membership leakage 跨权限层级持续存在，而且当前轻量防御还不足以消除它。

## 如果评委问“为什么你们老在讲边界”

因为我们的目标不是夸大攻击，而是给出可审计、可复述、可防御讨论的研究证据。边界说清楚，结论才站得住。
