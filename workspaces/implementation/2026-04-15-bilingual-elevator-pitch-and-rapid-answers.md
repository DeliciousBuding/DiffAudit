# 2026-04-15 Bilingual Elevator Pitch And Rapid Answers

## 用途

这份材料给两种场景直接使用：

1. 评委只给 `30-90` 秒发言时间
2. 评委临时切到英文或要求一句话总结

## 30 秒中文版本

DiffAudit 审计的是扩散模型的成员推断隐私风险。我们的核心发现是，这种泄漏并不只存在于一种攻击设定里，而是会随着攻击者权限从 black-box、gray-box 到 white-box 逐步增强；同时，轻量防御只能削弱信号，还不能真正消除泄漏。

## 60 秒中文版本

DiffAudit 不是只展示一次攻击成功，而是构建了一条完整的审计链。黑盒下，`Recon` 已经做到 `AUC 0.849`；灰盒下，`PIA` 在 `1024 / 1024` 规模上仍有 `AUC 0.83863`，加 stochastic dropout 后还有 `0.825966`；白盒上界 `GSA` 接近 `0.998192`。这说明扩散模型的 membership risk 会随着攻击者权限提升迅速放大，而当前轻量防御还不够。

## 90 秒中文版本

如果只看一条攻击线，很容易被怀疑是偶然结果，所以 DiffAudit 做的是多层、多方法、可交叉验证的审计。黑盒主讲线是 admitted 的 `Recon`，同时我们又用 `CLiD` 在同一资产家族上做了独立 corroboration；灰盒主讲线是 `PIA`，并用 `SecMI` 证明这个信号不依赖单一 scorer；白盒 `GSA` 则给出权限提升后的风险上界。我们还明确测试了轻量防御，发现 stochastic dropout 只能把灰盒 `AUC` 从 `0.83863` 拉到 `0.825966`，不能把风险打回随机水平。最重要的结论不是“某篇论文复现成功”，而是“扩散模型的隐私风险跨权限层级持续存在，而且当前简单防御还压不住”。

## 30-Second English Version

DiffAudit audits membership-inference privacy risk in diffusion models. Our key finding is that the leakage is not confined to one attack setting: it grows from black-box to gray-box to white-box access, while lightweight defenses weaken the signal only slightly instead of removing it.

## 60-Second English Version

DiffAudit is not a single attack demo. It is a unified audit stack across attacker capabilities. In black-box settings, admitted `Recon` already reaches `AUC 0.849`. In gray-box settings, `PIA` remains at `AUC 0.83863` on the `1024 / 1024` rung, and stochastic dropout only reduces it to `0.825966`. In white-box settings, `GSA` reaches `0.998192`, showing a near-saturated upper bound once privileged access is available. The overall message is that diffusion-model membership risk persists across access levels and currently outpaces simple lightweight defenses.

## One-Sentence English Version

DiffAudit shows that diffusion-model membership leakage persists across attacker capabilities and is not neutralized by lightweight defenses.

## 评委快问快答

### Q1. 你们一句话最强结论是什么？

扩散模型的 membership leakage 会随着攻击者权限提升而迅速放大，而当前轻量防御还不足以把它消掉。

### Q2. 你们最强的黑盒证据是什么？

admitted `Recon AUC 0.849`，再加上同资产家族上的 `CLiD` corroboration。

### Q3. 你们最强的灰盒证据是什么？

主讲线是 `PIA 1024 / 1024 AUC 0.83863`，最强 alternate scorer 是 `SecMI NNS 0.946286`。

### Q4. 防御到底有没有用？

有用，但只是温和削弱。`PIA` 从 `0.83863` 降到 `0.825966`，仍然显著高于随机。

### Q5. 英文最短怎么答？

`The leakage persists across access levels, and lightweight defenses still do not erase it.`

## 使用建议

- 时间少于 45 秒：直接用 `30 秒中文版本` 或 `30-Second English Version`
- 时间在 1 分钟左右：优先用 `60 秒中文版本`
- 评委要求英文：先说 `One-Sentence English Version`，再补 `60-Second English Version`
- 被追问可信度：补一句 `We use multiple attack mechanisms across the same local evidence stack, not a single lucky result.`
