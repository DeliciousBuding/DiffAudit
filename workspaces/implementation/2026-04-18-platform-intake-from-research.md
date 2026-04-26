# 2026-04-18 Platform Intake From Research

这份说明用于回答一个很实际的问题：

今晚如果 `Platform` 要交源码，`Research` 当前到底有哪些真实进展，值得同步进 `Platform`；哪些还应该留在研究侧，不要急着平台化。

## 1. 结论先行

有，而且已经不止一点。

但最稳妥的同步策略不是“把研究里所有新东西都塞进 Platform”，而是分三层：

1. **今晚必须同步的**
   - 刷新 `Platform/apps/api-go/data/public/attack-defense-table.json`
   - 保持 `catalog.json` 的 4 个主合同不扩张，但修正文案边界
   - 确保 `recon / PIA / GSA-W1` 三条 admitted 主线的字段、指标、边界与 `Research` 当前真值一致
2. **今晚可以安全补进去的 challenger / corroboration**
   - `TMIA-DM`
   - `SecMI`
   - `recon DDIM public-50 step10`
   - 如 UI 不受影响，也可补 `PIA GPU1024` 与 `CLiD local candidate` 行
3. **今晚不应平台化的**
   - `06/05/04` 这些报告驱动的规划层结论本身
   - `Finding NeMo` 当前 falsifier packet
   - `batch32 bridge diagnostic`
   - `loss-score white-box auxiliary packet`
   - `I-D` conditional packet truth

## 2. 今晚必须同步的内容

### 2.1 `attack-defense-table.json` 已经落后于 `Research`

当前 `Research` 的统一表有 `15` 行，更新时间是 `2026-04-16 07:20 +08:00`。

当前 `Platform` snapshot 里的 `attack-defense-table.json` 只有 `6` 行，更新时间还是 `2026-04-09 17:05 +08:00`。

所以如果今晚只做一件研究侧同步动作，最值钱的就是：

- 用 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`
- 重新生成或覆盖 `Platform/apps/api-go/data/public/attack-defense-table.json`

### 2.2 admitted 主线本身已经够进 Platform

以下几条，不仅能进，而且应该继续作为 Platform 的主展示面：

- 黑盒 `recon DDIM public-100 step30`
- 灰盒 `PIA GPU512 baseline`
- 灰盒 `PIA GPU512 + provisional G-1 stochastic-dropout(all_steps)`
- 白盒 `GSA 1k-3shadow`
- 白盒 `W-1 strong-v3 full-scale`

这些条目已经是当前最稳的 admitted 面，适合继续当 `/workspace` 与 `/reports` 的主表内容。

## 3. 今晚可以安全补进去的研究进展

### 3.1 可以补成 secondary / challenger 的

这些是“已经有真实结果，而且 Platform 结构能容纳”的：

- `recon DDIM public-50 step10`
  - 适合作为 `best single metric reference`
  - 但不替代 `public-100 step30` 主证据
- `TMIA-DM late-window GPU256`
  - 可作为 gray-box challenger
  - defended 侧可带 `dropout` 与 `temporal-striding`
- `SecMI GPU full split`
  - 可作为 gray-box 独立 corroboration line
- `PIA GPU1024 baseline / defended`
  - 可作为 scale-up validation rung
  - 不替代当前 admitted 主 headline
- `CLiD clip local target100 / partial-target100`
  - 只在 UI 已有 `runtime-candidate` / `local-candidate` 容器时才建议加
  - 否则今晚可以先不放，避免误导

### 3.2 这些补进去的价值

不是为了“让表更长”，而是为了让 Platform 开始具备：

- `headline / challenger / corroboration` 的层次
- 不只讲 admitted mainline，也能讲当前候选面
- 让 `Research` 的真实推进，不至于在平台上看起来像 4 月 9 日后就没动过

## 4. 今晚不建议塞进 Platform 的内容

这些不是没价值，而是当前仍属于研究判断层，不适合直接变成平台展示结果：

- `06-g1a / 05-cross-box / 04-defense` 的规划结论
  - 它们是当前主线排序
  - 不是平台结果
- `Finding NeMo`
  - 当前最诚实口径是 `non-admitted actual bounded falsifier`
  - 不适合今晚变成平台 headline
- `batch32 same-protocol bridge diagnostic`
  - 仍是 diagnostic，不是 benchmark-ready
- `white-box loss-score` 这条辅助线
  - 目前仍是 bounded auxiliary evidence
- `I-D` conditional packet truth
  - 现在更像 future-surface note，而不是产品层 contract

## 5. 今晚最需要修的不是“有没有数据”，而是边界文案

当前 `Platform` 里已经有一些研究边界不够稳的地方，今晚如果时间有限，至少要注意别继续放大。

### 5.1 灰盒文案不要写成“防御已验证有效”

当前最诚实的说法仍然是：

- `workspace-verified + adaptive-reviewed`
- `paper-alignment blocked by checkpoint/source provenance`

所以 Platform 上可以说：

- current admitted defense prototype
- measured reduction under current protocol

但不要写成：

- validated privacy protection
- defense solved the problem

### 5.2 白盒文案不要写成“风险被彻底消除”

当前 `catalog.json` 里的白盒解释已经有点过强。

更稳妥的说法应该是：

- current defended comparator is close to random under the present admitted protocol
- this supports strong mitigation under the current white-box comparison setup

而不是：

- strong defense effectively eliminates risk

因为这会把当前 defended comparator 误读成最终 benchmark 结论。

### 5.3 Platform 不要把自己写成“防御系统”

当前平台自己的边界是对的：

- 它是 risk audit platform
- 不是 defense system

这一点今晚一定要保住。Research 当前也没有支持“平台本身已经提供防御能力”的结论。

## 6. 给 Platform 今晚的最小动作建议

如果目标是“今晚交源码、比赛可用、别过度冒险”，我建议动作压成下面这几步：

1. 刷新 `attack-defense-table.json`
2. 复核 `catalog.json` 的 `risk_interpretation`，把过强表述降下来
3. 保持 `catalog` 里的 4 个主合同不变：
   - `black-box/recon/sd15-ddim`
   - `black-box/recon/kandinsky-v22`
   - `gray-box/pia/cifar10-ddpm`
   - `white-box/gsa/ddpm-cifar10`
4. 如果前端不会被额外行数扰乱，再把 `TMIA-DM / SecMI / recon-step10` 这些 challenger/corroboration 行补进表
5. 暂时不要为了展示“研究很活跃”而把规划层和 falsifier 层结果硬塞进平台

## 7. 可以继续从 Research 学什么

如果是为了比赛后继续推进，当前最值得学、也最值得让 `Research` 持续深挖的是：

### 7.1 `06-g1a`

- `temporal QR surrogate`
- `RMIA / BASE temporal LR`
- 如何在不补齐 `TMIA-DM 512` 的前提下，做一个足够诚实的 blocker-resolution packet

### 7.2 `05-cross-box`

- `GSA + PIA` shared pairboard
- `best-single / weighted / logistic`
- `support / disconfirm / neutral`
- low-FPR tail 上到底有没有真实增益

### 7.3 `04-defense`

- `risk-targeted SISS / retain-forget mixture`
- `privacy-aware adapter`
- 如何只挑一个 successor family 做 bounded pilot，而不是把防御面继续摊大

### 7.4 `02-gray-box`

- paper-faithful `SimA`
- `PIA + SimA`
- second signal 是否真的能服务 `05/04`

### 7.5 `03-white-box`

- `activation-subspace fingerprint`
- `score-vector geometry`
- `risky-subspace pruning / targeted unlearning`

## 8. 一句话给你的判断

如果你问“Research 现在有没有已经够真实、够稳、今晚就值得进入 Platform 的东西”，答案是：

**有，而且主线已经够。**

但今晚最该做的是：

- 把 `Research` 的 admitted truth 和 challenger truth 更完整地同步进 `Platform`
- 同时把过强文案降到诚实边界

而不是把所有新研究包装成产品结果。
