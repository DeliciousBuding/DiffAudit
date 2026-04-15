# 2026-04-15 One-Page Judge Cheat Sheet

## 一页只记这 6 句话

1. `DiffAudit` 审计的是扩散模型的 membership inference 风险，不是单篇论文复现。
2. 黑盒已经可观测：`Recon AUC 0.849`，且有 `CLiD` 独立 corroboration。
3. 灰盒在放大样本后仍稳定：`PIA 1024 / 1024 AUC 0.83863`。
4. 轻量防御只温和削弱：stochastic dropout 后仍有 `AUC 0.825966`。
5. 灰盒不是单方法假象：`SecMI stat 0.885833`，`SecMI NNS 0.946286`。
6. 白盒上界接近饱和：`GSA 0.998192`，说明权限提升会迅速放大风险。

## 评委如果只问“你们到底证明了什么”

我们证明了扩散模型的 membership leakage 跨 black-box、gray-box、white-box 三层权限持续存在，而且当前轻量防御还不足以把这个风险真正消掉。

## 评委如果问“最强证据是哪条”

- 黑盒主讲：`Recon 0.849`
- 灰盒主讲：`PIA 1024 / 1024 0.83863`
- 灰盒最强 alternate scorer：`SecMI NNS 0.946286`
- 白盒上界：`GSA 0.998192`

## 评委如果问“防御有没有用”

有用，但不够。`PIA` 从 `0.83863` 降到 `0.825966`，仍明显高于随机。

## 评委如果质疑“是不是就复现了几篇论文”

不是。我们的贡献是把多种攻击方法、多种访问权限、同资产交叉验证和防御对比收成一个本地统一审计栈。

## 评委如果追问“哪些地方要谨慎说”

- `CLiD`：说 local corroboration，不说 full paper-faithful benchmark
- `PIA / SecMI`：说 strong local runtime evidence，不夸大到通用 benchmark
- `GSA`：说 privileged upper bound，不包装成普通产品场景 KPI

## 如果只有 15 秒收尾

扩散模型的隐私风险会随着攻击者权限提升而迅速放大，而当前轻量防御还压不住这个泄漏信号。

## 上台前最后看一眼

- 先讲 threat-model ladder，再讲黑盒、灰盒、防御、白盒
- 不要把 `CLiD` 说成 paper-faithful
- 不要把 `GSA` 说成普通场景风险值
- 被压时间时，优先保留：`Recon 0.849`、`PIA 0.83863 -> 0.825966`、`GSA 0.998192`
