# 2026-04-15 Judge FAQ Short

## 一句话主结论

DiffAudit 证明：扩散模型的 membership risk 不只存在于一种攻击设定里，而是会随着攻击者权限提升而显著增强，且轻量防御目前只能削弱、不能消除该风险。

## 最短讲法

- 黑盒：`Recon` 已有 admitted 主讲线，`AUC 0.849`。
- 灰盒：`PIA` 在 `1024 / 1024` 规模下仍有 `AUC 0.83863`，加 stochastic dropout 后仍有 `AUC 0.825966`。
- 灰盒交叉验证：`SecMI stat 0.885833`，`SecMI NNS 0.946286`。
- 白盒上界：`GSA 0.998192`，说明 privileged access 风险接近饱和。

## 评委高频问题

### 1. 你们到底证明了什么？

证明了 diffusion-model membership leakage 不是单点现象，而是跨 black-box、gray-box、white-box 三层访问权限都可观测。

### 2. 这是不是只是复现论文？

不是。论文方法是起点，但当前价值在于把多条攻击线、多种权限假设、同资产交叉验证和防御对比收口成一个可审计的本地证据栈。

### 3. 黑盒结果会不会只是某一个方法碰巧有效？

主讲线是 `Recon`，但我们又补了独立的 `CLiD` corroboration。两种 black-box 机制在同一 CelebA 资产家族上都能稳定亮灯，所以不是单一 scorer 偶然有效。

### 4. 你们最强的结果是哪条？

如果问原始攻击强度，最强是 white-box `GSA 0.998192`。如果问灰盒交叉验证强度，最强是 `SecMI NNS 0.946286`。如果问最适合公开主讲的黑盒主线，还是 admitted `Recon 0.849`。

### 5. 防御有效吗？

有效，但只是温和削弱。`PIA 1024 / 1024` 从 `0.83863` 降到 `0.825966`，说明 stochastic dropout 会影响排序信号，但并没有把泄漏打回随机水平。

### 6. 最需要谨慎表述的点是什么？

- `CLiD` 现在应表述为 workspace-verified local corroboration，不宣称 paper-faithful benchmark。
- `PIA` 和 `SecMI` 是强本地证据，但仍受 checkpoint/protocol provenance 边界约束。
- `GSA` 是 privileged upper bound，不应被包装成普通产品场景下的默认风险值。

### 7. 如果只能留给评委一句话，应该说什么？

扩散模型的隐私风险会随攻击者权限提升而快速放大，而当前轻量防御还不足以把 membership leakage 消掉。
