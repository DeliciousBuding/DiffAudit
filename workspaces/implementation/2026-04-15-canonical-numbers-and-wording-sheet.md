# 2026-04-15 Canonical Numbers And Wording Sheet

## 用途

这份表是研究线答辩时的口播数字单一真源。

用途只有两个：

1. 防止不同材料里的数字在口播时串台
2. 给每个核心数字配一条“推荐讲法”，避免说过头

## 核心主讲数字

| Topic | Canonical Number | 推荐讲法 | 不要这样说 |
|---|---:|---|---|
| Black-box admitted headline | `Recon AUC 0.849` | 黑盒下已经能观察到稳定 membership signal | 这是通用互联网场景攻击成功率 |
| Black-box best single-metric rung | `Recon AUC 0.866` | 这是当前更强的单指标 rung，但 subset 更窄 | 这是新的 admitted 主讲线 |
| Black-box corroboration | `CLiD AUC 1.0 / 1.0` | CLiD 在同资产家族上完成独立 local corroboration | 这是 paper-faithful benchmark |
| Gray-box baseline | `PIA 1024 / 1024 AUC 0.83863` | 灰盒信号在 scale-up 后仍稳定 | 这是任何灰盒场景下的固定效果 |
| Gray-box defense | `0.83863 -> 0.825966` | stochastic dropout 有帮助，但只温和削弱 | 这个防御已经解决泄漏 |
| Gray-box corroboration | `SecMI stat 0.885833 / NNS 0.946286` | 灰盒信号不依赖单一 scorer | SecMI 完全替代 PIA 主线 |
| White-box upper bound | `GSA 0.998192` | 一旦进入 privileged access，风险接近饱和 | 这是普通产品默认风险值 |

## 最短主讲句

### 15 秒

扩散模型的 membership leakage 会随着攻击者权限从 black-box、gray-box 到 white-box 逐步放大，而当前轻量防御还压不住这个信号。

### 30 秒

DiffAudit 的核心发现不是某一次攻击成功，而是扩散模型的 membership risk 跨权限层级持续存在。黑盒有 `Recon 0.849`，灰盒有 `PIA 0.83863`，轻量防御后还有 `0.825966`，白盒上界 `GSA` 接近 `0.998192`。

## 高风险措辞替换表

| 容易说过头的话 | 建议替换成 |
|---|---|
| 我们证明了真实世界一定会被打穿 | 我们证明了在当前本地证据栈里风险信号持续存在 |
| CLiD 完整复现了论文 | CLiD 在本地工作区内完成了独立 corroboration |
| GSA 代表产品真实风险 | GSA 代表 privileged access 下的风险上界 |
| admitted 就是最强结果 | admitted 是当前最适合公开主讲、边界最稳的一条结果 |

## 数字回查位置

- `Recon admitted`: `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
- `Recon best single-metric`: `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- `PIA baseline`: `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- `PIA defended`: `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
- `SecMI`: `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- `GSA`: `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`

## 使用规则

- 临上场先看这页，再看 `one-page judge cheat sheet`
- 如果和其他文档数字冲突，以这页和对应 `summary.json` 为准
- 若要改口播数字，必须先改这里，再改其他包装材料
