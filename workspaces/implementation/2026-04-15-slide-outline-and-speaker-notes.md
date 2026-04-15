# 2026-04-15 Slide Outline And Speaker Notes

## 用途

这份稿本用于研究线 3-5 分钟答辩。

推荐顺序：

1. 先讲 threat-model ladder
2. 再讲 black-box admitted headline
3. 接 gray-box + defense
4. 最后用 white-box upper bound 收束

## Slide 1: 我们到底在审计什么

### 屏幕上放什么

- 标题：`DiffAudit: Membership Risk Across Access Levels`
- 一句话：扩散模型训练数据成员身份，是否会从模型行为中泄漏出来
- 三层攻击者：
  - black-box
  - gray-box
  - white-box

### 口播

我们不是在讨论某一个模型某一次偶然攻击成功，而是在审计一个更普遍的问题：当攻击者拥有不同程度的访问权限时，扩散模型的 membership signal 会不会逐步增强。DiffAudit 的核心价值，就是把 black-box、gray-box、white-box 三层风险放进同一个本地证据栈里统一量化。

## Slide 2: 黑盒风险已经可观测

### 屏幕上放什么

- admitted headline：`Recon DDIM public-100 step30`
- 核心指标：`AUC 0.849`
- best single-metric rung：`AUC 0.866`

### 口播

第一层是黑盒。即使攻击者只能看到生成行为，已经可以从输出中恢复出稳定 membership ranking。我们的 admitted 主讲线是 `Recon AUC 0.849`，同时还有一个更强的单指标 rung 到 `0.866`。这说明风险并不需要 privileged access 才会出现。

### 背后证据

- `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`

## Slide 3: 黑盒不是单方法偶然命中

### 屏幕上放什么

- `CLiD` corroboration
- `celeba_target`: `AUC 1.0`
- `celeba_partial_target`: `AUC 1.0`
- 旁注：`workspace-verified local corroboration`

### 口播

如果评委担心黑盒结果只是某一种 scorer 碰巧有效，我们就立刻给第二条黑盒线。`CLiD` 已经在同一 CelebA 资产家族上本地跑通，而且在两个 target-family checkpoint 上都完全分离。这里要说得严谨一点：我们把它表述为本地 corroboration，不宣称是 paper-faithful benchmark，但它足够说明黑盒信号不是单方法幻觉。

### 背后证据

- `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`
- `workspaces/black-box/runs/clid-recon-clip-partial-target100-20260415-r1/summary.json`

## Slide 4: 灰盒在放大样本后依然稳定

### 屏幕上放什么

- `PIA 512 / 512`: `AUC 0.841339`
- `PIA 1024 / 1024`: `AUC 0.83863`
- 关键词：`same-scale stability`

### 口播

第二层是灰盒，也就是攻击者具备有限模型侧访问能力。这里我们的主线是 `PIA`。最关键的点不是某个单次高分，而是规模放大以后结果还站得住。`512 / 512` 到 `1024 / 1024` 基本保持同一量级，说明这个信号不是小样本偶然波动。

### 背后证据

- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`

## Slide 5: 轻量防御有帮助，但不解决问题

### 屏幕上放什么

- baseline：`0.83863`
- stochastic dropout：`0.825966`
- takeaway：`helps, but does not solve`

### 口播

我们也不是只展示攻击，还展示防御。当前最强的轻量灰盒防御是 stochastic dropout at inference time。它确实会让 `PIA` 下降，但只是温和下降，从 `0.83863` 到 `0.825966`，并没有把信号打回随机水平。所以当前最诚实的结论是：轻量防御可以减弱泄漏，但远远谈不上消除泄漏。

### 背后证据

- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
- `workspaces/implementation/2026-04-15-attack-defense-matrix.md`

## Slide 6: 灰盒不是只靠 PIA 一家

### 屏幕上放什么

- `SecMI stat`: `0.885833`
- `SecMI NNS`: `0.946286`
- 关键词：`alternate scorer corroboration`

### 口播

为了避免灰盒叙事被理解成“只靠 PIA 一种目标函数”，我们补了 `SecMI`。它用的是另一类 scorer，却在 full split 上给出更强的区分。这一页的作用不是替代 PIA，而是证明灰盒信号不依赖于单一攻击实现。

### 背后证据

- `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- `workspaces/gray-box/2026-04-15-pia-scale-vs-secmi-note.md`

## Slide 7: 权限一旦升到白盒，风险接近饱和

### 屏幕上放什么

- `GSA`: `AUC 0.998192`
- defended comparator：`DPDM W-1 0.488783`
- 关键词：`privileged upper bound`

### 口播

最后是白盒。`GSA` 基本接近饱和，说明一旦攻击者拥有 privileged access，membership inference 会变得接近 trivial。这里一定要强调边界：这不是普通产品 KPI，而是风险上界。但对于评委来说，这一页非常重要，因为它把“权限越高，风险越大”的故事收得非常完整。

### 背后证据

- `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`

## Slide 8: 最终结论

### 屏幕上放什么

- `Leakage persists across access levels`
- `Method changes do not erase the signal`
- `Lightweight defenses are insufficient`

### 口播

如果最后只能留下一个结论，那就是：扩散模型的 membership leakage 不是某一篇论文、某一次攻击、某一个权限设定下的偶发现象。它会随着攻击者权限提升而迅速放大，而且当前轻量防御还不足以把这个信号真正消掉。DiffAudit 的价值，就是把这个事实从多条方法线、多层权限线和防御对照线一起证明出来。

## 备用页

### 备用页 A: 负结果也有价值

- `Recon + CLiD` sample-level late fusion 可行，但没有超过已饱和的 `CLiD`
- 说明：我们不仅记录成功线，也记录“为什么没有继续往这条线上烧 GPU”

### 备用页 B: 我们最谨慎的边界表述

- `Recon`：admitted public-subset black-box headline
- `CLiD`：workspace-verified local corroboration
- `PIA / SecMI`：strong local runtime evidence with provenance boundary
- `GSA`：privileged upper bound

## 使用建议

- 3 分钟版本：讲 Slide 1 -> 2 -> 4 -> 5 -> 7 -> 8
- 5 分钟版本：完整讲 1 到 8
- 如果评委追问可信度：优先跳到 Slide 3 和 Slide 6
- 如果评委追问防御：重点放 Slide 5，并补 white-box defended comparator
