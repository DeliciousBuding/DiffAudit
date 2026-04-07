# 2026-04-08 Black-Box Follow-Up: Variation Local Track

## 结论

`variation` 现在应被提升为正式本地黑盒次主线，但当前口径只能到：

- `local synthetic-smoke verified`
- 不能写成 `real black-box mainline`

## 依据

论文来源：

- `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`

本地代码入口：

- `src/diffaudit/attacks/variation.py`
- `configs/attacks/variation_plan.yaml`

本地运行证据：

- `experiments/variation-synth-smoke/summary.json`
- `experiments/variation-synth-smoke-local-20260408/summary.json`

## 当前能说的话

1. 这条线已经不是纯文献储备，而是有本地可重复的 CPU synthetic smoke。
2. 它和当前 black-box 主证据线 `recon` 的角色不同：
   - `recon` 负责当前最强风险证据
   - `variation` 负责“严格 API-only black-box”叙事和第二候选线
3. 它很适合对齐师兄口中的“黑盒简单一点、先做一个能讲的黑盒路线”。

## 当前不能说的话

1. 不能说已经有真实 variation API 闭环。
2. 不能说已经有真实 query-image set。
3. 不能说已经复现论文中的 DDIM / DiT / Stable Diffusion 完整结果。

## 当前阻塞

- 缺真实 API 凭据
- 缺真实 query image 集
- 缺真实 query budget
- 缺与论文一致的 variation endpoint 或等价本地代理

## 当前推荐口径

最稳妥的表述是：

`Towards/variation` 已从纯文献候选升级为 formal local black-box secondary track。当前已在 CPU 上完成重复 synthetic-smoke 验证，证明其 threshold-style black-box evaluation path 可在本地工作；但真实 API 资产仍未到位，因此暂不写成真实 black-box mainline。

## 下一步

1. 保留 `variation` 作为黑盒第二候选线
2. 不让它抢 `recon` 主证据线资源
3. 一旦真实 API、预算和 query image 集到位，就先做 `probe-variation-assets`
4. 如果申报阶段需要一个更贴近师兄口头方案的黑盒路线，就优先引用这条线，而不是把灰盒方法硬塞进黑盒叙事
