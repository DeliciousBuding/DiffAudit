# 黑盒实验入口

这份文档只回答一个问题：黑盒方向读完论文后，下一步该落到仓库哪个入口。

## 入口一：`recon-mainline`

- 目标：维持当前最强 black-box evidence line。
- 适配论文：`Black-box Membership Inference Attacks against Fine-tuned Diffusion Models`。
- 命令顺序：
  - `python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml`
  - `python -m diffaudit probe-recon-assets --config configs/attacks/recon_plan.yaml`
  - `python -m diffaudit dry-run-recon --config configs/attacks/recon_plan.yaml --repo-root external/Reconstruction-based-Attack`
  - `python -m diffaudit run-recon-mainline-smoke --workspace experiments/recon-mainline-smoke --repo-root external/Reconstruction-based-Attack --method threshold`
- 产出：black-box 主证据、对照基线和 artifact mainline 入口。

## 入口二：`variation-track`

- 目标：把 `Towards Black-Box` 升成正式本地黑盒次主线。
- 适配论文：`Towards Black-Box Membership Inference Attack for Diffusion Models`。
- 当前已验证：
  - `experiments/variation-synth-smoke/summary.json`
  - `experiments/variation-synth-smoke-local-20260408/summary.json`
- 命令顺序：
  - `python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml`
  - `python -m diffaudit probe-variation-assets --config configs/attacks/variation_plan.yaml`
  - `python -m diffaudit dry-run-variation --config configs/attacks/variation_plan.yaml`
  - `python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke`
- 产出：
  - local synthetic-smoke verified
  - real API blocked / ready 的显式判定

## 入口三：`clid-track`

- 目标：把 prompt-conditioned 成员推断补成独立轨道。
- 适配论文：`CLiD`。
- 最小前提：
  - 有图文对或 prompt 条件
  - 能定义条件似然差异或等价评分接口
  - 能记录 caption-known 的输入输出
- 预期补充：
  - 新配置模板
  - 条件输入 schema
  - 结果记录字段

## 入口四：`dataset-audit-track`

- 目标：把单样本成员判别扩展为数据集级审计。
- 适配论文：`CDI`，次选 `SIDE` 的泄露讨论。
- 最小前提：
  - 候选数据集或版权样本集合
  - 聚合证据统计方式
  - 明确单样本与数据集级任务的边界
- 预期补充：
  - 审计证据 schema
  - 聚合统计脚本
  - 更偏报告化的输出

## 不属于黑盒入口的内容

- `SecMI`
- `PIA`

它们属于灰盒路线，不应继续作为黑盒入口文档的一部分。
