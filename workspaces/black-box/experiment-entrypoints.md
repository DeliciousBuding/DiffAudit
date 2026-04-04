# 黑盒实验入口

这份文档只回答一个问题：黑盒方向读完论文后，下一步该落到仓库哪个入口。

## 入口一：`secmi-baseline`

- 目标：跑通当前仓库已有的 `SecMI` 资产探针、adapter、dry-run 和 synthetic smoke。
- 适配论文：`2025-324-paper` 的最小工程基础、`Towards Black-Box...` 的 baseline 准备。
- 命令顺序：
  - `python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml`
  - `python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml`
  - `python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi`
  - `python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi`
- 产出：`code-ready / asset-ready / blocked` 状态和缺失资产说明。

## 入口二：`clid-track`

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

## 入口三：`dataset-audit-track`

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
