# 黑盒工作区

用于存放：

- 黑盒论文阅读笔记
- 黑盒复现计划
- 黑盒实验对比表
- 黑盒方向任务认领

## 当前推荐主线

- 主论文：`2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- 对照论文：`2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`

## 当前补充文档

- `plan.md`
- `paper-matrix-2024-2026.md`
- `experiment-entrypoints.md`

## 第一轮建议任务

1. 先整理两篇论文的攻击假设、输入输出、指标和资产要求
2. 先按 `configs/assets/example.local.yaml` 填好本地资产路径，再跑 `plan-secmi / probe-secmi-assets / prepare-secmi / dry-run-secmi`
3. 先补黑盒方向阅读笔记和复现计划
4. 真实资产到位后再开始跑实验
