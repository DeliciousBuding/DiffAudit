# 黑盒工作区

用于存放：

- 黑盒论文阅读笔记
- 黑盒复现计划
- 黑盒实验对比表
- 黑盒方向任务认领

## 当前推荐主线

- 主证据线：`2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- 次主线候选：`2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`
- 条件差异补充：`2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf`

## 当前补充文档

- `plan.md`
- `paper-matrix-2024-2026.md`
- `experiment-entrypoints.md`
- `2026-04-08-variation-local-track.md`

## 第一轮建议任务

1. 先整理两篇论文的攻击假设、输入输出、指标和资产要求
2. 当前黑盒次主线 `variation/Towards` 已在本地 CPU 上重复跑过 synthetic smoke，可以作为“本地可讲的第二黑盒路线”继续保留
3. 先补黑盒方向阅读笔记和复现计划，不要把 `SecMI` 混进黑盒入口
4. 真实 API 资产到位后再把 `variation` 从 synthetic-smoke 推向真实 black-box mainline
