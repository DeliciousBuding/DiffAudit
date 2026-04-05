# 灰盒方向计划

## 状态面板

- `owner`: 待认领
- `scope`: 部分中间信息、条件似然、结构特征下的成员推断
- `status`: PIA 最小复现推进中
- `blocked by`: gray-box observable 定义、caption-known 与 unknown 的划分、统一成本指标
- `next command`: `conda run -n diffaudit-research python -m diffaudit run-pia-synth-smoke --workspace experiments/pia-synth-smoke-gpu --repo-root external/PIA --device cuda`
- `last updated`: 2026-04-05

## 推荐论文

- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-iclr-pia-proximal-initialization.pdf`
- `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf`
- `2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf`
- `2026-openreview-mofit-caption-free-membership-inference.pdf`
- `2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`
- `2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf`

## 当前定义建议

灰盒 = 可访问部分中间扩散信息、噪声预测、条件相关评分或中间去噪结果，但不可访问完整参数和梯度。

## 当前可执行证据

- `experiments/pia-synth-smoke-cpu/summary.json`
- `experiments/pia-synth-smoke-gpu/summary.json`

## 起步方案

1. 先把灰盒边界定义写清楚
2. 先复现 `SecMI`
3. 再用 `PIA` 做效率和攻击代价对照，并保留一条 synthetic smoke 复现链
4. 用 `CLiD` 作为更贴近 text-to-image / conditional setting 的扩展参考
5. 增补 `caption-known / caption-unknown` 双轨假设
6. 维护攻击代价 ledger：`queries/sample`、`timesteps`、优化步数、GPU-hours、AUC`

## 当前阻塞项

- 需要统一“灰盒”到底允许哪些内部信息
- 需要确定后续系统里灰盒接口如何表达
- 需要区分灰盒和黑盒条件信息利用的边界
