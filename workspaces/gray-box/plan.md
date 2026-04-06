# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件似然、结构特征下的成员推断
- `status`: `PIA runtime path ready; real-asset probe pending`
- `blocked by`: 真实 `checkpoint`、真实 `dataset_root`、`dataset_root/cifar10` 目录布局
- `next command`: `conda run -n diffaudit-research python -m diffaudit probe-pia-assets --config <real-pia-config> --member-split-root external/PIA/DDPM`
- `last updated`: 2026-04-07

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

- `experiments/pia-runtime-smoke-cpu/summary.json`
- `experiments/pia-runtime-smoke-gpu/summary.json`
- `experiments/pia-synth-smoke-cpu/summary.json`
- `experiments/pia-synth-smoke-gpu/summary.json`
- `workspaces/gray-box/runs/pia-followup-20260407/plan.json`
- `workspaces/gray-box/runs/pia-followup-20260407/probe.json`
- `workspaces/gray-box/runs/pia-followup-20260407/dry-run.json`
- `experiments/clid-dry-run-smoke/summary.json`
- `experiments/clid-artifact-summary/summary.json`

## 本地代码上下文

- `external/PIA`
- `external/CLiD`

## 起步方案

1. 先把灰盒边界定义写清楚
2. 先把 `PIA` 的真实 `checkpoint + dataset_root` 绑定到非占位 config
3. 先跑 `probe-pia-assets`，确认缺口只剩真实资产，不再重复 GPU synthetic smoke
4. `probe` 通过后再跑 `runtime-probe-pia --device cpu`
5. 只有在真实资产探针通过后，才评估是否进入更正式的 gray-box benchmark
6. `SecMI` 继续作为第二条 baseline，等 `flagfile + checkpoint` 布局就位再接续

## 当前阻塞项

- 模板 config 仍是占位符，不能直接声称 `PIA` 真实资产已可用
- `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` 已在位，但真实 `checkpoint / dataset_root / dataset layout` 还没到位
- 当前最短下一步是 CPU 上的真实资产探针，不是重复的 GPU smoke
