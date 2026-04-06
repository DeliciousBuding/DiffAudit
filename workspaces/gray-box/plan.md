# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件似然、结构特征下的成员推断
- `status`: `PIA real-asset preview ready`
- `blocked by`: 真实 `PIA` runtime mainline 仍未接进 `Project`，且当前 DDPM checkpoint 的论文来源尚未核准
- `next step`: 在不申请 GPU 的前提下，决定是先补 `Project` 侧真实 `PIA` 执行封装，还是先核准这批 DDPM checkpoint 的 provenance
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
- `workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- `workspaces/gray-box/runs/pia-cifar10-graybox-assets-probe-20260407/runtime-preview.json`
- `experiments/clid-dry-run-smoke/summary.json`
- `experiments/clid-artifact-summary/summary.json`

## 本地代码上下文

- `external/PIA`
- `external/CLiD`

## 起步方案

1. 先把灰盒边界定义写清楚
2. 已把 `PIA` 的真实 `checkpoint + dataset_root + member split` 绑定到本地-only config，并确认 `probe/dry-run/runtime-probe/runtime-preview` 都能在 CPU 上走通
3. 不再重复 GPU synthetic smoke
4. 下一步只做两件事中的一件：
   - 补 `Project` 侧真实 `PIA` 执行封装
   - 先核准当前 DDPM checkpoint 的 paper provenance
5. `SecMI` 继续作为第二条 baseline，等 `flagfile + checkpoint` 布局就位再接续

## 当前阻塞项

- `PIA` 的真实数据 preview 已经通过，但当前仓库还没有真实 runtime mainline runner
- 这批新到位的 DDPM checkpoint 能通过当前 probe/runtime-probe，不等于已经核准为论文口径资产
- 当前最短下一步不是 GPU，而是执行封装或 provenance 核准
