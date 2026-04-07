# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件相关评分、噪声预测与结构特征下的成员推断
- `status`: `PIA real-asset runtime-mainline ready; G-1 formalization pending`
- `blocked by`: `PIA` provenance 尚未升级到 paper-aligned；`SecMI` 真实资产闭环尚未完成；当前 gray-box defense 仍未形成正式同口径对比表
- `next step`: 基于 `32` 样本 run 继续扩大 `PIA baseline + defended` 的样本量与重复次数，并把现有 defense prototype 正式定义成 `G-1`
- `last updated`: `2026-04-08`

## 推荐论文

- `2024-iclr-pia-proximal-initialization.pdf`
- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf`
- `2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`
- `2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf`
- `2026-openreview-mofit-caption-free-membership-inference.pdf`

## 当前主线与 baseline

- 主线：`PIA`
- baseline：`SecMI`

当前判断：

- `PIA` 是当前最成熟、最适合作为“攻击 + 防御”主讲闭环的一条线
- `SecMI` 是当前应继续推进但不能抢主线资源的 baseline

## 当前可执行证据

- `workspaces/gray-box/2026-04-07-pia-runtime-mainline.md`
- `workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- `workspaces/gray-box/pia-intake-gate.md`
- `workspaces/gray-box/assets/pia/manifest.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260407-cpu/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260407-cpu/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`
- `experiments/pia-runtime-smoke-cpu/summary.json`
- `experiments/pia-runtime-smoke-gpu/summary.json`
- `experiments/pia-synth-smoke-cpu/summary.json`
- `experiments/pia-synth-smoke-gpu/summary.json`
- `experiments/secmi-synth-smoke/summary.json`
- `experiments/secmi-synth-smoke-gpu/summary.json`

## 本地代码上下文

- `external/PIA`
- `external/SecMI`
- `third_party/secmi`

## 当前推荐执行顺序

1. 读透 `PIA` 论文，把攻击依赖的核心信号写清楚
2. 重新跑 `PIA baseline`
3. 重新跑 `PIA defended`
4. 用同一套字段记录：
   - `AUC`
   - `ASR`
   - `TPR@1%FPR`
   - `TPR@0.1%FPR`
   - `elapsed_seconds`
   - `num_samples`
5. 把现有 defense prototype 正式命名为 `G-1`
6. 对 `SecMI` 做 promote / block 判定

## 2026-04-08 新观察

- `PIA baseline` 的 `32` 样本 CPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
  - 结果：`auc=0.782227 / asr=0.765625 / tpr@1%fpr=0.09375`
- `PIA + stochastic-dropout defense` 的 `32` 样本 CPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`
  - 结果：`auc=0.769531 / asr=0.75 / tpr@1%fpr=0.09375`
- 当前结论：
  - `stochastic-dropout` 仍只是 runnable hook
  - 在当前 `32` 样本 local run 下，它没有带来更好的排序指标，反而略差
  - 因此当前不能把它写成有效 `G-1`，只能写成“已具备正式可比接口，但尚未验证有效”
- 本轮顺手修复了 `PIA` runtime mainline summary 的 `runtime.num_samples` 口径 bug：
  - 现在 summary 会记录实际生效样本数，而不是配置中的默认值

## 当前阻塞项

- `PIA` 当前仍是 `source-retained-unverified`
- 当前 gray-box defense 只是 runnable hook，不是 validated privacy win
- `SecMI` 仍缺真实 checkpoint、flagfile 与论文一致 layout
- 当前还没有 gray-box attack-defense 总表

## 当前最短路径

1. `PIA baseline + defended`
2. `PIA` provenance 核准
3. `SecMI` 真实资产闭环或降级
4. 把 gray-box 对比结果接入统一总表
