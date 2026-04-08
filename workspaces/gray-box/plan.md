# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件相关评分、噪声预测与结构特征下的成员推断
- `status`: `PIA real-asset runtime-mainline ready; GPU128/GPU256/GPU512 baseline + defended pairs landed; provisional G-1 established`
- `blocked by`: `PIA` provenance 尚未升级到 paper-aligned；`SecMI` 当前已判定为 blocked baseline；当前 gray-box defense 仍缺同档重复轮次确认
- `next step`: 固定 `stochastic-dropout = provisional G-1`，然后决定是做 `GPU512` 重复确认还是继续推进 `PIA` provenance 核准；`SecMI` 只保留为 blocked baseline
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
- `SecMI` 是当前已明确阻塞、不能继续抢主线资源的 baseline

## 当前可执行证据

- `workspaces/gray-box/2026-04-07-pia-runtime-mainline.md`
- `workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- `workspaces/gray-box/pia-intake-gate.md`
- `workspaces/gray-box/assets/pia/manifest.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260407-cpu/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260407-cpu/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`
- `workspaces/gray-box/2026-04-08-pia-gpu128-attack-defense.md`
- `workspaces/gray-box/2026-04-08-pia-gpu256-attack-defense.md`
- `workspaces/gray-box/2026-04-08-pia-gpu512-attack-defense.md`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512/summary.json`
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
6. 固定 `SecMI = blocked baseline`，直到真实资产到位

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
- `PIA baseline` 的 `128` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`
  - 结果：`auc=0.817444 / asr=0.765625 / tpr@1%fpr=0.046875 / tpr@0.1%fpr=0.039062`
- `PIA + stochastic-dropout defense` 的 `128` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128/summary.json`
  - 结果：`auc=0.803955 / asr=0.757812 / tpr@1%fpr=0.03125 / tpr@0.1%fpr=0.015625`
- 当前结论补充：
  - `GPU128` 这对结果第一次在同口径下给出了防御优于 baseline 的方向性信号
  - 这足以把 `stochastic-dropout` 从“只有 runnable hook”推进到“provisional G-1 candidate”
  - 但当前仍不能写成已验证有效防御，因为还缺重复轮次或更大样本量确认
- `PIA baseline` 的 `256` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`
  - 结果：`auc=0.841293 / asr=0.78125 / tpr@1%fpr=0.039062 / tpr@0.1%fpr=0.019531`
- `PIA + stochastic-dropout defense` 的 `256` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/summary.json`
  - 结果：`auc=0.82901 / asr=0.767578 / tpr@1%fpr=0.027344 / tpr@0.1%fpr=0.015625`
- 当前结论再补充：
  - `GPU256` 继续复现了 `GPU128` 的同方向下降
  - 当前 gray-box defense 已不只是 provisional candidate，而是本轮最合理的 provisional `G-1`
  - 仍不能写成 validated privacy win，因为还没有 `GPU512` 或重复轮次确认
- `PIA baseline` 的 `512` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512/summary.json`
  - 结果：`auc=0.841339 / asr=0.786133 / tpr@1%fpr=0.058594 / tpr@0.1%fpr=0.011719`
- `PIA + stochastic-dropout defense` 的 `512` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512/summary.json`
  - 结果：`auc=0.82938 / asr=0.769531 / tpr@1%fpr=0.023438 / tpr@0.1%fpr=0.009766`
- 当前结论最终补充：
  - `GPU512` 继续复现了 `GPU128 / GPU256` 的同方向下降
  - 当前 gray-box defense 已经可以正式写成 `provisional G-1`
  - 仍不能写成 validated privacy win，因为还缺同档重复确认与 provenance 升级

## 当前阻塞项

- `PIA` 当前仍是 `source-retained-unverified`
- 当前 gray-box defense 已出现三档 favorable signal，但仍不是 validated privacy win
- `SecMI` 当前 probe 已明确缺真实 `flagfile.txt`

## 当前最短路径

1. 固定 `stochastic-dropout = provisional G-1`
2. 决定 `GPU512` 重复确认还是继续推进 `PIA` provenance 核准
3. 保持 `SecMI = blocked baseline`
4. 复用 [unified table](../implementation/2026-04-08-unified-attack-defense-table.md) 作为灰盒对外引用入口
5. 只有在真实 `flagfile + checkpoint root` 到位后，才恢复 `SecMI`
