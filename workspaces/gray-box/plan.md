# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件相关评分、噪声预测与结构特征下的成员推断
- `status`: `PIA real-asset runtime-mainline ready; GPU128/GPU256/GPU512 baseline + defended pairs landed; GPU512 rerun confirmed; GPU128/GPU256 adaptive portability pair landed on RTX4070 8GB; provisional G-1 established; provenance now workspace-verified`
- `blocked by`: `PIA` 仍未升级到 `paper-aligned`；`SecMI` 当前已判定为 blocked baseline
- `next step`: 保持 `stochastic-dropout = provisional G-1 (repeat-confirmed at GPU512)`；当前已将 `2026-04-10-pia-defense-cost-frontier-stop-decision.md` 固定为 `G1-B = no-go / queue_state = not-requestable`，并转回 `2026-04-10-pia-provenance-upstream-identity-note.md` 的 CPU 补件；`SecMI` 只保留为 blocked baseline
- `last updated`: `2026-04-10`

## 推荐论文

- `2024-iclr-pia-proximal-initialization.pdf`
- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf`
- `2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`
- `2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf`
- `2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf`
- `2026-openreview-mofit-caption-free-membership-inference.pdf`

## 当前主线与 baseline

- 主线：`PIA`
- baseline：`SecMI`

当前判断：

- `PIA` 是当前最成熟、最适合作为“攻击 + 防御”主讲闭环的一条线
- `SecMI` 是当前已明确阻塞、不能继续抢主线资源的 baseline
- `TMIA-DM` 已归档，但当前应作为时间相关噪声信号的灰盒候选论文，而不是黑盒主线

## 当前可执行证据

- `workspaces/gray-box/2026-04-07-pia-runtime-mainline.md`
- `workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- `workspaces/gray-box/pia-intake-gate.md`
- `workspaces/gray-box/assets/pia/manifest.json`
- `workspaces/gray-box/2026-04-08-pia-gpu128-attack-defense.md`
- `workspaces/gray-box/2026-04-08-pia-gpu256-attack-defense.md`
- `workspaces/gray-box/2026-04-08-pia-gpu512-attack-defense.md`
- `workspaces/gray-box/2026-04-08-pia-gpu512-rerun1.md`
- `workspaces/gray-box/2026-04-09-pia-signal-and-cost.md`
- `workspaces/gray-box/2026-04-09-pia-gpu512-adaptive-ablation.md`
- `workspaces/gray-box/2026-04-10-pia-8gb-portability-ladder-execution-packet.md`
- `workspaces/gray-box/2026-04-10-pia-8gb-supporting-frontier-note.md`
- `workspaces/gray-box/2026-04-10-pia-defense-cost-frontier-stop-decision.md`
- `workspaces/gray-box/2026-04-10-pia-provenance-upstream-identity-note.md`
- `workspaces/gray-box/2026-04-09-tmia-dm-intake.md`
- `workspaces/gray-box/2026-04-08-secmi-blocked.md`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-rerun1/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-rerun1/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-256-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-256-allsteps-adaptive/summary.json`
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
- `external/PIA/DDPM/main.py`
- `external/SecMI/score`
- `external/DPDM/utils/util.py`

## 当前推荐执行顺序

1. 固定 `stochastic-dropout = provisional G-1`
2. 以 `GPU512` 为主档，做单旋钮最小消融，不再继续扩大样本规模
3. 为 `PIA` defense pair 补 `FID / IS / LPIPS` 中至少一项可落地 utility 指标，并把 wall-clock 写成正式成本列
4. 复用 [2026-04-09-pia-signal-and-cost.md](2026-04-09-pia-signal-and-cost.md) 作为灰盒主讲线的机理与成本说明
5. 把 repeated-query adaptive review 设成硬门槛，并比较 `off / all_steps / late_steps_only`
6. 补同档 repeat / 多 seed 的最小稳健性说明，而不是换数据集
7. 保持 `SecMI = blocked baseline`
8. 保持 `TMIA-DM` 为研究候选，不提前伪装成可执行主线
9. 复用 [2026-04-09-graybox-signal-axis-note.md](2026-04-09-graybox-signal-axis-note.md) 统一 `PIA / TMIA-DM / SimA / MoFit` 的信号轴叙事
10. 将灰盒主结果接入统一总表并持续复用

## 2026-04-08 新观察

- `PIA baseline` 的 `32` 样本 CPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-cpu-32/summary.json`
  - 结果：`auc=0.782227 / asr=0.765625 / tpr@1%fpr=0.09375`
- `PIA + stochastic-dropout defense` 的 `32` 样本 CPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-cpu-32/summary.json`
  - 结果：`auc=0.769531 / asr=0.75 / tpr@1%fpr=0.09375`
- 本轮顺手修复了 `PIA` runtime mainline summary 的 `runtime.num_samples` 口径 bug：
  - 现在 summary 会记录实际生效样本数，而不是配置中的默认值
- `PIA baseline` 的 `128` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128/summary.json`
  - 结果：`auc=0.817444 / asr=0.765625 / tpr@1%fpr=0.046875 / tpr@0.1%fpr=0.039062`
- `PIA + stochastic-dropout defense` 的 `128` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128/summary.json`
  - 结果：`auc=0.803955 / asr=0.757812 / tpr@1%fpr=0.03125 / tpr@0.1%fpr=0.015625`
- `PIA baseline` 的 `256` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json`
  - 结果：`auc=0.841293 / asr=0.78125 / tpr@1%fpr=0.039062 / tpr@0.1%fpr=0.019531`
- `PIA + stochastic-dropout defense` 的 `256` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/summary.json`
  - 结果：`auc=0.82901 / asr=0.767578 / tpr@1%fpr=0.027344 / tpr@0.1%fpr=0.015625`
- `PIA baseline` 的 `512` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512/summary.json`
  - 结果：`auc=0.841339 / asr=0.786133 / tpr@1%fpr=0.058594 / tpr@0.1%fpr=0.011719`
- `PIA + stochastic-dropout defense` 的 `512` 样本 GPU run 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512/summary.json`
  - 结果：`auc=0.82938 / asr=0.769531 / tpr@1%fpr=0.023438 / tpr@0.1%fpr=0.009766`
- `GPU512` 同档 repeat 已落盘：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-rerun1/summary.json`
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-rerun1/summary.json`
- 当前结论：
  - `stochastic-dropout` 已不再是 runnable hook
  - `GPU128 / GPU256 / GPU512` 三档都低于对应 baseline
  - `GPU512` 同档 repeat 下，baseline 指标保持一致，defense 仍低于配对 baseline
  - 当前 gray-box defense 已可正式写成 `provisional G-1`
  - 仍不能写成 validated privacy win，因为还缺 provenance 升级

## 当前阻塞项

- `PIA` 当前已是 `workspace-verified`，但仍不是 `paper-aligned`
- 当前 gray-box defense 已出现三档 favorable signal，并完成 `GPU512` 同档重复确认，但仍不是 validated privacy win
- `SecMI` 当前 probe 已明确缺真实 `flagfile.txt`

## 2026-04-10 新观察

- `runtime-probe-pia` 与 `runtime-preview-pia` 已在 `cuda:0` 上通过，说明 `RTX4070 8GB` 的真资产路径可直接进入 bounded supporting run
- 新的 `GPU128 adaptive-reviewed` pair 已落盘：
  - baseline `adaptive AUC = 0.817444 / ASR = 0.765625 / wall-clock = 49.544877s`
  - defense `adaptive AUC = 0.806274 / ASR = 0.761719 / wall-clock = 50.667101s`
- 新的 `GPU256 adaptive-reviewed` pair 已落盘：
  - baseline `adaptive AUC = 0.841293 / ASR = 0.78125 / wall-clock = 95.934721s`
  - defense `adaptive AUC = 0.829559 / ASR = 0.763672 / wall-clock = 213.126361s`
- 当前最合理的 portability frontier 结论是：
  - `GPU128 = quickest portable pair`
  - `GPU256 = decision rung with cost warning`
  - `GPU512` 当前不值得再做 mechanical rerun
- 当前 `G1-B / PIA defense-cost frontier` 已正式收口为：
  - `no-go`
  - `queue_state = not-requestable`
  - 只有在新的 low-cost hypothesis/budget note 被写清后才允许重审 GPU

## 当前最短路径

1. 固定 `stochastic-dropout = provisional G-1 (repeat-confirmed at GPU512)`
2. 新增结构化 `quality/cost` 支撑，并把 adaptive repeated-query review 与 summary 一起落盘
3. 用单旋钮消融解释 defense 如何削弱 `epsilon-trajectory consistency`
4. 保持 [2026-04-09-pia-signal-and-cost.md](2026-04-09-pia-signal-and-cost.md) 与状态页一致
5. 用 [2026-04-09-graybox-signal-axis-note.md](2026-04-09-graybox-signal-axis-note.md) 统一灰盒文献叙事
6. 保持 `SecMI = blocked baseline`
7. 复用 [unified table](../implementation/2026-04-08-unified-attack-defense-table.md) 作为灰盒对外引用入口
8. 只有在真实 `flagfile + checkpoint root` 到位后，才恢复 `SecMI`
