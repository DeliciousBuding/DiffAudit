# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件相关评分、噪声预测与结构特征下的成员推断
- `status`: `PIA real-asset runtime-mainline ready; GPU128/GPU256/GPU512 baseline + defended pairs landed; GPU512 rerun confirmed; GPU128/GPU256 adaptive portability pair landed on RTX4070 8GB; provisional G-1 established; SecMI full-split corroboration landed; PIA-vs-SecMI disagreement verdict landed; TMIA-DM late-window + temporal-striding(stride=2) is now the strongest defended gray-box challenger reference; Noise as a Probe is a strengthened bounded challenger candidate; gray-box current gpu question = none`
- `blocked by`: `PIA` 仍未升级到 `paper-aligned`；`PIA + SecMI` 还没有 promotion-worthy 的 fusion story；当前 `SimA` feasibility 与 later-timestep rescan 虽都可执行但仍明显偏弱；`structural memorization` 当前 local faithful approximation 也已落成 `negative but useful`；`Noise as a Probe` 在当前 local `SD1.5` 合同上没有 honest defended-extension gate；新的 `MoFit` 已被重选为下一 genuinely new mechanism，但当前仍缺 surrogate optimization、fitted-embedding optimization、artifact schema，以及 dedicated scaffold`
- `next step`: 保持 `PIA + stochastic-dropout(all_steps)` 为 admitted defended headline；保持 `TMIA-DM late-window + temporal-striding(stride=2)` 为 strongest defended challenger reference；保持 `Noise as a Probe` 为 strengthened bounded challenger candidate；灰盒当前不再重开已收口分支，而是把下一条 live CPU-first lane 切到 `MoFit dedicated scaffold / schema decision`；在 surrogate path、fitted-embedding path、schema 与 scaffold 冻结前，`gpu_release` 继续保持 `none``
- `last updated`: `2026-04-16`

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
- `SecMI` 已完成 full-split local execution，当前更适合作为独立 corroboration line，而不是 blocked placeholder
- `TMIA-DM` 当前已升级为 strongest packaged gray-box challenger，应写成灰盒时间相关噪声信号主候选，而不是 intake-only 占位或黑盒主线

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
- `workspaces/gray-box/2026-04-15-pia-vs-secmi-graybox-comparison.md`
- `workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-challenger-boundary-review.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-defended-extension-feasibility-review.md`
- `workspaces/gray-box/2026-04-16-post-temporal-striding-graybox-next-question-review.md`
- `workspaces/gray-box/2026-04-16-post-noise-next-family-reselection.md`
- `workspaces/gray-box/2026-04-16-mofit-protocol-asset-contract.md`
- `workspaces/gray-box/2026-04-16-mofit-implementation-surface-review.md`
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
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
- `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- `workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/summary.json`
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

1. 固定 `PIA + stochastic-dropout(all_steps)` 为 admitted defended headline，不再回退成单纯“provisional but unstable”
2. 固定 `TMIA-DM late-window + temporal-striding(stride=2)` 为 strongest defended gray-box challenger reference，不再继续机械补 rung
3. 固定 `Noise as a Probe` 为 `strengthened bounded challenger candidate`，但不把它提前包装成 active challenger
4. 将灰盒当前状态明确写成 `no-new-gpu-question`，直到出现 genuinely new mechanism 或真实 contract shift
5. 把 `SecMI` 固定为 `independent corroboration line`，把 `PIA vs SecMI` 固定为 `naive fusion = no-go`
6. 复用 [2026-04-09-pia-signal-and-cost.md](2026-04-09-pia-signal-and-cost.md) 与 [2026-04-09-graybox-signal-axis-note.md](2026-04-09-graybox-signal-axis-note.md) 维持灰盒主讲线机理与信号轴叙事
7. 当前下一条 genuinely new family 已重选为 `MoFit`，先做 `protocol / asset contract` 与 implementation-surface review，不提前开 smoke
8. `MoFit` 当前实现层结论是：必须走 dedicated scaffold，而不是复用 `structural memorization` 或 `semantic-aux` 脚本
9. 在 `MoFit` 的 surrogate path、fitted-embedding path、artifact schema 与 dedicated scaffold 冻结前，继续保持 `gpu_release = none`
10. 如果 `MoFit` 也被证明不具备 honest bounded entry，再切去别的 lane 处理更高价值问题

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
- `SecMI` 不再是资产 blocker，但当前仍没有 defended comparator，也不值得直接和 `PIA` 做 naive fusion 升级

## 2026-04-15 新观察

- `SecMI` 的 full-split local rung 已落盘：
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
  - `stat AUC = 0.885833 / NNS AUC = 0.946286`
- `PIA` 的 `1024 / 1024` adaptive-reviewed pair 已落盘：
  - baseline `AUC = 0.838630 / ASR = 0.782715`
  - defense `AUC = 0.825966 / ASR = 0.770508`
- 新的 `PIA vs SecMI` disagreement run 已落盘：
  - `workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/summary.json`
  - `Spearman = 0.907588 / disagreement = 0.122559 / ensemble AUC = 0.868736`
- 当前结论：
  - `SecMI` 已升级为灰盒独立 corroboration line
  - `PIA` 仍是 defended gray-box mainline
  - `PIA + SecMI` 的简单融合不值得升格为当前 gray-box 新分支
  - `epsilon-output-noise (std = 0.1)` 在 `cpu-32` bounded smoke 上也未能压低攻击，因此不应作为下一个 GPU defended candidate
  - `input-gaussian-blur (sigma = 1.0)` 在 `cpu-32` bounded smoke 上进一步放大了攻击，因此也不应作为下一个 GPU defended candidate
- `SimA` 已在当前 DDPM asset line 上完成 bounded CPU feasibility 与 later-timestep rescan：
  - 初始最优 `AUC = 0.542969 @ t=120`
  - rescan 最优 `AUC = 0.584961 @ t=160`
  - 仍不足以升级为 challenger 或 GPU 题
- `structural memorization` 已在当前 CelebA target-family faithful approximation 上完成 bounded smoke，但结果方向为负：
  - `AUC = 0.375 / ASR = 0.53125`
  - member mean `SSIM = 0.730527` 低于 non-member mean `0.750170`
  - 当前不应升级为 active gray-box family 或下一条 GPU 题

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
6. 保持 `SecMI = corroboration line`，不回退到 blocked wording
7. 复用 [unified table](../implementation/2026-04-08-unified-attack-defense-table.md) 作为灰盒对外引用入口
8. 不再为 naive `PIA + SecMI` ensemble 追加预算，除非先写出新的 gating hypothesis
9. 不把 `epsilon-output-noise` 或 `input-gaussian-blur` 重开成 GPU 题；下一候选必须明显区别于小幅输入/输出扰动
10. 若 `G-2 distillation` 仍无正式训练/评估链，则灰盒下一活跃任务应先转到 `GB-3` 新 family 选择，而不是继续机械扩第二防御小 smoke
11. 当前 `GB-3 / SimA` 已完成 bounded CPU feasibility 与 later-timestep rescan，但结论仍是 strength-negative；`structural memorization` 也已完成 local faithful-approximation smoke，但结论是 direction-negative；下一灰盒活跃任务已重选为 `TMIA-DM protocol / asset decomposition`
12. 当前 `TMIA-DM` 已完成最小 `protocol probe`、一次 repeat、same-split 比较、late-window refine、两次正向 `GPU128` rung，以及两次正向 `GPU256` rung；结论是该线已成为稳定 GPU challenger，且 `late_steps_only` 与 `timestep-jitter` defense ablation 都已证伪为弱解，下一条任务应是材料化总表或新的 defense hypothesis
13. 新的 `TMIA-DM late-window temporal-striding(stride=2)` 已完成两个 `cpu-32` repeat，`AUC` 从 `0.823242 / 0.760742` 降到 `0.697266 / 0.696289`；结论是它成为当前最值得过 `GPU128` gate 的 challenger-specific 新防御假设，但还不能直接写成第二 defended comparator
14. `TMIA-DM late-window temporal-striding(stride=2)` 已完成两个 `GPU128` repeat，`AUC` 进一步压到 `0.727234 / 0.711609`；结论是它已成为当前最强的 `TMIA-DM`-specific defended candidate，下一门槛是单个 `GPU256` scale rung，而不是重新回到 defense shortlist
15. `TMIA-DM late-window temporal-striding(stride=2)` 已完成两个 `GPU256` repeat，`AUC` 维持在 `0.733322 / 0.7173`；结论是它已成为 repeat-confirmed 的 scale-positive defended candidate，下一门槛不再是更多盲目 rung，而是 defended operating-point comparison 与系统层摘要同步审查
16. defended operating-point comparison 与统一总表 sync 已完成；结论是 `TMIA + temporal-striding` 现在应取代 `TMIA + dropout`，成为更高层 gray-box defended challenger 摘要里的首选 defended reference
