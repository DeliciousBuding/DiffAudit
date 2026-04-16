# 灰盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 部分中间信息、条件相关评分、噪声预测与结构特征下的成员推断
- `status`: `PIA real-asset runtime-mainline ready; GPU128/GPU256/GPU512 baseline + defended pairs landed; GPU512 rerun confirmed; GPU128/GPU256 adaptive portability pair landed on RTX4070 8GB; provisional G-1 established; SecMI full-split corroboration landed; PIA-vs-SecMI disagreement verdict landed; TMIA-DM late-window + temporal-striding(stride=2) is now the strongest defended gray-box challenger reference; Noise as a Probe is a strengthened bounded challenger candidate; MoFit is now current-contract hold; CDI contract review and first internal canary both landed; PIA 2048 shared-score surface landed; repaired SecMI 2048 paired surface now also landed and recovers the old strong paired regime; CDI paired-feature scorer design now also landed; CDI paired scorer boundary is now frozen to default-internal-only; summary-layer wording is now synchronized; default-run policy is now frozen; machine-readable contract is now emitted; gray-box current gpu question = none`
- `blocked by`: `PIA` 仍未升级到 `paper-aligned`；`CDI` paired scorer 的 machine-readable contract 已经存在，但还没有补一份更高层 consumer handoff note，来明确 `Leader / materials / future Platform-Runtime consumer` 各自该看哪些字段、忽略哪些字段；当前 `SimA` feasibility 与 later-timestep rescan 虽都可执行但仍明显偏弱；`structural memorization` 当前 local faithful approximation 也已落成 `negative but useful`；`Noise as a Probe` 在当前 local `SD1.5` 合同上没有 honest defended-extension gate；`MoFit` 在当前 local contract 下只给出 tiny weak-positive gap`
- `next step`: 保持 `PIA + stochastic-dropout(all_steps)` 为 admitted defended headline；保持 `TMIA-DM late-window + temporal-striding(stride=2)` 为 strongest defended challenger reference；保持 `Noise as a Probe` 为 strengthened bounded challenger candidate；保持 `MoFit` 为 `current-contract hold`；当前下一步转到 `CDI paired-scorer consumer handoff note`，把 `Leader / materials / future consumer` 应消费的字段优先级和禁止过度解读的规则写成短交接说明`
- `last updated`: `2026-04-16`

## 推荐论文

- `2024-iclr-pia-proximal-initialization.pdf`
- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf`
- `2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf`
- `2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf`
- `2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf`
- `2026-openreview-mofit-caption-free-membership-inference.pdf`
- `2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf`

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
- `workspaces/gray-box/2026-04-16-mofit-scaffold-schema-decision.md`
- `workspaces/gray-box/2026-04-16-mofit-scaffold-implementation-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-record-schema-integration-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-score-trace-update-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-optimization-helper-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-latent-loss-contract-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-real-target-path-wiring-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-sample-level-execution-assembly-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-script-level-canary-execution-review.md`
- `workspaces/gray-box/2026-04-16-mofit-script-level-canary-implementation-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-real-asset-canary-launch-gate-review.md`
- `workspaces/gray-box/2026-04-16-mofit-launch-budget-tightening-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-fresh-real-asset-canary-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-canary-score-shape-review.md`
- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-design.md`
- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-execution-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-score-review.md`
- `workspaces/gray-box/2026-04-16-mofit-final-cpu-reviewrung-verdict.md`
- `workspaces/gray-box/2026-04-16-mofit-current-contract-hold-verdict.md`
- `workspaces/gray-box/2026-04-16-post-mofit-next-family-reselection.md`
- `workspaces/gray-box/2026-04-16-cdi-protocol-asset-contract.md`
- `workspaces/gray-box/2026-04-16-cdi-feature-collection-surface-review.md`
- `workspaces/gray-box/2026-04-16-cdi-internal-canary-verdict.md`
- `workspaces/gray-box/2026-04-16-pia-2048-cdi-rung-runtime-health-review.md`
- `workspaces/gray-box/2026-04-16-pia-2048-cdi-rung-verdict.md`
- `workspaces/gray-box/2026-04-16-secmi-pia-2048-paired-surface-verdict.md`
- `workspaces/gray-box/2026-04-16-cdi-paired-feature-extension-review.md`
- `workspaces/gray-box/2026-04-16-cdi-paired-surface-mismatch-review.md`
- `workspaces/gray-box/2026-04-16-secmi-paired-surface-repair-contract-review.md`
- `workspaces/gray-box/2026-04-16-secmi-pia-2048-repaired-paired-surface-verdict.md`
- `workspaces/gray-box/2026-04-16-cdi-paired-feature-repromotion-review.md`
- `workspaces/gray-box/2026-04-16-cdi-paired-feature-scorer-design.md`
- `workspaces/gray-box/2026-04-16-cdi-paired-scorer-boundary-review.md`
- `workspaces/gray-box/runs/cdi-internal-canary-20260416-r1/audit_summary.json`
- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1/summary.json`
- `workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/summary.json`
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
7. `MoFit` 当前最终 verdict 已固定为 `execution-positive but signal-weak / current-contract hold`
8. `MoFit` 只有在 target-family、caption/bootstrap、surrogate objective、embedding objective 或 score definition 发生实质变化时，才允许重开
9. 当前下一条 live lane 已重选为 `CDI`
10. `CDI` 当前应写成 `gray-box collection-level audit extension`，而不是单样本 MIA 替代线
11. `CDI` 的第一步必须保持 `CPU-only + contract-first`，并冻结 `P/U/control/test` collection schema、首个可复用 score source 与 Welch-`t` audit summary contract
12. `CDI` 的 honest local surface 当前固定为共享 local `CIFAR-10 DDPM` score contract，不应一开始跳到 `SD1.5`、latent-diffusion 或 white-box-only feature path
13. `CDI` 的 first bounded canary 已冻结为：现有 `1024 / 1024` shared surface 上的 deterministic `512 / 512` control-test partition + `SecMI stat only`
14. paired `PIA + SecMI` 当前保留为下一条 extension，而不是 first canary 默认值
15. 当前最有价值的 GPU follow-up 不是重开旧 family，而是把 `PIA` 的 shared-score surface 扩大，供 `CDI` 的 paired-method follow-up 直接消费
16. `CDI` 的 first internal canary 已经真实落盘，并且在 memberness orientation 显式归一后给出强同向 Welch statistic；这证明当前 lane 已经从 contract truth 升级到 execution truth
17. 活跃中的 `PIA 2048` GPU rung 目前仍在真实计算，但已超过最简单线性时长预期且尚未吐出首批 artifact；因此当前策略不是无限等待，而是保留到显式 runtime-health cap 后再决定继续或止损
18. `PIA 2048` rung 现已完成并被固定为 `positive but cost-heavy`：它足以作为 `CDI` paired follow-up 的新 `PIA` surface，但不足以单独再放行更大的同家族 `PIA` 扩档
19. `SecMI 2048` paired-surface export 已执行成功，但结果是 `mixed but useful`：当前 widened paired surface 并不稳定 enough for immediate paired `CDI` promotion，因此下一步应先 review mismatch，再决定是保留 `1024`、修正 `2048`，还是另建 paired contract
20. `CDI paired-surface mismatch review` 已将当前主嫌疑收敛到 `export/config drift`，其中最明确的是新 `2048` export 使用了 `t_sec = 20`，而不是已承认 `SecMI` mainline 与旧强 `1024` paired packet 所使用的 `t_sec = 100`；因此当前不应把这次弱化直接写成 `SecMI` 的真实 scale collapse
21. `SecMI paired-surface repair contract review` 已完成并放行一个有界 GPU 问题：修复后的 paired export 必须回到 admitted `SecMI stat` 合同（`t_sec = 100 / timestep = 10 / batch_size = 64 / canonical split root = external/SecMI/mia_evals/member_splits`），而且 `SecMI` 与 `PIA` 的 CIFAR-10 half-split 文件已验证为字节级一致，因此下一步可以诚实地重跑一次 `2048` paired surface
22. 修复合同后的 `SecMI 2048` paired rerun 已真实落盘，并把 paired surface 拉回到旧强 `1024` 的同一量级（`AUC = 0.876912 / combined Spearman = 0.906879 / disagreement = 0.121582`）；因此旧弱 `r2` packet 只保留为 drift-history，而 paired `PIA + SecMI` feature promotion 现在可以重新打开，但下一步应先做 scorer design 而不是继续烧 GPU
23. `CDI paired-feature scorer design` 已完成并在 repaired `2048` surface 上给出正向 bounded canary：当前冻结 scorer 为 control-fitted `control-z-linear`（`SecMI stat` 权重 `0.526839`，`PIA` 权重 `0.473161`），paired `t = 30.027926` 略高于单独 `SecMI` 的 `29.637878`；因此下一步不是新 GPU，而是 paired scorer 的 boundary review
24. `CDI paired-scorer boundary review` 已完成：paired scorer 在反向 half-split 上仍然强正向，且权重稳定在 `SecMI ~0.52 / PIA ~0.48`；但它并不稳定压过 `SecMI`，所以当前最诚实的边界是 `default internal paired scorer on the repaired 2048 surface`，而不是 headline scorer 或外部证据
25. `CDI paired-scorer default-run policy note` 已完成：默认 CLI 策略现在是 `--paired-scorer auto`，即 paired inputs 存在时自动启用 `control-z-linear`，否则退回 component-only；且 paired runs 必须继续并报 `paired + SecMI + PIA` 三项统计，因此当前下一步不是新 GPU，而是把这些字段要求固定成 machine-readable contract
26. `CDI paired-scorer machine-readable contract note` 已完成：新的 paired canary artifact 已显式写出 `contract.name / version / feature_mode / paired_scorer_policy_requested / paired_scorer_policy_effective / component_reporting_required / headline_use_allowed / external_evidence_allowed`，因此更高层消费者现在可以直接读 contract 字段而不是回推 Markdown 规则

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
