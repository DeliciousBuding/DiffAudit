# 黑盒方向计划

## 状态面板

- `owner`: active-thread
- `scope`: 统一黑/灰/白研究规划下的第一优先执行线，负责黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中，`recon` 的 black-box 主证据、最佳单指标参考和 `variation` 次主线口径已冻结；`semantic-auxiliary-classifier` 已落成当前 leading new-family challenger；当前 `variation` 的真实 API 资产 probe 已正式确认 blocked；`TMIA-DM` 已完成 intake，但被判定为灰盒候选而非黑盒主线
- `blocked by`: `recon` 公开资产包（DOI: `10.5281/zenodo.13371475`）的语义 gate 现已 machine-audited 到 `proxy-shadow-member`，但仍未升级到 paper-aligned；`variation` 真实 query image root 仍缺；`Kandinsky 10/10` 当前本机链路仍异常慢
- `next command`: no immediate black-box rerun; keep `Recon / CLiD / semantic-auxiliary-classifier` wording consistent across `blackbox-status`, `reproduction-status`, `comprehensive-progress`, and `ROADMAP`; turn `variation` recovery into an explicit asset contract instead of speculative reruns
- `next command`: no immediate black-box rerun; keep `Recon / CLiD / semantic-auxiliary-classifier` wording consistent across `blackbox-status`, `reproduction-status`, `comprehensive-progress`, and `ROADMAP`; treat `semantic-aux` multi-feature scoring as `negative but useful` unless a new feature family appears; turn `variation` recovery into an explicit asset contract instead of speculative reruns
- `last updated`: 2026-04-16

## 统一规划定位

- 该工作区只承接黑盒执行层，不负责替代灰盒或白盒计划
- 黑盒、灰盒、白盒已被纳入统一路线图，但当前资源分层仍以黑盒为先
- 本文件的职责是把黑盒主线写清楚，避免被误读成“仓库只做黑盒”
- 灰盒与白盒的推进状态以 `docs/reproduction-status.md` 和 `ROADMAP.md` 的统一口径为准
- 当前算法主讲闭环不是黑盒，而是灰盒 `PIA`；黑盒当前更适合作为“风险已被验证”的主证据线

## 主论文与场景

- `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`
- `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf`
- `2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf`
- `2026-04-09-blackbox-method-boundary.md`

当前约束：

- `TMIA-DM` 只作为 black-box threat-model 边界纠偏材料引用
- 不进入黑盒主论文或黑盒执行候选层级

## 当前可执行证据

- `experiments/clid-dry-run-smoke/summary.json`
- `experiments/clid-artifact-summary/summary.json`
- `experiments/recon-eval-smoke/summary.json`
- `experiments/recon-mainline-smoke/summary.json`
- `experiments/recon-artifact-summary/summary.json`
- `experiments/recon-upstream-eval-smoke/summary.json`
- `experiments/recon-runtime-mainline-ddim-smoke/summary.json`
- `experiments/recon-runtime-mainline-ddim-smoke/artifact-mainline-final/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-10-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-10-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-25-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-25-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
- `experiments/recon-runtime-mainline-kandinsky-public-smoke/summary.json`
- `experiments/recon-runtime-mainline-kandinsky-public-smoke/artifact-mainline/summary.json`
- `experiments/variation-synth-smoke/summary.json`
- `experiments/variation-synth-smoke-local-20260408/summary.json`
- `workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
- `workspaces/black-box/runs/semantic-aux-classifier-probe-20260415-r1/summary.json`
- `workspaces/black-box/runs/semantic-aux-classifier-comparator-20260415-r1/summary.json`
- `workspaces/black-box/runs/semantic-aux-classifier-comparator-20260416-r2/summary.json`
- `workspaces/black-box/2026-04-09-recon-evidence-freeze.md`
- `workspaces/black-box/2026-04-09-recon-public-bundle-audit.md`
- `experiments/blackbox-status/summary.json`
- `experiments/dit-sample-smoke/summary.json`
- `experiments/dit-sample-step10/summary.json`
- `experiments/dit-sample-step50/summary.json`

## 本地代码上下文

- `external/CLiD`
- `external/Reconstruction-based-Attack`
- 公开资产包：`https://doi.org/10.5281/zenodo.13371475`
- 语义映射说明：`docs/recon-public-asset-mapping.md`

## 推荐分工

- 主复现：`Black-box Membership Inference Attacks against Fine-tuned Diffusion Models`
- 对照参考：`Towards Black-Box Membership Inference Attack for Diffusion Models`
- 场景补充：`CLiD`
- 审计扩展：`CDI`
- 细分场景补充：`Membership Inference Attacks for Face Images Against Fine-tuned Latent Diffusion Models`

## 一周行动清单

1. 把 `main evidence / best single metric reference / secondary track` 三层口径固定到所有状态文档
2. 复用 `audit-recon-public-bundle` 与 [2026-04-09-recon-public-bundle-audit.md](2026-04-09-recon-public-bundle-audit.md) 持续跟踪本机 `recon` 公开 Zenodo 资产的 target/shadow/member/non-member 语义 gate
3. 统一 `recon DDIM public-100 step30` 的固定话术，并强制带上 `proxy-shadow-member` 限制
4. 对比 `DDIM public-100 step10` / `step30` 指标差异，并记录运行成本
5. 把真实 target/shadow score artifact 的命名和目录约束落实到 `recon` 主线
6. 把 `variation` 的恢复条件写成资产契约包：`query_image_root`、query images、endpoint/proxy、query budget
7. 将 `semantic-auxiliary-classifier` 固定为 current challenger，不把它误写成 `Recon` replacement
8. 在真实资产到位前，`variation` 只允许继续做 probe，不再做结果型 run
9. `kandinsky_v22` public smoke 已通，但 `10/10` 与单样本直跑当前都异常慢；在拿到有效日志前继续暂停 `Kandinsky`
10. 评估 `CLiD` 的真实 text-to-image 资产是否可在当前机器上最小复现
11. 保持 `TMIA-DM` 只作为灰盒候选论文，不写进黑盒执行层级
12. 维持黑盒状态文档、实验目录和主线命令说明同步，并明确这些同步属于统一三线规划下的黑盒执行层收口
13. 把 `B-1 / B-2` 只记录为 black-box defense backlog，不提前写成已有可比较结果

## 2026-04-16 新观察

- `semantic-aux-classifier-comparator-20260416-r2` 已在相同协议下完成 `32 / 32` 放大量级 comparator：
  - `AUC = 0.90918`
  - `ASR = 0.84375`
  - `TPR@1%FPR = 0.25`
- 对现有 `semantic-auxiliary-classifier` 的评分层又做了 bounded review：
  - `16 / 16` 上 logistic `AUC = 0.910156`，但单一 `mean_cos = 0.945312`
  - `32 / 32` 上 logistic `AUC = 0.90625`，但单一 `mean_cos = 0.916992`
  - `Spearman(logistic, mean_cos) = 0.973607 / 0.978709`
- 该结果与上一档 `16 / 16` comparator (`AUC = 0.910156`) 基本同向稳定，没有出现放大后信号塌缩
- 当前结论：
  - `semantic-auxiliary-classifier` 仍是 black-box leading challenger
  - 当前多特征 logistic 校准没有带来新的排序信息，`mean_cos` 已经捕获了这条线的大部分黑盒信号
  - 这条线适合继续做 bounded hypothesis，而不是机械扩样本

## 当前阻塞项

- 公开 `recon` checkpoint 与 dataset 已在本机落地，运行时语义链现在已被 machine-audited，但仍停在 `proxy-shadow-member`
- 仍缺与论文一致的 target/shadow/member/non-member 直接映射说明
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
- `variation` 当前只有本地 synthetic smoke，与正式 blocked 的真实 API 资产探针；没有 query image root 就不能继续往真实 black-box 推进
- `TMIA-DM` 已确认不是严格黑盒，如果口径管理不严，后续汇报很容易把它误归进黑盒
