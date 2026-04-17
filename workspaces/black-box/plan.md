# 黑盒方向计划

## 状态面板

- `owner`: active-thread
- `scope`: 统一黑/灰/白研究规划下的第一优先执行线，负责黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中，`recon` 的 black-box 主证据、最佳单指标参考和 `variation` 次主线口径已冻结；`semantic-auxiliary-classifier` 已落成当前 leading new-family challenger；`CLiD` 已收紧到 `evaluator-near local clip-only corroboration`；`served-image-sanitization` 已记为 mitigation `no-go`；`X-61` 又进一步确认 paper-backed backlog 里没有新的 promotable family，因此黑盒当前重新回到 `stable but not innovation-leading`
- `blocked by`: `recon` 公开资产包（DOI: `10.5281/zenodo.13371475`）的语义 gate 现已 machine-audited 到 `proxy-shadow-member`，但仍未升级到 paper-aligned；`CLiD` 仍缺 shadow-side evaluator assets；`variation` 真实 `query_image_root` 仍缺；`semantic-auxiliary-classifier` 当前也还没有 genuinely new feature family；`Kandinsky 10/10` 当前本机链路仍异常慢；剩余 face-image LDM 论文路线当前又落在 domain-specific collection-level risk note，且缺 bounded local execution contract
- `next command`: keep black-box wording fixed to `Recon / semantic-auxiliary-classifier / CLiD / variation` and do not promote a GPU question; `X-61` already closed the remaining paper-backed scouting surface as `negative but useful`, so black-box should now yield the live slot back to the cross-box control plane while only reopening on a genuinely distinct bounded family or real asset/boundary shift
- `last updated`: 2026-04-17

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
- `workspaces/black-box/2026-04-16-post-second-signal-blackbox-next-question-review.md`
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

1. 固定 `Recon = headline / best project-level existence-proof package`
2. 固定 `semantic-auxiliary-classifier = leading black-box challenger`，但不把它误写成 `Recon` replacement
3. 固定 `CLiD = evaluator-near local clip-only corroboration`，不把它提前写成 paper-aligned benchmark
4. 固定 `variation = contract-ready blocked`，在真实资产到位前只允许保持 blocked contract，不开结果型 run
5. 固定黑盒当前状态为 `no-new-gpu-question`，直到出现 genuinely new feature family 或真实 asset/boundary change
6. 复用 `audit-recon-public-bundle` 与 [2026-04-09-recon-public-bundle-audit.md](2026-04-09-recon-public-bundle-audit.md) 持续跟踪 `recon` 语义 gate，并强制带上 `proxy-shadow-member` 限制
7. 维持黑盒状态文档、实验目录和主线命令说明同步，并明确这些同步属于统一三线规划下的黑盒执行层收口
8. 把 `B-1 / B-2` 只记录为 black-box defense backlog，不提前写成已有可比较结果

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
- `served-image-sanitization` 的第一条黑盒缓解 probe 也已经收口：
  - `CLiD` sanitized probe 与 frozen baseline 都保持 `AUC = 1.0 / ASR = 1.0 / TPR@1%FPR = 1.0`
  - 但 utility check 仍显示 mild perturbation（`mean PSNR = 38.286 dB`）
  - 所以这条黑盒 mitigation 当前是 `negative but useful no-go`

## 当前阻塞项

- 公开 `recon` checkpoint 与 dataset 已在本机落地，运行时语义链现在已被 machine-audited，但仍停在 `proxy-shadow-member`
- 仍缺与论文一致的 target/shadow/member/non-member 直接映射说明
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
- `variation` 当前只有本地 synthetic smoke，与正式 blocked 的真实 API 资产探针；没有 query image root 就不能继续往真实 black-box 推进
- `variation` 的恢复条件现在已经 contract-ready：
  - 第一硬门槛仍是 `query_image_root / query images`
  - endpoint/proxy、budget 和固定参数则是复开前必须补齐的后续门槛
- `TMIA-DM` 已确认不是严格黑盒，如果口径管理不严，后续汇报很容易把它误归进黑盒
- `2026-04-17` 的 black-box candidate refresh review 也已收口：
  - 当前看得到的候选要么还是 `semantic-aux` 同家族延展，要么是 `CLiD` 边界升级、`variation` 资产复开，或者已被 gray-box `CDI` 吸收的集合审计方向
  - 因此黑盒当前仍没有 honest ready next-family promotion candidate
- `2026-04-17` 的 `X-61` paper-backed scoping review 又进一步确认：
  - 剩余的 `face-image LDM` 论文路线更像 domain-specific collection-level proxy-distribution audit
  - 它与 `semantic-auxiliary-classifier` 和 gray-box `CDI` 都存在结构重叠
  - 当前不能作为新的 black-box live family promotion
