# 黑盒方向计划

## 状态面板

- `owner`: active-thread
- `scope`: 统一黑/灰/白研究规划下的第一优先执行线，负责黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中，`recon` 是当前最强 black-box evidence line；`variation` 已在本地 CPU 上再次通过 synthetic smoke（`experiments/variation-synth-smoke-local-20260408/summary.json`）；`DDIM public-100 step30` 已完成收口并进入 `blackbox-status` 主证据，但当前 black-box 防御仍未正式落地
- `blocked by`: `recon` 公开资产包（DOI: `10.5281/zenodo.13371475`）的 target/shadow/member/non-member 论文语义仍需核准；`Kandinsky 10/10` 当前本机链路仍异常慢
- `next command`: `conda run -n diffaudit-research python -m diffaudit summarize-blackbox-results --experiments-root experiments --workspace experiments/blackbox-status`
- `last updated`: 2026-04-07

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

1. 保持 `recon` 统一 mainline smoke 可重复执行，并在拿到 score artifact 后先跑 `probe-recon-score-artifacts` 再转到 `run-recon-artifact-mainline`
2. 用 `probe-recon-runtime-assets` 先核准本机 `recon` 公开 Zenodo 资产的 target/shadow/member/non-member 映射
3. 对比 `DDIM public-100 step10` / `step30` 指标差异，并记录运行成本
4. 把真实 target/shadow score artifact 的命名和目录约束落实到 `recon` 主线
5. 维持 `DiT` 官方 `sample.py` 路线可重复执行，并把本地 checkpoint 驱动的 `step50` 证据视需要继续往更高步数或更高分辨率推进
6. `kandinsky_v22` public smoke 已通，但 `10/10` 与单样本直跑当前都异常慢；`Stable Diffusion + DDIM` 的 `100-sample public` 也已通；`DiT step50` 也已补上，下一步是解释 `DDIM step10` / `step30` 的指标差异，并在拿到有效日志前继续暂停 `Kandinsky`
7. 评估 `variation` 真实 API 调用所需的凭据、预算和 query image 约束
8. 把 `variation` 明确写成“可本地重复验证的第二黑盒候选线”，方便申报阶段引用
9. 评估 `CLiD` 的真实 text-to-image 资产是否可在当前机器上最小复现
10. 维持黑盒状态文档、实验目录和主线命令说明同步，并明确这些同步属于统一三线规划下的黑盒执行层收口
11. 把 `B-1 / B-2` 只记录为 black-box defense backlog，不提前写成已有可比较结果

## 当前阻塞项

- 公开 `recon` checkpoint 与 dataset 已在本机落地，但运行时语义映射尚未核准
- 仍缺与论文一致的 target/shadow/member/non-member 直接映射说明
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
