# 黑盒方向计划

## 状态面板

- `owner`: active-thread
- `scope`: 黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中，`Stable Diffusion + DDIM` 最小真实 runtime-mainline 已通，`DiT` 官方 sample smoke 已通
- `blocked by`: `recon` 公开资产包（DOI: `10.5281/zenodo.13371475`）的 target/shadow/member/non-member 论文语义仍需核准，并且当前只验证了极小子集
- `next command`: `conda run -n diffaudit-research python -m diffaudit run-recon-runtime-mainline --target-member-dataset path/to/target_member_dataset.pkl --target-nonmember-dataset path/to/target_nonmember_dataset.pkl --shadow-member-dataset path/to/shadow_member_dataset.pkl --shadow-nonmember-dataset path/to/shadow_nonmember_dataset.pkl --target-model-dir path/to/target_lora_checkpoint --shadow-model-dir path/to/shadow_lora_checkpoint --workspace experiments/recon-runtime-mainline --repo-root external/Reconstruction-based-Attack --scheduler ddim --method threshold`
- `last updated`: 2026-04-06

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
- `experiments/variation-synth-smoke/summary.json`
- `experiments/blackbox-status/summary.json`
- `experiments/dit-sample-smoke/summary.json`

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
3. 将 `recon` 从 `1-sample` 极小子集扩到更有意义的公开样本规模，并记录运行成本
4. 把真实 target/shadow score artifact 的命名和目录约束落实到 `recon` 主线
5. 维持 `DiT` 官方 `sample.py` 路线可重复执行，并视需要补本地 checkpoint 路径
6. `kandinsky_v22` 后端接口已接入；已下载一对公开 `decoder/prior` LoRA 权重并启动真实 smoke，当前等待上游 `kandinsky2_2_inference.py` 给出终态
7. 评估 `variation` 真实 API 调用所需的凭据、预算和 query image 约束
8. 评估 `CLiD` 的真实 text-to-image 资产是否可在当前机器上最小复现
9. 维持黑盒状态文档、实验目录和主线命令说明同步

## 当前阻塞项

- 公开 `recon` checkpoint 与 dataset 已在本机落地，但运行时语义映射尚未核准
- 仍缺与论文一致的 target/shadow/member/non-member 直接映射说明
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
