# 黑盒方向计划

## 状态面板

- `owner`: active-thread
- `scope`: 黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中，`Stable Diffusion + DDIM` 的 `100-sample public runtime-mainline` 已通，`kandinsky_v22` 最小真实 runtime-mainline 已通，`DiT` 官方 `step50 sample-smoke` 已通
- `blocked by`: `recon` 公开资产包（DOI: `10.5281/zenodo.13371475`）的 target/shadow/member/non-member 论文语义仍需核准；当前公开主线已经推进到 `100-sample` 上限，但 `public-50` 与 `public-100` 的指标差异还需要解释
- `next command`: `conda run -n diffaudit-research python -m diffaudit probe-recon-runtime-assets --target-member-dataset external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-smoke/target_member.pt --target-nonmember-dataset external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-smoke/target_non_member.pt --shadow-member-dataset external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-smoke/shadow_member.pt --shadow-nonmember-dataset external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-smoke/shadow_non_member.pt --target-model-dir external/recon-assets/public-kandinsky-pokemon/decoder/pytorch_lora_weights.safetensors --shadow-model-dir external/recon-assets/public-kandinsky-pokemon/decoder/pytorch_lora_weights.safetensors --backend kandinsky_v22 --target-decoder-dir external/recon-assets/public-kandinsky-pokemon/decoder/pytorch_lora_weights.safetensors --target-prior-dir external/recon-assets/public-kandinsky-pokemon/prior/pytorch_lora_weights.safetensors --shadow-decoder-dir external/recon-assets/public-kandinsky-pokemon/decoder/pytorch_lora_weights.safetensors --shadow-prior-dir external/recon-assets/public-kandinsky-pokemon/prior/pytorch_lora_weights.safetensors --repo-root external/Reconstruction-based-Attack`
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
- `experiments/recon-runtime-mainline-ddim-public-10-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-10-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-25-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-25-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step10/artifact-mainline/summary.json`
- `experiments/recon-runtime-mainline-kandinsky-public-smoke/summary.json`
- `experiments/recon-runtime-mainline-kandinsky-public-smoke/artifact-mainline/summary.json`
- `experiments/variation-synth-smoke/summary.json`
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
3. 对比 `recon` 的 `public-50` 与 `public-100` 指标差异，并记录运行成本
4. 把真实 target/shadow score artifact 的命名和目录约束落实到 `recon` 主线
5. 维持 `DiT` 官方 `sample.py` 路线可重复执行，并把本地 checkpoint 驱动的 `step50` 证据视需要继续往更高步数或更高分辨率推进
6. `kandinsky_v22` public smoke 已通；`Stable Diffusion + DDIM` 的 `100-sample public` 也已通；`DiT step50` 也已补上，下一步是解释 `public-50` 与 `public-100` 的指标变化，并继续补齐 `Kandinsky` 覆盖
7. 评估 `variation` 真实 API 调用所需的凭据、预算和 query image 约束
8. 评估 `CLiD` 的真实 text-to-image 资产是否可在当前机器上最小复现
9. 维持黑盒状态文档、实验目录和主线命令说明同步

## 当前阻塞项

- 公开 `recon` checkpoint 与 dataset 已在本机落地，但运行时语义映射尚未核准
- 仍缺与论文一致的 target/shadow/member/non-member 直接映射说明
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
