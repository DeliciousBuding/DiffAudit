# 黑盒方向计划

## 状态面板

- `owner`: 待认领
- `scope`: 黑盒成员推断、数据集级审计、black-box leakage 线索整理
- `status`: 进行中
- `blocked by`: 真实 checkpoint、flagfile、与论文一致的数据布局
- `next command`: `conda run -n diffaudit-research python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml`
- `last updated`: 2026-04-05

## 主论文与场景

- `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf`
- `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf`
- `2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf`

## 当前可执行证据

- `experiments/clid-dry-run-smoke/summary.json`
- `experiments/clid-artifact-summary/summary.json`
- `experiments/recon-eval-smoke/summary.json`
- `experiments/recon-artifact-summary/summary.json`
- `experiments/recon-upstream-eval-smoke/summary.json`
- `experiments/variation-synth-smoke/summary.json`
- `experiments/blackbox-status/summary.json`

## 本地代码上下文

- `external/CLiD`
- `external/Reconstruction-based-Attack`

## 推荐分工

- 主复现：`Black-box Membership Inference Attacks against Fine-tuned Diffusion Models`
- 对照参考：`Towards Black-Box Membership Inference Attack for Diffusion Models`
- 场景补充：`CLiD`
- 审计扩展：`CDI`
- 细分场景补充：`Membership Inference Attacks for Face Images Against Fine-tuned Latent Diffusion Models`

## 一周行动清单

1. 整理三篇论文的问题定义、攻击假设、输入输出和指标
2. 对比三篇论文需要的资产：checkpoint、flagfile、dataset_root、prompt 或 text 条件
3. 跑通仓库现有黑盒基础命令：
   - `plan-secmi`
   - `probe-secmi-assets`
   - `prepare-secmi`
   - `dry-run-secmi`
4. 形成一份黑盒方向复现计划和 threat model 对照表
5. 明确当前缺失的真实资产
6. 补 `paper-matrix` 与 `experiment-entrypoints` 文档

## 当前阻塞项

- 缺真实 checkpoint 和训练期 flagfile
- 缺与论文一致的实验数据布局
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
