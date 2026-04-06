# 白盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 白盒成员推断、记忆定位、梯度与激活级审计
- `status`: `repo-ready + gradient-smoke + closed-loop-smoke`
- `blocked by`: 论文对齐的 `target/shadow` checkpoint、`member/non-member` 划分、统一 adapter / CLI 接口；新到位的 DDPM `.pt` 不是 `GSA` 直接可恢复的 `checkpoint-*` accelerate state
- `next command`: `conda run -n diffaudit-research python workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py --train_data_dir <paper-target-member-train> --model_dir <paper-target-checkpoint-dir> --resume_from_checkpoint latest --resolution 32 --ddpm_num_steps 20 --sampling_frequency 2 --attack_method 1 --output_name <target-member-gradients.pt>`
- `last updated`: 2026-04-07

## 推荐论文

- `2025-popets-white-box-membership-inference-diffusion-models.pdf`
- `2025-local-mirror-white-box-membership-inference-diffusion-models.pdf`
- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf`

## 推荐定位

- 主论文：`2025-popets-white-box-membership-inference-diffusion-models.pdf`
- 基础参照：`2023-icml-secmi-membership-inference-diffusion-models.pdf`
- 对照阅读：`2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- 记忆定位扩展：`Finding NeMo`

## 起步建议

1. 先明确白盒边界：哪些内部信号可以访问
2. 先列资产清单：checkpoint、训练配置、loss、gradient、中间状态
3. 补白盒信号访问矩阵：`loss / gradient / activations / cross-attention / score direction`
4. 先写实验假设和指标，再决定执行顺序
5. 先从最小可验证路径开始，不要一上来追完整论文指标

## 当前启动进展

- 已选主线：`2025 PoPETS White-box Membership Inference Attacks against Diffusion Models`
- 已下载官方代码：`workspaces/white-box/external/GSA`
- 已完成最小白盒梯度 smoke：`workspaces/white-box/smoke-ddpm/member-gradients.pt`
- 已完成最小白盒 closed-loop smoke：`workspaces/white-box/runs/gsa-closed-loop-smoke-20260407-cpu/summary.json`
- 已完成新资源适用性判断：`workspaces/white-box/2026-04-07-gsa-asset-intake.md`
- 详细记录：`workspaces/white-box/2026-04-06-gsa-kickoff.md`

## 当前难点

- 白盒和灰盒边界容易混
- 对 checkpoint 和训练配置要求高
- 当前 closed-loop smoke 仍是 toy synthetic assets，不具备论文级解释力
- `CIFAR-10` 归档只能补数据桶，`ckpt_cifar10.pt` / `ckpt_tini.pt` 仍有 checkpoint format mismatch
- 仓库内还没有 `diffaudit` 级白盒 adapter / CLI
