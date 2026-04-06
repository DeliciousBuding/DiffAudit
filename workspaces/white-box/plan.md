# 白盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 白盒成员推断、记忆定位、梯度与激活级审计
- `status`: `repo-ready + gradient-smoke + toy-end-to-end-executable`
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
- 已建立正式资产根目录：`workspaces/white-box/assets/gsa/`
- 已完成最小白盒梯度 smoke：`workspaces/white-box/smoke-ddpm/member-gradients.pt`
- 已完成最小白盒 toy 端到端执行 smoke：`workspaces/white-box/runs/gsa-closed-loop-smoke-20260407-cpu/summary.json`
- 已完成新资源适用性判断：`workspaces/white-box/2026-04-07-gsa-asset-intake.md`
- 详细记录：`workspaces/white-box/2026-04-06-gsa-kickoff.md`

## 当前难点

- 白盒和灰盒边界容易混
- 对 checkpoint 和训练配置要求高
- 当前 toy 端到端执行 smoke 仍是 toy synthetic assets，不具备论文级解释力
- `CIFAR-10` 归档只能补数据桶，`ckpt_cifar10.pt` / `ckpt_tini.pt` 仍有 checkpoint format mismatch
- 仓库内还没有 `diffaudit` 级白盒 adapter / CLI
- 当前正式 handoff 已收口到 `workspaces/white-box/assets/gsa/HANDOFF.md`

## 面向论文的可执行路径

- **真实 blocker 决定路径**：新拿到的 CIFAR10 `.pt` checkpoint 不符合 upstream `GSA` 的 `accelerate` `checkpoint-*` layout，无法直接 resume。要在 `white-box` 实验中完成 end-to-end 结果，就必须通过 target/shadow 自训练生成 `checkpoint-*` 目录；这同时也是为什么我们强调 `end-to-end executable` 的 pipeline 而不是 toy metrics。
- **最短命令路径**（每一步都在白盒目录、无新共享代码、可用 CPU）：
  1. `tar -xzf /path/to/cifar-10-python.tar.gz -C Project/workspaces/white-box/data/cifar-10-batches-py`（若还没存档，请先在白盒目录下下载或从 `LocalOps` 复制，确保数据只落在 owned files）。
  2. `cd Project/workspaces/white-box/external/GSA/DDPM && conda run -n diffaudit-research python process_DDPM_ds.py --dataset_dir ../../data/cifar-10-batches-py --output_dir ../../runs/gsa-paper-buckets --datanum_target_model 10000 --datanum_per_shadow_model 2000 --number_of_shadow_model 3`，把 CIFAR10 影像变为 `target_model` 与多个 `shadow_model` 文件夹。
  3. `conda run -n diffaudit-research python external/GSA/DDPM/train_unconditional.py --train_data_dir runs/gsa-paper-buckets/target_model/model_member --output_dir runs/gsa-paper-checkpoints/target --num_epochs 1 --train_batch_size 4 --resolution 32 --save_model_epochs 1 --mixed_precision no`（可根据时间把 `--num_epochs` 填 2 或 5，但关键是让 `accelerate.save_state` 生成 `runs/gsa-paper-checkpoints/target/checkpoint-*`）。
  4. 复用第 3 步命令分别训练 `target_model/non_model_member` 和每组 `shadow_model/{i}`，把输出丢到 `runs/gsa-paper-checkpoints/target-nonmember`, `runs/gsa-paper-checkpoints/shadow-0`, …，确保所有 `checkpoint-*` 目录都基于真实 asset。
  5. `conda run -n diffaudit-research python external/GSA/DDPM/gen_l2_gradients_DDPM.py --train_data_dir runs/gsa-paper-buckets/target_model/model_member --model_dir runs/gsa-paper-checkpoints/target --resume_from_checkpoint latest --resolution 32 --ddpm_num_steps 20 --sampling_frequency 2 --attack_method 1 --output_name runs/gsa-paper-gradients/target_member-gradients.pt`，其他 split 同理；此步骤依赖 accelerate checkpoint compatibility，所以必须用最新 `checkpoint-*` 路径。
  6. `conda run -n diffaudit-research python external/GSA/test_attack_accuracy.py --target_gradients runs/gsa-paper-gradients/target_member-gradients.pt --target_non_gradients runs/gsa-paper-gradients/target_non_member-gradients.pt --shadow_gradients runs/gsa-paper-gradients/shadow_member-gradients.pt --shadow_non_gradients runs/gsa-paper-gradients/shadow_non_member-gradients.pt`，最终产出可复现的 `closed-loop` 指标。
- **成品要求**：只有全链路（数据拆分→target/shadow checkpoint→梯度抽取→攻击评估）都能在这些目录上运行，才算 `end-to-end executable`；在此之前仍然以 `closed-loop-smoke` 语言描述，说明尚未使用论文级 asset。
