# 白盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 白盒成员推断、梯度级攻击、记忆定位与内部信号审计
- `status`: `GSA 1k-3shadow paper-aligned runtime complete; W-1 strong-v2 defended comparator complete; strong-v3 checkpoints valid but GPU comparator still unstable`
- `blocked by`: `W-1` 仍是 `runtime-smoke` comparator；`DPDM` 与 `GSA` 仍有模型结构不一致；`strong-v3` GPU comparator 仍不稳定；`W-2` 仍缺稳定训练目标与实现
- `next step`: keep the CPU diagnostic as evidence, then narrow the GPU comparator instability before moving to the next defended rung
- `last updated`: `2026-04-08`

## 推荐论文

- `2025-popets-white-box-membership-inference-diffusion-models.pdf`
- `2025-local-mirror-white-box-membership-inference-diffusion-models.pdf`
- `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf`
- `2023-tmlr-differentially-private-diffusion-models.pdf`
- `2023-arxiv-differentially-private-diffusion-models-generate-useful-synthetic-images.pdf`

## 当前主线与防御候选

- 主线：`GSA`
- 防御候选：`W-1 = DPDM / Diffusion-DP`
- 扩展研究：`Finding NeMo`

## 当前可执行证据

- `workspaces/white-box/2026-04-07-gsa-runtime-mainline.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-smoke.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-target-only.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-shadow-comparator.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-comparator.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-targetmember.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv2.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv2-max512.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv2-3shadow-max512.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv2-3shadow-full.md`
- `workspaces/white-box/2026-04-08-dpdm-strongv3-rung.md`
- `workspaces/white-box/2026-04-08-dpdm-strongv3-comparator-diagnostic.md`
- `workspaces/white-box/2026-04-08-whitebox-attack-defense-table.md`
- `workspaces/white-box/2026-04-08-gsa-1k-3shadow-asset-prep.md`
- `workspaces/white-box/2026-04-07-gsa-asset-intake.md`
- `workspaces/white-box/2026-04-07-gsa-closed-loop-smoke.md`
- `workspaces/white-box/2026-04-06-gsa-kickoff.md`
- `workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260407-cpu/summary.json`
- `workspaces/white-box/assets/gsa-gpu-128/manifests/cifar10-ddpm-gpu-128.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260408-gpu-128/summary.json`

## 本地代码上下文

- `workspaces/white-box/external/GSA`
- `external/DPDM`
- `scripts/rebuild_gsa_cifar_buckets.py`

## 当前推荐执行顺序

1. 起 `GSA` 的 `target + shadow-01/02/03` 训练，先拿到第一批 `checkpoint-*`
2. 将 `DPDM` 从 `loss.n_noise_samples=1` 逐步恢复到论文更接近的训练口径
3. 用上游更强口径提升 `checkpoint-*` 训练强度与 epochs
4. 用上游更接近论文的梯度提取与攻击评估配置重跑 `GSA`
5. 用 `DPDM` checkpoint 做第一版 `W-1` 白盒防御比较
6. 暂不优先做 `W-2`

## 当前阻塞项

- 现在能证明“闭环已通”，还不能证明“攻击已稳定成立”
- `GSA 1k-3shadow` 训练还没有产出新的 `checkpoint-*`
- `W-1` 已有正式 defense-vs-attack 对比，但仍是 `runtime-smoke` 级 comparator
- `DPDM` 当前 checkpoint 既是单文件字典，又不是 `GSA` 的 `UNet2DModel` 架构，不能直接喂给当前提取器
- 当前 `DPDM` comparator 已扩到 defended target + defended shadows strong-v2，并完成 three-shadow full-scale 比较，但仍不是最终 benchmark
- `strong-v3` checkpoint 集合已确认可读，但 GPU comparator 路径仍不稳定
- `Finding NeMo` 仍缺 neuron-level 分析接口与资产
- 还缺跨黑/灰/白统一总表

## 当前最短路径

1. 将 `W-1 strong-v2 3-shadow full-scale` 与 `GSA 1k-3shadow` 的白盒 attack-defense 对比接入统一总表
2. 保留 `strong-v3` CPU diagnostic 作为资产有效性证据
3. 收口 `strong-v3` GPU comparator 失败模式
4. 暂不重跑 `GSA` 主攻击线
