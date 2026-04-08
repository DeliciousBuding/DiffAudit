# 白盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 白盒成员推断、梯度级攻击、记忆定位与内部信号审计
- `status`: `GSA 1k-3shadow paper-aligned runtime complete; W-1 strong-v3 full-scale frozen as defended main rung; strong-v2 full-scale retained as reference rung; independent GSA epoch300 rerun1 is actively training`
- `blocked by`: `W-1` 仍是 defended comparator 而不是最终 benchmark；`DPDM` 与 `GSA` 仍有模型结构不一致；`W-2` 仍缺稳定训练目标与实现
- `next step`: keep `strong-v3 full-scale` as the admitted defended main rung, low-frequency monitor `gsa-cifar10-1k-3shadow-epoch300-rerun1`, and only revisit white-box wording when the rerun produces a new `summary.json` or a stable failure mode
- `last updated`: `2026-04-09`

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
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-max128.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-max256.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-max512.md`
- `workspaces/white-box/2026-04-08-dpdm-w1-multi-shadow-strongv3-3shadow-full.md`
- `workspaces/white-box/2026-04-08-whitebox-attack-defense-table.md`
- `workspaces/white-box/2026-04-08-gsa-1k-3shadow-asset-prep.md`
- `workspaces/white-box/2026-04-09-gsa-epoch300-rerun1-launch.md`
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

- 现在能证明“强攻击基线已成立”，但还不能写成论文最终 benchmark
- `W-1` 已有正式 defense-vs-attack 对比，但仍是 defended comparator 而不是论文最终防御 benchmark
- `DPDM` 当前 checkpoint 既是单文件字典，又不是 `GSA` 的 `UNet2DModel` 架构，不能直接喂给当前提取器
- 当前 `DPDM` comparator 已扩到 defended target + defended shadows strong-v2，并完成 three-shadow full-scale 比较，但仍不是最终 benchmark
- `strong-v3` checkpoint 集合已确认可读，并已成功跑出 three-shadow `max128 / max256 / max512 / full-scale` GPU comparator，当前主口径已冻结但仍非论文最终 benchmark
- `Finding NeMo` 仍缺 neuron-level 分析接口与资产

## 当前最短路径

1. 固定 `W-1 strong-v3 full-scale` 为当前白盒 defended 主结果
2. 保留 `strong-v2 full-scale` 作为参考 rung
3. 低频监控 `gsa-cifar10-1k-3shadow-epoch300-rerun1`
4. 在 rerun1 产出 `summary.json` 前，不改 admitted `GSA` 主攻击口径
5. 继续把系统侧优先级转回 admitted 结果接入与灰盒主讲线
