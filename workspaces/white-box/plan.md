# 白盒方向计划

## 状态面板

- `owner`: `research_leader`
- `scope`: 白盒成员推断、梯度级攻击、记忆定位与内部信号审计
- `status`: `GSA epoch300 rerun1 admitted as white-box attack main evidence; W-1 strong-v3 full-scale frozen as defended main rung; strong-v2 full-scale retained as reference rung; direct GSA gradient extraction verified on admitted assets`
- `blocked by`: `W-1` 仍是 defended comparator 而不是最终 benchmark；`DPDM` 与 `GSA` 仍有模型结构不一致；第二白盒线虽已形成 `GSA2` bounded corroboration，但 distinct 新家族仍未形成`
- `next step`: keep `strong-v3 full-scale` as the admitted defended main rung, keep `GSA2` as a bounded corroboration line, and shift white-box work to defense breadth plus bridge-contract hygiene`
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
- `workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md`
- `workspaces/white-box/2026-04-08-gsa-1k-3shadow-asset-prep.md`
- `workspaces/white-box/2026-04-09-gsa-epoch300-rerun1-launch.md`
- `workspaces/white-box/2026-04-07-gsa-asset-intake.md`
- `workspaces/white-box/2026-04-07-gsa-closed-loop-smoke.md`
- `workspaces/white-box/2026-04-06-gsa-kickoff.md`
- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260407-cpu/summary.json`
- `workspaces/white-box/assets/gsa-gpu-128/manifests/cifar10-ddpm-gpu-128.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260408-gpu-128/summary.json`

## 本地代码上下文

- `workspaces/white-box/external/GSA`
- `external/DPDM`
- `scripts/rebuild_gsa_cifar_buckets.py`

## 当前推荐执行顺序

1. 固定 `GSA rerun1` 为 admitted 白盒攻击主结果
2. 固定 `W-1 strong-v3 full-scale` 为 defended 主 rung，并保留 `strong-v2 full-scale` 作为参考 rung
3. 建立 `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 same-protocol bridge 合同
4. 收口 `DPDM` 启动入口可移植性，不让本地 scheduler 成为外部协作者前置条件
5. 只启动同一个白盒主 GPU 问题：same-protocol bridge
6. 暂不优先做 `W-2`

## 当前阻塞项

- 现在能证明“强攻击基线已成立”，但还不能写成论文最终 benchmark
- `W-1` 已有正式 defense-vs-attack 对比，但仍是 defended comparator 而不是论文最终防御 benchmark
- `DPDM` 当前 checkpoint 既是单文件字典，又不是 `GSA` 的 `UNet2DModel` 架构，不能直接喂给当前提取器
- 当前 `DPDM` comparator 已扩到 defended target + defended shadows strong-v2，并完成 three-shadow full-scale 比较，但仍不是最终 benchmark
- `strong-v3` checkpoint 集合已确认可读，并已成功跑出 three-shadow `max128 / max256 / max512 / full-scale` GPU comparator，当前主口径已冻结但仍非论文最终 benchmark
- `Finding NeMo` 仍缺 neuron-level 分析接口与资产
- 当前白盒最大问题已不是“能不能继续把攻击跑高”，而是 `GSA` 与 `DPDM/W-1` 仍不在同一个协议面上
- 直接 upstream `gen_l2_gradients_DDPM.py` 在 admitted target-member assets 上已被重新验证为可跑；当前执行层问题已缩到 dataset-mode 与 output-path hygiene，而不是梯度提取本身
- 当前 `WB-2` 已收口为 `positive secondary line`：`GSA2 comparator` 在 admitted target pair + first shadow pair 上完成 bounded comparator，`AUC = 0.922498`，但仍低于 admitted `GSA1` 主线，因此只保留为 corroboration，不升级为新 headline
- 白盒 breadth shortlist 已完成第一轮负筛选：
  - `Finding NeMo` 当前仍是 observability / mechanism 路线，不是可执行 defended comparator
  - `Local Mirror` 不是 defense family
  - `DPDM` 的各个 strong-v2/strong-v3 变体仍只是在扩 `W-1`，不是第二防御家族

## 当前最短路径

1. 固定 `GSA rerun1` 为 admitted 白盒攻击主结果
2. 固定 `W-1 strong-v3 full-scale` 为当前白盒 defended 主结果
3. 保留 `strong-v2 full-scale` 作为参考 rung
4. 当前唯一 active GPU 问题是 same-protocol benchmark bridge：
   - 方案 A：训练/对接架构对齐的 DP-DDPM，直接进入 GSA 协议
   - 方案 B：为 `DPDM` 实现 GSA-style gradient feature extraction
5. bridge 入口必须 portable，且不依赖本地 scheduler
6. 继续把系统侧优先级转回 admitted 结果接入与灰盒主讲线
