# 2026-04-10 Finding NeMo Protocol Reconciliation

## 目的

这份文档用于判断：

- 当前 admitted 白盒资产是否与 `Finding NeMo` 原始协议面兼容
- 如果不兼容，当前还能否保留一条 `zero-GPU-preferred` 的可观测性规划路线

它不是运行报告，不是 GPU 申请单，也不是 admitted 升级文档。

## 固定字段

- `owner`: `research_leader`
- `track`: `white-box`
- `artifact_type`: `asset + protocol compatibility memo`
- `candidate`: `Finding NeMo + local memorization + FB-Mem`
- `decision_scope`: `Stage 0: protocol reconciliation`
- `gpu_release`: `none`
- `admitted_change`: `none`

## 当前固定边界

- `current_mainline = PIA + provisional G-1(all_steps)`
- `current_depth_line = GSA + W-1 strong-v3 full-scale`
- `white-box same-protocol bridge = closed-frozen`
- `batch32 diagnostic comparator = runtime-smoke / diagnostic`
- `active_gpu_question = none`
- 当前只允许推进 `protocol reconciliation + portable observability smoke planning`

## 决定性输入

### admitted 白盒锚点

- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`

### 候选参考

- `docs/paper-reports/white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models-report.md`
- `workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md`
- `workspaces/white-box/signal-access-matrix.md`
- `workspaces/white-box/2026-04-06-gsa-kickoff.md`
- `workspaces/white-box/2026-04-08-gsa-1k-3shadow-asset-prep.md`

### 仓内相关代码面

- `workspaces/white-box/external/GSA`
- `external/DPDM`
- `external/PIA/stable_diffusion/stable_attack.py`
- `external/Reconstruction-based-Attack/*`

## 协议面对齐

### 1. 模型家族

`Finding NeMo` 原论文要求：

- `Stable Diffusion v1.4`
- text-conditioned latent diffusion
- `cross-attention value layers`

当前 admitted 白盒攻击主线实际是：

- `CIFAR-10 DDPM`
- `diffusers.UNet2DModel`
- `GSA` 梯度抽取与 shadow/target 比较

当前 defended 主 rung 实际是：

- `DPDM`
- `NCSNpp`-based denoiser stack
- CIFAR-10 class-conditional route

结论：

- `paper-faithful NeMo on current admitted white-box assets = no-go`

这是结构性不兼容，不是“还差一次 run”。

### 2. 信号面

`Finding NeMo` 原始观测面需要：

1. `cross-attention value layers`
2. `memorized prompts`
3. `non-memorized holdout prompts`
4. 首步去噪噪声差异
5. 神经元停用 / 回激活能力

当前仓库已经证实可用的白盒信号面是：

1. `gradient`
2. `activations` 只停留在规划层

并且 kickoff 已明确记录：

- `activation: no hook pipeline in repo yet`

结论：

- `cross-attention observability = missing`
- `activation observability = conceptually aligned but implementation-missing`

### 3. 资产面

`Finding NeMo` 需要：

1. 可核准的 `memorized prompts`
2. 原图或可回收 URL 证据
3. 大规模非记忆 holdout 激活统计
4. `Stable Diffusion v1.4` 权重与层级命名

当前 admitted 白盒资产只有：

1. `CIFAR-10` target/shadow member/non-member splits
2. `accelerate-checkpoint-dir` 形式的 `GSA` checkpoints
3. `DPDM` 单文件 checkpoint 与 defended comparator

结论：

- `memorized prompt asset line = absent`
- `text-conditioned holdout statistics = absent`
- `SD v1.4 layer naming map = absent`

### 4. 入口面

当前仓库可复用的 related code 面主要有两类：

1. `GSA`
   - 证明 `DDPM` 梯度抽取与分类比较可运行
   - 不能直接提供 `Finding NeMo` 所需的 activation / cross-attention hook pipeline
2. `PIA stable_diffusion` 与 `Reconstruction-based-Attack`
   - 证明仓内有 `StableDiffusionPipeline`、`encoder_hidden_states`、`attn_procs` 等技术参考
   - 但这些路径不属于当前 admitted 白盒主线，也没有现成的 neuron-level hook / ablation flow

结论：

- `portable entry for Finding NeMo-style observability = not yet implemented`

## 兼容性矩阵

| 维度 | Finding NeMo 原始要求 | 当前 admitted 白盒资产 | 结论 |
| --- | --- | --- | --- |
| 模型家族 | `Stable Diffusion v1.4` | `DDPM/CIFAR-10` + `DPDM/NCSNpp` | `paper-faithful incompatible` |
| 条件输入 | `prompt/caption` | 当前主线无 text prompt 资产 | `missing` |
| 观测面 | `cross-attention value layers` | 当前仅 `gradient`; `activations` 无 hook pipeline | `missing` |
| 资产 | `memorized prompts + original images/URLs` | 当前仅 member/non-member image splits | `missing` |
| 统计基线 | 非记忆 prompt 激活统计 | 当前无 prompt holdout | `missing` |
| 干预面 | neuron ablation / reactivation | 当前无相应实现 | `missing` |
| 可迁移面 | layer-level activations / grad_norm | 当前理论可迁移到 DDPM | `partial` |

## Reconciliation Decision

### A. Paper-Faithful NeMo

当前裁决：

- `paper-faithful reproduction = no-go`

理由：

1. 模型家族不一致
2. 观测面不一致
3. 资产面不一致
4. 当前没有 neuron-level hook / ablation 入口

### B. Migrated DDPM Observability Route

当前裁决：

- `migrated DDPM observability route = not-yet but plannable`

这里的“migrated”只表示：

- 在当前 admitted `DDPM/CIFAR-10` 资产上，先尝试定义 `activations + optional grad_norm` 的最小可观测接口

它不表示：

- 当前已经对齐到 `Finding NeMo` 原论文协议
- 当前已经具备 `cross-attention` 可观测性
- 当前已经有机制证据

## Portable Observability Smoke Planning Boundary

当前允许规划的最小 smoke 只服务于“可观测性是否存在”，不服务于“机制是否成立”。

### 允许的输入合同

1. 一个固定 admitted 资产根
2. 一个固定 checkpoint root
3. 一个固定 layer selector
4. 一个固定 sample binding rule
5. 一个固定输出 schema

### 建议输出 schema

- `sample_id`
- `split`
- `checkpoint_root`
- `layer_id`
- `signal_type`
- `timestep`
- `tensor_shape`
- `summary_stat`
- `artifact_path`

### 当前允许的信号

1. `activations`
2. `grad_norm` 可选

### 当前不允许的信号

1. `cross-attention intervention`
2. 神经元停用
3. 回激活验证
4. 前景/背景机制解释

## 当前最小 blocker

在本 memo 完成后，当前 blocker 已收缩为 4 项：

1. `checkpoint root` 需固定为单一 admitted 路径
2. `layer naming map` 仍未写出
3. `sample binding rule` 仍未写出
4. `output schema` 仍未写成可审查合同

## 当前 Verdict

- `paper_faithful_reproduction`: `no-go`
- `ddpm_activation_migration`: `not-yet but designable`
- `gpu_release`: `none`
- `next_required_proof`: `portable entry contract for observability smoke`

## 下一步

1. 不申请 GPU
2. 先补一份 `portable observability smoke` 计划文档
3. 在计划文档里写死：
   - `checkpoint root`
   - `layer naming map`
   - `sample binding rule`
   - `output schema`
4. 只有这 4 项都 review-ready，才允许重新评估是否进入一次最小 `validation-smoke`
