# 2026-04-11 DP-LoRA Comparability Note

## 目的

这份文档旨在明确 `DP-LoRA`（实际论文方法为 SMP-LoRA）与当前 admitted 白盒线 `GSA + W-1 strong-v3 full-scale` 的比较关系，为未来的 release-review 做准备。

## 基本信息

- `candidate_key`: `white-box/dp-lora/cifar10-ddpm`
- `track`: `white-box`
- `method`: `dp-lora`（实际论文方法为 SMP-LoRA）
- `current_status`: `comparability/intake hardening`
- `local_execution_status`: `handover-reviewed / partial-evidence-ready / gpu-expansion-not-released`
- `gpu_release`: `none`
- `queue_state`: `not-requestable`
- `paper_ref`: `2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models`
- `code_repo`: `https://github.com/WilliamLUO0/StablePrivateLoRA`

## Paper Surface Memo

### 论文核心方法

论文实际提出的方法是 **SMP-LoRA**（Stable Membership-Privacy-preserving LoRA），不是 DP-LoRA（Differentially Private LoRA）：

- **核心思想**：在 LoRA 适配训练过程中，通过 min-max 优化同时最小化适配损失和最大化代理攻击模型的 MI gain
- **MP-LoRA（基线方法）**：直接最小化 `L_ada + lambda * G`，存在优化不稳定问题
- **SMP-LoRA（改进方法）**：最小化 `L_ada / (1 - lambda * G + delta)`，通过约束局部平滑性实现稳定收敛
- **隐私保证**：不是差分隐私，而是通过 min-max 博弈使攻击模型无法区分成员和非成员
- **参数高效性**：只训练 LoRA 低秩矩阵（B, A），冻结预训练权重

### 模型族与数据域

- **原论文语境**：Stable Diffusion v1.5（Latent Diffusion Model）+ text-conditioned generation
- **论文实验数据集**：Pokemon, CelebA (Small/Large/Gender/Varying), AFHQ, MS-COCO
- **当前仓库适配目标**：DDPM / CIFAR-10（无条件生成）
- **关键差异**：论文使用 text-conditioned LDM，当前仓库使用 unconditional DDPM

### 攻击面

- 论文评估了三种攻击：white-box loss-based MI attack, white-box gradient-based MI attack (GSA), black-box MI attack
- 论文直接使用了 GSA (Pang et al. 2023) 作为评估攻击之一
- 与 `W-1 (DPDM)` 的评估协议有部分重叠

### 论文关键实验结果（Pokemon 数据集）

| 方法 | FID ↓ | ASR (%) ↓ | |AUC-0.5| ↓ |
| --- | --- | --- | --- |
| LoRA | 0.20±0.04 | 82.27±4.38 | 0.73±0.09 |
| MP-LoRA | 2.10±0.51 | 51.67±2.73 | 0.07±0.04 |
| SMP-LoRA | 0.32±0.07 | 51.97±1.20 | 0.14±0.02 |

## Protocol Comparability

### 可比较维度

| 维度 | 当前 admitted 白盒线 | SMP-LoRA | 比较状态 |
| --- | --- | --- | --- |
| 攻击方法 | GSA | GSA (论文已评估) | 完全一致 |
| 评估指标 | AUC, ASR, TPR@1%FPR, TPR@0.1%FPR | AUC, ASR, TPR@5%FPR | 部分一致（FPR 阈值不同） |
| 威胁模型 | 成员推断攻击 | 成员推断攻击 | 完全一致 |

### 不可比较维度

| 维度 | 当前 admitted 白盒线 | SMP-LoRA | 差异说明 |
| --- | --- | --- | --- |
| 模型族 | DDPM (unconditional) | Stable Diffusion v1.5 (text-conditioned LDM) | 架构根本不同 |
| 数据集 | CIFAR-10 | Pokemon, CelebA, AFHQ, MS-COCO | 数据域不同 |
| 防御方法 | 全模型 DP 训练 (DPDM) | LoRA min-max 优化 (SMP-LoRA) | 防御机制完全不同 |
| 训练方式 | 全模型重新训练 | LoRA 适配训练 | 计算成本不同 |
| 条件类型 | 无条件 | text-conditioned | 输入接口不同 |
| 防御强度调节 | DP epsilon | lambda 系数 | 调节方式不同 |

## 关键适配问题

### 1. 模型族差异

论文使用 Stable Diffusion v1.5，当前仓库使用 DDPM。这意味着：

- LoRA 作用位置不同：SD 的 attention 层 vs DDPM 的 attention 层
- 条件输入不同：text prompt vs 无条件
- 潜在空间不同：SD 在 latent space，DDPM 在 pixel space

**适配路径**：
- 路径 A：在当前 DDPM 上实现 LoRA + SMP-LoRA（需要自行实现 LoRA 层）
- 路径 B：引入 SD v1.5 作为新的模型族（需要新的预训练模型和评估协议）

### 2. 数据集差异

论文使用 Pokemon/CelebA 等，当前仓库使用 CIFAR-10：

- CIFAR-10 是 32x32，Pokemon/CelebA 是 512x512
- CIFAR-10 无 text prompt，论文方法依赖 text conditioning

### 3. 评估协议差异

- 论文使用 TPR@5%FPR，当前仓库使用 TPR@1%FPR 和 TPR@0.1%FPR
- 论文的 proxy attack model 是 3-layer MLP，当前仓库使用 logistic-regression-1d

## Asset Checklist

### 模型资产

| 资产 | 是否必需 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| 预训练 DDPM checkpoint | 是 | 已有 | 复用当前 PIA/GSA 的 DDPM checkpoint |
| LoRA 适配层定义 | 是 | 待实现 | 需要为 DDPM attention 层实现 LoRA |
| SMP-LoRA 训练逻辑 | 是 | 待实现 | min-max 优化 + 代理攻击模型 |
| 代理攻击模型 | 是 | 待实现 | 3-layer MLP，基于 adaptation loss |
| 训练配置文件 | 是 | 待创建 | 定义 LoRA rank、lambda、学习率等 |

### 数据资产

| 资产 | 是否必需 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| CIFAR-10 训练集 | 是 | 已有 | 复用当前数据集 |
| 成员/非成员划分 | 是 | 已有 | 复用当前 PIA 的 split 文件 |
| 辅助数据集（D_aux） | 是 | 待构建 | SMP-LoRA 需要 member + non-member 辅助集 |

### 代码资产

| 资产 | 是否必需 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| 上游代码仓库 | 是 | 可用 | `https://github.com/WilliamLUO0/StablePrivateLoRA` |
| LoRA 训练脚本 | 是 | 待适配 | 上游基于 SD v1.5，需适配到 DDPM |
| 代理攻击模型 | 是 | 待实现 | 3-layer MLP |
| GSA 攻击集成 | 是 | 待实现 | 当前仓库已有 GSA 攻击代码 |

### 评估资产

| 资产 | 是否必需 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| 统一评估指标计算 | 是 | 可复用 | 复用当前的评估脚本 |
| 与 W-1 的比较脚本 | 是 | 待实现 | 定义统一的比较协议 |

## Expected Artifact

如果未来进入准入验证，预期产物包括：

1. **训练产物**
   - LoRA 适配后的 DDPM checkpoint
   - 代理攻击模型 checkpoint
   - 训练日志（包含 lambda、loss、MI gain 等）

2. **评估产物**
   - GSA 攻击评估结果（AUC, ASR, TPR@1%FPR, TPR@0.1%FPR）
   - 与 W-1 的比较报告

3. **分析产物**
   - 防御强度 vs 图像质量权衡分析（lambda 扫描）
   - 不同 LoRA rank 下的性能曲线

4. **复现资产**
   - 可复现的训练脚本
   - 评估脚本

## Stop Conditions

在以下情况下应直接判 `not-yet / no-go`：

1. **模型适配失败**
   - 无法在 DDPM (unconditional) 上成功应用 LoRA 适配
   - DDPM 的 attention 层结构与 SD 差异过大，LoRA 无法直接移植

2. **防御效果不足**
   - 防御效果显著低于 `W-1` 且计算成本更高
   - 无条件生成模型上的 MI gain 信号与条件生成模型有本质差异

3. **集成失败**
   - 无法与 GSA 攻击进行有效集成
   - 评估协议无法统一（TPR@1%FPR vs TPR@5%FPR）

4. **可复现性**
   - 代码无法在无本地 scheduler 的环境中运行
   - 缺少必要的依赖或配置

5. **协议不可比**
   - DDPM/CIFAR-10 与 SD/Pokemon 的差异导致无法建立可辩护的比较关系

## 仓库基础设施审查

### LoRA 基础设施

- 当前仓库的 LoRA 代码**仅存在于 black-box recon 攻击管线**中
- LoRA 用途是 Stable Diffusion / Kandinsky 的微调，不是独立的 LoRA 模块
- 核心入口：`train_text_to_image_lora.py`（recon 攻击的工作区文件）
- 加载方式：`unet.load_attn_procs()`（diffusers 库的标准 LoRA 接口）
- **没有**独立的 LoRA 层实现，完全依赖 diffusers 库的 `AttnProcessor` 机制

### DP 基础设施

- 当前仓库**不存在**任何 DP-SGD / Opacus / 差分隐私训练代码
- DP 防御方面仅有 DPDM 的推理评估代码（`dpdm_w1.py`），使用 EDM/NCSNpp 架构
- **没有**可复用的 DP 训练基础设施

### DDPM Attention 层

- GSA 攻击使用 diffusers 的 `UNet2DModel`，含 `AttnDownBlock2D` / `AttnUpBlock2D`
- SecMI 使用自定义 `AttnBlock`（GroupNorm + Q/K/V Conv2d + 缩放点积注意力 + 残差）
- GSA 默认攻击层：`mid_block.attentions.0.to_v`
- **关键发现**：DDPM 的 attention 层使用的是标准 self-attention（Q/K/V 线性投影），LoRA 可以直接作用于 `to_q`/`to_k`/`to_v` 线性层

### GSA 攻击集成

- GSA 攻击代码完整，支持从 checkpoint 到评估指标的闭环
- 核心流程：生成 L2 梯度 → XGBoost 训练 → 评估 AUC/ASR/TPR
- **可以直接复用** GSA 攻击代码评估 SMP-LoRA 防御效果

## Decision-Grade Comparability Verdict

### Verdict: `not-yet`（有条件推进）

SMP-LoRA 与当前 admitted 白盒线存在**结构性协议差异**，但差异不是不可逾越的：

### 可以辩护的比较关系

1. **攻击方法一致**：论文已直接评估 GSA 攻击，与当前仓库的评估协议完全一致
2. **LoRA 技术上可行**：DDPM 的 attention 层使用标准 Q/K/V 线性投影，LoRA 可以直接作用于这些线性层
3. **GSA 攻击代码可复用**：当前仓库的 GSA 攻击代码完整，可以直接用于评估 SMP-LoRA 防御效果

### 不可辩护的比较关系

1. **模型族差异**：SD v1.5 (text-conditioned LDM) vs DDPM (unconditional) — 架构根本不同
2. **数据域差异**：512x512 text-image pairs vs 32x32 无条件图像
3. **防御机制差异**：min-max 博弈 vs 全模型 DP 训练 — 防御范式不同

### 推进条件

若要将 SMP-LoRA 从 `not-yet` 推进到 `admitted`，需要满足以下条件：

1. **路径 A（推荐）**：在当前 DDPM/CIFAR-10 上实现 LoRA + SMP-LoRA
   - 技术可行性：DDPM attention 层支持 LoRA（Q/K/V 线性层）
   - 代码起点：上游仓库 `https://github.com/WilliamLUO0/StablePrivateLoRA`
   - 需要实现：LoRA 层定义、SMP-LoRA min-max 优化、代理攻击模型
   - 需要适配：去除 text conditioning 依赖、适配 unconditional DDPM 训练循环

2. **路径 B（不推荐）**：引入 SD v1.5 作为新的模型族
   - 需要新的预训练模型和评估协议
   - 与当前仓库的 DDPM 评估体系不兼容
   - 违反"同一协议下比较"的原则

### 当前最合理定位

- `SMP-LoRA is a successor defense candidate with significant protocol delta, not a direct replacement for W-1.`
- 路径 A 在技术上可行，且当前仓库已经有一轮真实本地执行证据
- 上游代码仓库 `https://github.com/WilliamLUO0/StablePrivateLoRA` 可作为参考起点
- **不建议立即申请新的 GPU 扩跑**；当前应先按真实产物完成 intake / 准入判断，再决定是否释放 GPU

## 本地执行状态 (2026-04-12)

### 当前真实完成状态

当前本地证据已经不是纯 `CPU smoke`。截至 `2026-04-12`，这条线的真实完成状态是：

1. `Phase 1`：已完成一轮真实 sweep，但**实际只有 6 组配置**，不是早期口径里的 14 组。
2. `Phase 2`：已完成至少一组无防御基线评估。
3. `Phase 3`：已完成一条 `100 epochs` 长训练及对应评估。
4. `Phase 5`：存在 `lambda=0.05` 的中断训练残留，但**没有 final 结果，也没有评估结果**。

### 已核实指标

| 项目 | 产物 | 指标 |
| --- | --- | --- |
| sweep 最优配置 | `outputs/smp-lora-sweep/sweep_results.json` | `lambda=0.1, rank=4, ep=10, AUC=0.3438, Acc=0.3947` |
| 无防御基线 | `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json` | `AUC=0.5565, Acc=0.5263` |
| 最优配置相对提升 | 基于上面两项 | `AUC` 下降约 `38%` |
| 100 epochs 长训练 | `outputs/smp-lora-best-config/evaluation.json` | `AUC=0.3785, Acc=0.3947` |

### 当前可得出的局部结论

- `lambda=0.1, rank=4, epochs=10` 是当前本地产物里的最优配置。
- `epochs=10` 明显优于 `epochs=20` 与 `epochs=100`，当前可直接写成 `longer training did not improve privacy outcome under current local protocol`。
- 当前本地结果支持一句有限主张：
  - `under the current DDPM/CIFAR-10 local protocol, SMP-LoRA reduced the observed GSA AUC from 0.5565 to 0.3438 on the best tested configuration`
- 但当前**不能**把这条线写成：
  - `admitted`
  - `GPU released`
  - `queue released`
  - `W-1 replacement`
  - `paper-level benchmark complete`

### 中断残留

目录 `outputs/smp-lora-lambda005/` 当前只落到 `step_500`，已有 checkpoint 与训练日志，但没有：

- `final/lora_weights.pt`
- `evaluation.json`
- 任何可写入 comparability 口径的最终结果

因此这条 `lambda=0.05` 路线当前只能写成：

- `interrupted residual`
- `not yet evaluable`
- `not releasable`

### 当前治理定位

这条线当前最合理的定位是：

- `local execution evidence exists`
- `comparability/intake hardening remains`
- `gpu_release = none`
- `queue_state = not-requestable`
- `phase-e admission not yet granted`

当前不能因为已经有本地结果、或者因为 GPU 空闲，就直接把 handover 里的后续 14 个 GPU 重任务视为默认放行。

### 关联产物

- `D:/Code/DiffAudit/Research/workspaces/GPU_TRAINING_HANDOVER.md`
- `D:/Code/DiffAudit/Research/outputs/smp-lora-sweep/sweep_results.json`
- `D:/Code/DiffAudit/Research/outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `D:/Code/DiffAudit/Research/outputs/smp-lora-best-config/evaluation.json`
- `D:/Code/DiffAudit/Research/outputs/smp-lora-lambda005/`

### 下一步

当前下一步不是直接扩跑 GPU，而是：

1. 先把真实完成状态固定为 intake 事实。
2. 再判断这条线是否进入 `Phase E` 候选审查面。
3. 只有在单独 release-review / admission 判断通过后，才允许决定是否继续 `lambda=0.05`、`lambda=0.01`、`rank=1`、FID、自适应攻击等 GPU 或长任务。
