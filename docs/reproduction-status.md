# 复现状态总览

这份文档用于汇总仓库当前各条攻击线的真实推进状态。

判断标准统一分为：

- `research-ready`：论文、代码仓库、资产要求已整理
- `code-ready`：仓库内已有 planner / probe / dry-run / smoke 等实现
- `evidence-ready`：仓库内已经有可提交的 `summary.json` 或等价运行证据
- `asset-ready`：真实论文资产已到位，可以开始逼近真实 benchmark
- `benchmark-ready`：已经可以合理声称在跑或复算论文级结果

## 黑盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `clid` | `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf` | `evidence-ready` | `plan-clid` / `probe-clid-assets` / `dry-run-clid` / `run-clid-dry-run-smoke` / `summarize-clid-artifacts` | [clid-dry-run-smoke](../experiments/clid-dry-run-smoke/summary.json), [clid-artifact-summary](../experiments/clid-artifact-summary/summary.json) | 缺真实 text-to-image 资产、缺可重复的目标模型路径与数据集映射 |
| `recon` | `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf` 与 `external/Reconstruction-based-Attack` | `code-ready + evidence-ready` | `plan-recon` / `probe-recon-assets` / `dry-run-recon` / `run-recon-mainline-smoke` / `probe-recon-runtime-assets` / `run-recon-runtime-mainline` / `probe-recon-score-artifacts` / `run-recon-artifact-mainline` / `run-recon-eval-smoke` / `summarize-recon-artifacts` / `run-recon-upstream-eval-smoke` | [recon-mainline-smoke](../experiments/recon-mainline-smoke/summary.json), [recon-eval-smoke](../experiments/recon-eval-smoke/summary.json), [recon-artifact-summary](../experiments/recon-artifact-summary/summary.json), [recon-upstream-eval-smoke](../experiments/recon-upstream-eval-smoke/summary.json), [recon-runtime-ddim-mainline](../experiments/recon-runtime-mainline-ddim-smoke/summary.json), [recon-runtime-ddim-artifact-mainline](../experiments/recon-runtime-mainline-ddim-smoke/artifact-mainline-final/summary.json), [recon-runtime-ddim-public-10](../experiments/recon-runtime-mainline-ddim-public-10-step10/summary.json), [recon-runtime-ddim-public-10-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-10-step10/artifact-mainline/summary.json), [recon-runtime-kandinsky-mainline](../experiments/recon-runtime-mainline-kandinsky-public-smoke/summary.json), [recon-runtime-kandinsky-artifact-mainline](../experiments/recon-runtime-mainline-kandinsky-public-smoke/artifact-mainline/summary.json) | 公开资产包（DOI: `10.5281/zenodo.13371475`）已经支撑 `Stable Diffusion + DDIM` 的 `10-sample public runtime-mainline` 与 `kandinsky_v22` 的最小真实 runtime-mainline；当前最强公开子集证据是 `DDIM public-10 step10`，指标为 `auc=0.82 / asr=0.55 / tpr@1%fpr=1.0`，主要阻塞转为更大公开子集扩展，以及 `target/shadow/member/non-member` 语义映射仍未最终核准 |
| `variation` | `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf` 对应 API-only 路线 | `code-ready + evidence-ready` | `plan-variation` / `probe-variation-assets` / `dry-run-variation` / `run-variation-synth-smoke` | [variation-synth-smoke](../experiments/variation-synth-smoke/summary.json) | 缺真实 API 凭据、调用预算和真实 query image 集 |

## 灰盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `secmi` | `2023-icml-secmi-membership-inference-diffusion-models.pdf` | `code-ready + evidence-ready` | `plan-secmi` / `probe-secmi-assets` / `prepare-secmi` / `dry-run-secmi` / `runtime-probe-secmi` / `run-secmi-synth-smoke` | [secmi-synth-smoke](../experiments/secmi-synth-smoke/summary.json), [secmi-synth-smoke-gpu](../experiments/secmi-synth-smoke-gpu/summary.json) | 缺真实 checkpoint、flagfile 和论文一致资产布局 |
| `pia` | `2024-iclr-pia-proximal-initialization.pdf` | `code-ready + evidence-ready` | `plan-pia` / `probe-pia-assets` / `dry-run-pia` / `runtime-probe-pia` / `run-pia-runtime-smoke` / `run-pia-synth-smoke` | [pia-runtime-smoke-cpu](../experiments/pia-runtime-smoke-cpu/summary.json), [pia-runtime-smoke-gpu](../experiments/pia-runtime-smoke-gpu/summary.json), [pia-synth-smoke-cpu](../experiments/pia-synth-smoke-cpu/summary.json), [pia-synth-smoke-gpu](../experiments/pia-synth-smoke-gpu/summary.json) | 缺真实 DDPM checkpoint 与数据集根目录 |

## 白盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `white-box lead` | `2025-popets-white-box-membership-inference-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺可访问 checkpoint、训练配置、样本级梯度或激活接口 |
| `finding-nemo` | `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺 neuron-level 分析接口和相应模型资产 |

## 当前判断

1. 黑盒方向已经不是纯调研状态，至少有三条可执行证据链。
2. `recon` 已经有统一 mainline smoke，并且现在补上了 dataset/checkpoint 直达 score artifact 的 runtime mainline 入口；`Stable Diffusion + DDIM` 已经从 `1-sample` 提升到 `10-sample public runtime-mainline`，`kandinsky_v22` 的最小真实 runtime-mainline 与 artifact-mainline 也已经产出 summary。
3. `DiT` 官方采样路径不再只停留在 `2-step` 极小 smoke，当前还存在本地 checkpoint 驱动的 [dit-sample-step10](../experiments/dit-sample-step10/summary.json)，说明官方 `DiT-XL/2 256x256` 权重在本机可重复跑到 `10-step`。
4. 纯黑盒主线还没有真实 benchmark-ready 结果，因为 `recon` 当前仍是公开子集级证据，target/shadow/member/non-member 的论文级语义映射尚未最终核准；`DiT` 虽然已经有 `step10` 官方采样证据，但还没有进入与成员推断直接相连的实验协议，`variation` 仍停在 synthetic smoke、artifact summary、upstream smoke 或 dry-run 级别。
5. 下一步优先级建议：
   - 先把 `recon` 的公开资产从 `10-sample` 再扩到更大样本规模，并核准 target/shadow/member/non-member 映射
   - 再把 `DiT` 从 `step10 sample-smoke` 往更强的本地 checkpoint 采样证据推进，然后再决定是推进 `variation` 真实 API 调用还是补 `clid` 的真实资产驱动执行

## 统一黑盒汇总

- [experiments/blackbox-status/summary.json](/D:/Code/DiffAudit/Project/experiments/blackbox-status/summary.json)
