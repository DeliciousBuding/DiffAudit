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
| `clid` | `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf` | `evidence-ready` | `plan-clid` / `probe-clid-assets` / `dry-run-clid` / `run-clid-dry-run-smoke` / `summarize-clid-artifacts` | [clid-dry-run-smoke](/D:/Code/DiffAudit/Project/experiments/clid-dry-run-smoke/summary.json), [clid-artifact-summary](/D:/Code/DiffAudit/Project/experiments/clid-artifact-summary/summary.json) | 缺真实 text-to-image 资产、缺可重复的目标模型路径与数据集映射 |
| `recon` | `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf` 与 `external/Reconstruction-based-Attack` | `code-ready + evidence-ready` | `plan-recon` / `probe-recon-assets` / `dry-run-recon` / `run-recon-eval-smoke` / `summarize-recon-artifacts` | [recon-eval-smoke](/D:/Code/DiffAudit/Project/experiments/recon-eval-smoke/summary.json), [recon-artifact-summary](/D:/Code/DiffAudit/Project/experiments/recon-artifact-summary/summary.json) | 缺真实 LoRA checkpoint、生成图像目录、embedding 输入和目标/影子分割资产 |
| `variation` | `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf` 对应 API-only 路线 | `code-ready + evidence-ready` | `plan-variation` / `probe-variation-assets` / `dry-run-variation` / `run-variation-synth-smoke` | [variation-synth-smoke](/D:/Code/DiffAudit/Project/experiments/variation-synth-smoke/summary.json) | 缺真实 API 凭据、调用预算和真实 query image 集 |

## 灰盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `secmi` | `2023-icml-secmi-membership-inference-diffusion-models.pdf` | `code-ready + evidence-ready` | `plan-secmi` / `probe-secmi-assets` / `prepare-secmi` / `dry-run-secmi` / `runtime-probe-secmi` / `run-secmi-synth-smoke` | [secmi-synth-smoke](/D:/Code/DiffAudit/Project/experiments/secmi-synth-smoke/summary.json), [secmi-synth-smoke-gpu](/D:/Code/DiffAudit/Project/experiments/secmi-synth-smoke-gpu/summary.json) | 缺真实 checkpoint、flagfile 和论文一致资产布局 |
| `pia` | `2024-iclr-pia-proximal-initialization.pdf` | `code-ready + evidence-ready` | `plan-pia` / `probe-pia-assets` / `dry-run-pia` / `runtime-probe-pia` / `run-pia-runtime-smoke` / `run-pia-synth-smoke` | [pia-runtime-smoke-cpu](/D:/Code/DiffAudit/Project/experiments/pia-runtime-smoke-cpu/summary.json), [pia-runtime-smoke-gpu](/D:/Code/DiffAudit/Project/experiments/pia-runtime-smoke-gpu/summary.json), [pia-synth-smoke-cpu](/D:/Code/DiffAudit/Project/experiments/pia-synth-smoke-cpu/summary.json), [pia-synth-smoke-gpu](/D:/Code/DiffAudit/Project/experiments/pia-synth-smoke-gpu/summary.json) | 缺真实 DDPM checkpoint 与数据集根目录 |

## 白盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `white-box lead` | `2025-popets-white-box-membership-inference-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺可访问 checkpoint、训练配置、样本级梯度或激活接口 |
| `finding-nemo` | `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺 neuron-level 分析接口和相应模型资产 |

## 当前判断

1. 黑盒方向已经不是纯调研状态，至少有三条可执行证据链。
2. `clid` 的 artifact summary 是当前最接近论文结果复算的黑盒证据。
3. 纯黑盒主线还没有真实 benchmark-ready 结果，因为 `recon` 和 `variation` 还停在 synthetic smoke、artifact summary 或 dry-run 级别。
4. 下一步优先级建议：
   - 先给 `recon` 接真实生成与 embedding artifact
   - 再决定是推进 `variation` 真实 API 调用，还是补 `clid` 的真实资产驱动执行
