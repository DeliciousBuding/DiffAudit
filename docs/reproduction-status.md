# 复现状态总览

这份文档用于汇总仓库当前各条攻击线的真实推进状态。

如果你只想先看一页综合判断，而不是逐线细节，优先看 [comprehensive-progress.md](comprehensive-progress.md)。

口径约束：

- 黑盒、灰盒、白盒三条线都已纳入统一研究规划
- 统一规划只表示都在正式路线图内，不表示当前资源平均分配
- 当前执行优先级仍是黑盒第一，灰盒第二层维护，白盒以研究准备为主

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
| `recon` | `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf` 与 `external/Reconstruction-based-Attack` | `code-ready + evidence-ready` | `plan-recon` / `probe-recon-assets` / `dry-run-recon` / `run-recon-mainline-smoke` / `probe-recon-runtime-assets` / `run-recon-runtime-mainline` / `probe-recon-score-artifacts` / `run-recon-artifact-mainline` / `run-recon-eval-smoke` / `summarize-recon-artifacts` / `run-recon-upstream-eval-smoke` | [recon-mainline-smoke](../experiments/recon-mainline-smoke/summary.json), [recon-runtime-ddim-public-10](../experiments/recon-runtime-mainline-ddim-public-10-step10/summary.json), [recon-runtime-ddim-public-25](../experiments/recon-runtime-mainline-ddim-public-25-step10/summary.json), [recon-runtime-ddim-public-50](../experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json), [recon-runtime-ddim-public-100-step10](../experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json), [recon-runtime-ddim-public-100-step30](../experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json), [recon-runtime-ddim-public-100-step30-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-100-step30/artifact-mainline/summary.json), [recon-runtime-kandinsky-mainline](../experiments/recon-runtime-mainline-kandinsky-public-smoke/summary.json) | 公开资产包已经支撑 `Stable Diffusion + DDIM` 的 `100-sample public runtime-mainline` 与 `kandinsky_v22` 的最小真实 runtime-mainline；当前最可信主证据已切到 `DDIM public-100 step30`，但 `target/shadow/member/non-member` 语义映射仍未最终核准，且 `step10` / `step30` 只能解释为“校正后证据差异”，不能直接外推成采样步数单调提升结论 |
| `variation` | `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf` 对应 API-only 路线 | `code-ready + evidence-ready` | `plan-variation` / `probe-variation-assets` / `dry-run-variation` / `run-variation-synth-smoke` | [variation-synth-smoke](../experiments/variation-synth-smoke/summary.json) | 缺真实 API 凭据、调用预算和真实 query image 集 |

## 灰盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `secmi` | `2023-icml-secmi-membership-inference-diffusion-models.pdf` | `code-ready + evidence-ready` | `plan-secmi` / `probe-secmi-assets` / `prepare-secmi` / `dry-run-secmi` / `runtime-probe-secmi` / `run-secmi-synth-smoke` | [secmi-synth-smoke](../experiments/secmi-synth-smoke/summary.json), [secmi-synth-smoke-gpu](../experiments/secmi-synth-smoke-gpu/summary.json) | 缺真实 checkpoint、flagfile 和论文一致资产布局 |
| `pia` | `2024-iclr-pia-proximal-initialization.pdf` | `asset-ready + evidence-ready` | `plan-pia` / `probe-pia-assets` / `dry-run-pia` / `runtime-probe-pia` / `runtime-preview-pia` / `run-pia-runtime-mainline` / `run-pia-runtime-smoke` / `run-pia-synth-smoke` | [pia-runtime-smoke-cpu](../experiments/pia-runtime-smoke-cpu/summary.json), [pia-runtime-smoke-gpu](../experiments/pia-runtime-smoke-gpu/summary.json), [pia-synth-smoke-cpu](../experiments/pia-synth-smoke-cpu/summary.json), [pia-synth-smoke-gpu](../experiments/pia-synth-smoke-gpu/summary.json), [pia-template-followup](../workspaces/gray-box/2026-04-07-pia-followup.md), [pia-real-asset-probe](../workspaces/gray-box/2026-04-07-pia-real-asset-probe.md), [pia-runtime-mainline](../workspaces/gray-box/2026-04-07-pia-runtime-mainline.md), [pia-intake-gate](../workspaces/gray-box/pia-intake-gate.md) | 当前最强口径已升级为“single-machine real-asset runtime mainline ready”；baseline `auc=0.90625 / asr=0.875 / tpr@1%fpr=0.75` 已落到 [pia-cifar10-runtime-mainline-20260407-cpu](../workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260407-cpu/summary.json)，并由 `workspaces/gray-box/assets/pia/manifest.json` 记录 contract_stage、asset_grade、provenance_status 以供系统 intake；由于 provenance 仍是 `source-retained-unverified`，尚不能升级为 `paper-aligned` |

## 白盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `gsa` | `2025-popets-white-box-membership-inference-diffusion-models.pdf` | `asset-ready + evidence-ready` | `probe-gsa-assets` / `run-gsa-runtime-mainline` | [gsa-kickoff](../workspaces/white-box/2026-04-06-gsa-kickoff.md), [gsa-closed-loop-smoke](../workspaces/white-box/2026-04-07-gsa-closed-loop-smoke.md), [gsa-asset-intake](../workspaces/white-box/2026-04-07-gsa-asset-intake.md), [gsa-runtime-mainline](../workspaces/white-box/2026-04-07-gsa-runtime-mainline.md) | 当前最强说法已升级为“CPU 上 real-asset closed loop ready”；canonical 资产根在 [workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json](../workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json) 记录了 dataset roots、checkpoint directories、split counts 以及 provenance，真实 `checkpoint-*` 与四组 CIFAR10 bucket 已跑出 [gsa-runtime-mainline-20260407-cpu](../workspaces/white-box/runs/gsa-runtime-mainline-20260407-cpu/summary.json)，但当前指标 `auc=0.5 / asr=0.5` 仅代表第一批极小本地资产闭环，不代表论文级结果 |
| `finding-nemo` | `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺 neuron-level 分析接口和相应模型资产 |

## 当前判断

1. 当前仓库不是“只做黑盒”，而是“黑/灰/白三线统一规划、分层执行”；其中黑盒仍是当前第一优先执行线，灰盒维持可运行与待资产闭环状态，白盒维持研究准备态。
2. 黑盒主线当前不是“继续扩实验规模”，而是“固化 `recon` 公开子集的最终口径并保持文档一致”。
3. `recon` 当前最可信主证据是 [recon-runtime-mainline-ddim-public-100-step30](../experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json)，指标为 `auc=0.849 / asr=0.51 / tpr@1%fpr=1.0`；它之所以比 `step10` 更应被当作主证据，不是因为所有指标都更高，而是因为这次收口修正了 `shadow_non_member` 半成品目录被误复用的问题，并补齐了对应 artifact-mainline 证据。
4. [recon-runtime-mainline-ddim-public-100-step10](../experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json) 仍然是有价值的对照基线，指标为 `auc=0.788 / asr=0.63 / tpr@1%fpr=0.99`；`step30` 相比它提升了排序相关指标，但 `asr` 更低，因此当前只能说“修正后的 `step30` 证据更完整”，不能直接声称“更多采样步数必然更强”。
5. `recon` 当前最佳 AUC 仍来自 [recon-runtime-mainline-ddim-public-50-step10](../experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json) 的 `0.866`。因此，“最大公开子集主证据”与“当前最佳单指标结果”是两个不同结论，不能混写。
6. `kandinsky_v22` 的最小真实 runtime-mainline 已经有证据，但 `10/10` 与单样本诊断都仍异常慢；在拿到能定位首个阶段耗时的有效日志前，不应默认继续占用 GPU。
7. `DiT` 官方采样路径已经有 [dit-sample-step10](../experiments/dit-sample-step10/summary.json) 和 [dit-sample-step50](../experiments/dit-sample-step50/summary.json)，说明模型覆盖证据在推进，但它还没有进入成员推断协议，不应被写成黑盒主线已经闭环。
8. `PIA` 的同日时间线现在要分三层读：[pia-template-followup](../workspaces/gray-box/2026-04-07-pia-followup.md) 是模板仍 blocked；[pia-real-asset-probe](../workspaces/gray-box/2026-04-07-pia-real-asset-probe.md) 是 canonical 本地资产 probe/preview ready；[pia-runtime-mainline](../workspaces/gray-box/2026-04-07-pia-runtime-mainline.md) 才是第一份真实 mainline 结果，并由 `workspaces/gray-box/assets/pia/manifest.json` 记录 contract_stage / asset_grade / provenance_status 供系统 intake；当前最稳妥的口径应是“single-machine real-asset runtime mainline ready”。
9. `PIA` 的第一版灰盒防御原型也已通过同一入口落盘：[pia-cifar10-runtime-mainline-dropout-defense-20260407-cpu](../workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260407-cpu/summary.json) 证明了 `stochastic-dropout` 钩子可运行，但当前 tiny local run 下并未带来指标改进，不能把它写成已验证有效防御。
10. `GSA` 当前最强说法已不再是 [gsa-closed-loop-smoke](../workspaces/white-box/2026-04-07-gsa-closed-loop-smoke.md) 的 toy executable，而是 [gsa-runtime-mainline](../workspaces/white-box/2026-04-07-gsa-runtime-mainline.md) 所代表的“CPU 上 real-asset closed loop ready”，并由 `workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json` 向系统契约层提供 canonical roots、checkpoint directories 和 split counts。
11. 新资源对白盒的帮助已从“只能补数据桶”推进到真实资产闭环：`CIFAR-10` 归档被转成四组真实 bucket，`GSA` 自训生成了兼容的 `checkpoint-*` 目录，当前白盒阻塞已不再是 checkpoint format mismatch，而是下一轮应扩大样本量和训练强度。
12. 白盒当前仍不能写成论文复现成功。`gsa-runtime-mainline` 的当前弱指标只说明第一批极小本地资产闭环已通，不说明攻击本身无效，更不说明论文结果被推翻。

## 下一步

1. 固化 `recon` 公开资产映射与 `public-100 step10` / `step30` 的解释口径，确保 [README.md](../README.md)、[ROADMAP.md](../ROADMAP.md)、[workspaces/black-box/plan.md](../workspaces/black-box/plan.md) 和 [experiments/blackbox-status/summary.json](../experiments/blackbox-status/summary.json) 一致。
2. 把 `target/shadow/member/non-member` 的当前最可辩护语义继续收口到 [recon-public-asset-mapping.md](recon-public-asset-mapping.md)，在拿到更强证据前不要越权声称论文语义已核准。
3. 继续暂停 `Kandinsky 10/10`，直到先拿到能定位首个阶段耗时的有效日志；如果只能做一件事，优先做文档固化而不是新 GPU 任务。
4. `PIA` 的单机真实资产已经升级成真实 `runtime-mainline`。下一步不是再补 runner，而是先继续核准 provenance，并扩大 `num_samples` 与 defense 对照规模。
5. `GSA` 已从“端到端可执行”推进到“real-asset closed loop ready”。下一步不是再解释 checkpoint format mismatch，而是扩大 bucket 数量、训练 epoch 和 closed-loop 统计稳定性。

更新时间：`2026-04-07 23:59:00 +08:00`
