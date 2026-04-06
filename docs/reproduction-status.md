# 复现状态总览

这份文档用于汇总仓库当前各条攻击线的真实推进状态。

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
| `pia` | `2024-iclr-pia-proximal-initialization.pdf` | `code-ready + evidence-ready` | `plan-pia` / `probe-pia-assets` / `dry-run-pia` / `runtime-probe-pia` / `run-pia-runtime-smoke` / `run-pia-synth-smoke` | [pia-runtime-smoke-cpu](../experiments/pia-runtime-smoke-cpu/summary.json), [pia-runtime-smoke-gpu](../experiments/pia-runtime-smoke-gpu/summary.json), [pia-synth-smoke-cpu](../experiments/pia-synth-smoke-cpu/summary.json), [pia-synth-smoke-gpu](../experiments/pia-synth-smoke-gpu/summary.json), [pia-template-probe-20260407](../workspaces/gray-box/runs/pia-followup-20260407/probe.json), [pia-template-dry-run-20260407](../workspaces/gray-box/runs/pia-followup-20260407/dry-run.json) | 缺真实 DDPM checkpoint 与数据集根目录；`2026-04-07` 的 template probe 重新确认当前剩余缺口就是 `checkpoint / dataset_root / dataset layout`，不是 repo 布局或 member split |

## 白盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `gsa` | `2025-popets-white-box-membership-inference-diffusion-models.pdf` | `code-ready + evidence-ready` | `workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py` / `workspaces/white-box/external/GSA/test_attack_accuracy.py` | [gsa-kickoff](../workspaces/white-box/2026-04-06-gsa-kickoff.md), [gsa-gradient-smoke-artifact](../workspaces/white-box/smoke-ddpm/member-gradients.pt), [gsa-closed-loop-smoke](../workspaces/white-box/runs/gsa-closed-loop-smoke-20260407-cpu/summary.json) | 已完成 toy CPU closed-loop smoke，但仍缺论文对齐的 `target/shadow` checkpoints、`member/non-member` 划分以及统一 adapter / CLI 接入 |
| `finding-nemo` | `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺 neuron-level 分析接口和相应模型资产 |

## 当前判断

1. 当前仓库不是“只做黑盒”，而是“黑/灰/白三线统一规划、分层执行”；其中黑盒仍是当前第一优先执行线，灰盒维持可运行与待资产闭环状态，白盒维持研究准备态。
2. 黑盒主线当前不是“继续扩实验规模”，而是“固化 `recon` 公开子集的最终口径并保持文档一致”。
3. `recon` 当前最可信主证据是 [recon-runtime-mainline-ddim-public-100-step30](../experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json)，指标为 `auc=0.849 / asr=0.51 / tpr@1%fpr=1.0`；它之所以比 `step10` 更应被当作主证据，不是因为所有指标都更高，而是因为这次收口修正了 `shadow_non_member` 半成品目录被误复用的问题，并补齐了对应 artifact-mainline 证据。
4. [recon-runtime-mainline-ddim-public-100-step10](../experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json) 仍然是有价值的对照基线，指标为 `auc=0.788 / asr=0.63 / tpr@1%fpr=0.99`；`step30` 相比它提升了排序相关指标，但 `asr` 更低，因此当前只能说“修正后的 `step30` 证据更完整”，不能直接声称“更多采样步数必然更强”。
5. `recon` 当前最佳 AUC 仍来自 [recon-runtime-mainline-ddim-public-50-step10](../experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json) 的 `0.866`。因此，“最大公开子集主证据”与“当前最佳单指标结果”是两个不同结论，不能混写。
6. `kandinsky_v22` 的最小真实 runtime-mainline 已经有证据，但 `10/10` 与单样本诊断都仍异常慢；在拿到能定位首个阶段耗时的有效日志前，不应默认继续占用 GPU。
7. `DiT` 官方采样路径已经有 [dit-sample-step10](../experiments/dit-sample-step10/summary.json) 和 [dit-sample-step50](../experiments/dit-sample-step50/summary.json)，说明模型覆盖证据在推进，但它还没有进入成员推断协议，不应被写成黑盒主线已经闭环。
8. `secmi / pia` 仍在统一规划里，当前状态是“代码与 smoke 证据已具备，但缺真实论文资产”；它们不是被移出路线图，而是暂未进入第一执行层。`2026-04-07` 的 [pia-template-probe-20260407](../workspaces/gray-box/runs/pia-followup-20260407/probe.json) 与 [pia-template-dry-run-20260407](../workspaces/gray-box/runs/pia-followup-20260407/dry-run.json) 进一步说明：当前需要补的是真实 `checkpoint / dataset_root / dataset layout`，不是 GPU，也不是上游仓库结构。
9. `GSA` 当前已经有 [gsa-closed-loop-smoke](../workspaces/white-box/runs/gsa-closed-loop-smoke-20260407-cpu/summary.json)，说明 CPU 上的 `gradient extraction -> xgboost classifier` 管线可以闭环；但这只是 toy synthetic assets 上的 `closed-loop-smoke`，不能写成论文级白盒结果。
10. 白盒相关论文仍在统一规划里，当前状态是“研究问题与资产条件已整理，且 `GSA` 已经从 `gradient-smoke` 推进到 `closed-loop-smoke`”；在 paper-aligned checkpoints 和划分资产未满足前，不应写成即将复现成功。

## 下一步

1. 固化 `recon` 公开资产映射与 `public-100 step10` / `step30` 的解释口径，确保 [README.md](../README.md)、[ROADMAP.md](../ROADMAP.md)、[workspaces/black-box/plan.md](../workspaces/black-box/plan.md) 和 [experiments/blackbox-status/summary.json](../experiments/blackbox-status/summary.json) 一致。
2. 把 `target/shadow/member/non-member` 的当前最可辩护语义继续收口到 [recon-public-asset-mapping.md](recon-public-asset-mapping.md)，在拿到更强证据前不要越权声称论文语义已核准。
3. 继续暂停 `Kandinsky 10/10`，直到先拿到能定位首个阶段耗时的有效日志；如果只能做一件事，优先做文档固化而不是新 GPU 任务。
4. 用真实配置推进 `PIA` 的 `probe-pia-assets -> runtime-probe-pia --device cpu`，在 `checkpoint / dataset_root / dataset layout` 任一项缺失时保持 `blocked`，不要回退到重复的 GPU smoke。
5. 把 `GSA` 保持在 `closed-loop-smoke` 阶段，下一步只做 paper-aligned `target/shadow + member/non-member + checkpoint` 资产接入，不把 toy synthetic 闭环误写成 benchmark。

更新时间：`2026-04-07 17:55:00 +08:00`
