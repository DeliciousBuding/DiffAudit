# DiffAudit在线索引

> 这份文件是 Feishu 总报告的唯一维护源。
> 不再在 Feishu 内手改长索引布局。
> 需要更新总报告时，只改这份 Markdown，再覆盖同步到：
> `https://www.feishu.cn/docx/ITzEdcyWSoXRqKxuLe3cx4yInEe`

**GitHub 仓库主页**：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit-Research/)

**更新时间**：
2026-04-06 16:47:42 +08:00

**文档用途**：供团队内部查看当前实验主线、已验证证据、关键阻塞和下一步。这里维护的是稳定可验证的实验状态索引，不再承载复杂长篇论文目录布局。

**同步基线提交**：
8651ea0

## 当前状态概览

| 模块 | 当前阶段 | 当前判断 | 主要阻塞 | 下一步 |
| --- | --- | --- | --- | --- |
| 黑盒 | runtime evidence expanding | 黑盒主线已推进到公开包可达到的最大子集 evidence：`recon` 已打通 `Stable Diffusion + DDIM` 的 `100-sample public runtime-mainline`；同时 `public-50` 仍保持当前最佳 AUC。`kandinsky_v22` 的最小真实 runtime-mainline 与 `DiT` 本地 checkpoint 驱动的 `step50 sample-smoke` 也已通。 | 公开资产仍未达到论文级 `target/shadow/member/non-member` 语义核准，且 `public-50` 与 `public-100` 的指标差异尚待解释；`Kandinsky` 仍停在最小 smoke。 | 先核准映射语义并解释 `public-50` / `public-100` 的指标变化，再继续补 `Kandinsky 10/10`。 |
| 灰盒 | code-ready + evidence-ready | `SecMI`、`PIA` 均已具备 planner / probe / dry-run / smoke 级可运行能力，可维持实现不断档。 | 缺真实 checkpoint、flagfile、数据集根目录和论文一致资产布局。 | 保持代码链可运行，等真实资产到位后优先补单条高质量复现。 |
| 白盒 | research-ready | 白盒主论文与解释型路线已完成详细阅读和索引整理，方向判断已经明确。 | 缺 checkpoint、训练配置、梯度/激活接口与复现实验资产。 | 先补访问条件，再决定优先复现哪条白盒线。 |
| 防御/综述 | indexed | 防御和综述材料已可作为后续选题与叙事背景，不再是当前主执行线。 | 尚未全部映射为仓库实验路线。 | 按黑盒主线推进情况，择机挑选少量条目进入下一轮实现。 |

## 当前主线

当前实验主线仍然是黑盒方向中的微调扩散模型成员推断。仓库状态已经从“最小链路打通”推进到“公开中样本 evidence”，并且现在已经摸到当前公开包可支持的 `100-sample` 上限。

当前最大公开子集证据：

- 方法：`recon`
- 模型：`Stable Diffusion v1.5 + DDIM`
- workspace：`experiments/recon-runtime-mainline-ddim-public-100-step10`
- `auc = 0.788`
- `asr = 0.63`
- `tpr@1%fpr = 0.99`

当前最佳 AUC 仍来自：

- workspace：`experiments/recon-runtime-mainline-ddim-public-50-step10`
- `auc = 0.866`

## 关键入口

- 复现状态总览：[docs/reproduction-status.md](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/docs/reproduction-status.md)
- 黑盒统一结果摘要：[experiments/blackbox-status/summary.json](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/experiments/blackbox-status/summary.json)
- 黑盒主线计划：[workspaces/black-box/plan.md](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/workspaces/black-box/plan.md)
- 论文索引源文件：[references/materials/paper-index.md](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/paper-index.md)
- 飞书总报告源文件：[docs/paper-reports/master-feishu-index.md](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/docs/paper-reports/master-feishu-index.md)

## 当前判断

1. `recon` 已经不是 smoke 级验证，而是公开中样本 evidence 级主线，并且当前公开包已经推进到 `public-100`。
2. `Stable Diffusion + DDIM` 的 `public-50` 与 `public-100` 现在形成了第一组可比较证据，下一步应解释为什么 AUC 回落而 ASR 上升。
3. `DiT step50` 已经把模型覆盖往前推进了一档，但 `kandinsky_v22` 仍只有最小可验证证据，还没有达到和 `SD/DDIM public-50/public-100` 同等级的主证据强度。
4. `variation` 与 `CLiD` 目前仍停在 synthetic / artifact / dry-run 级别，不应抢主线资源。
5. 当前真正限制项目从 evidence-ready 走向 benchmark-ready 的，仍然是公开资产语义映射核准，而不是 CLI 或 smoke 缺口。

## 下一步

1. 核准公开资产的 `target/shadow/member/non-member` 语义映射，并明确 `public-50` / `public-100` 的解释口径。
2. 视 GPU 窗口优先把 `Kandinsky` 扩到 `10/10`，再决定 `DiT` 是否继续往更高步数推进。
3. 在主线解释清楚之前，不让 `variation` / `CLiD` 抢占黑盒主线资源。
