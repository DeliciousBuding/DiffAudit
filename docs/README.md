# 文档目录

- `runtime.md`: Runtime 执行层运行说明（活跃入口位于 `Services/Local-API`，目录名待迁移到 `Services/Runtime`）
- `asset-registry-local-api.md`: minimal asset registry intake contract for Runtime legacy compatibility
- `reproduction-status.md`: per-track reproduction status
- `comprehensive-progress.md`: one-page integrated progress view for the research repo
- `mainline-narrative.md`: current research storyline, claimable contributions, and presentation-ready framing
- `storage-boundary.md`: where external code, raw downloads, lane assets, and run evidence should live
- `data-and-assets-handoff.md`: how a teammate gets the same datasets, weights, supplementary bundles, and local path bindings
- `recon-artifact-replay-guidance.md`: separation notes for Runtime recon-artifact replay directories versus admitted evidence
- `future-phase-e-intake.md`: ordered intake queue, entry gates, and expected outputs for the next research question
- `next-run-intake-index.md`: next-run intake gate entrypoints and contracts
- `mia-defense-research-index.md`: route, literature, and asset mapping for the MIA defense strategy document
- `mia-defense-execution-checklist.md`: actionable execution checklist derived from the MIA defense strategy document
- `mentor-strict-reproduction-plan.md`: strict parallel reproduction plan aligned with mentor guidance
- `teammate-setup.md`: teammate bring-up guide for a portable research environment
- `handoff.md`: cross-session handoff transcript for takeover and historical decision replay

这里存放项目级文档，而不是个人临时笔记。

临时 `prompt/context/work package` 不应长期留在 `docs/`。

需要长期保留的外部模型批量输出，只保留结果包，统一放在 `report-bundles/`。

## 建议放什么

- 环境说明
- GitHub 协作规范
- 入手指南
- 架构说明
- 实验协议
- benchmark 定义
- 设计决策记录
- 长期有参考价值的外部研究结果包（只保留结果，不保留临时工作包）

## 当前核心文档

- [reproduction-status.md](reproduction-status.md)：当前各条方法线的真实推进状态
- [comprehensive-progress.md](comprehensive-progress.md)：研究仓库的综合进度入口
- [mainline-narrative.md](mainline-narrative.md)：当前主线叙事、可主张创新点与答辩/PPT 素材入口
- [storage-boundary.md](storage-boundary.md)：`external / third_party / Download / workspaces/*/assets / runs` 的存放边界
- [data-and-assets-handoff.md](data-and-assets-handoff.md)：新同学获取同样数据集/权重/补充包并绑定本机路径的入口
- [future-phase-e-intake.md](future-phase-e-intake.md)：`Phase E` 候选池排序、进入条件与退出条件
- [next-run-intake-index.md](next-run-intake-index.md)：可运行 next-run 入口索引（用于挂任务板的一句话口径）
- [repo-map.md](repo-map.md)：仓库目录地图和代码职责说明
- [environment.md](environment.md)：环境与依赖说明
- [getting-started.md](getting-started.md)：新成员上手指南
- [github-collaboration.md](github-collaboration.md)：协作与分支规范
- [mia-defense-research-index.md](mia-defense-research-index.md)：`mia-defense-document.docx` 的正式研究索引
- [mia-defense-execution-checklist.md](mia-defense-execution-checklist.md)：基于当前仓库状态整理的执行清单
- [mentor-strict-reproduction-plan.md](mentor-strict-reproduction-plan.md)：严格按师兄方案收口的三线并行复现计划
- [teammate-setup.md](teammate-setup.md)：队友接仓与环境自检指南
- [handoff.md](handoff.md)：跨 session 交接记录与历史决策回放入口

## 阶段性结果包

- `report-bundles/gpt54/round1-results/`：GPT-5.4 第一轮原始结果
- `report-bundles/gpt54/round2-results/`：GPT-5.4 第二轮原始结果与扩展问题结果
