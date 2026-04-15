# Recon Artifact Replay Clarification

## 这些 replay artifacts 是什么
- 路径 `Research/experiments/recon-artifact-mainline-YYYYMMDD-HHMMSS/` 对应近期从 Local-API `recon_artifact_mainline` job 触发的 replay 资产。每个目录仅包含 `summary.json`，记录 contract `black-box/recon/sd15-ddim`、method（当前为 `threshold`）以及 `artifact_dir` 的本地 score-artifacts 位置。它们只复现特定 `recon-runtime-mainline` 结果的 artifact-generation 阶段，用于调试、验证阈值、核对 replay 流程以及备份 Local-API 的 replay trace。
- 这些 replay 不是新的模型 checkpoint，也没有通过 `Research/docs/admitted-results-summary.md` 里列出的 admitted 复现流程，它们只是 Local-API 输出的 replay trace 的自说明记录。

## 为什么它们不能替代 admitted main evidence
- replay 目录内的 artefact 没有经过跨团队审查、比赛导师确认或主线 narrative 的最终 gating。它们没有被纳入 `Research/docs/competition-evidence-pack.md` 里的 admitted 视觉证明，也没有被列入任何 mainline narrative claim，因此不能作为比赛主讲或对外证明的 evidence。
- 由于它们直接对应的是 `Local-API` 的内部 replay step，而非 end-to-end 合法复现 pipeline（没有 downstream review、admitted-checklist、mission logs），所以不能用来替代 `admitted-results-summary` 中列出的任何一个确认过的 defense claim，否则会混淆“admitted”“demo/replay”两层语义。

## Leader / Developer 应如何引用
- Leader 在统筹汇报材料、路线规划或将来对外 demo 时，如果需要提到这些 replay 资产，要把它们明确跨号为“Local-API recon replay / demo-level artifact”并说明“仅供 Research 内部审查”，绝不纳入 admitted claim 列表，也不用于比赛讲稿除了说明 “我们留存了 replay 记录” 之外。
- Developer 在准备系统展示、Local-API 现象级问题复盘或写 `workspaces/local-api/` intake 时，才可以引用这些目录以便描述可再现的 replay 流程。但在任何 `Platform`/`Services` 级别的 material（包括 `docs/competition-innovation-summary.md` 或 demo screencast）里，应使用 `admitted-results` 的主线 artifact；若同时演示 replay 的机制，请把它们呈现为 debugging/demo artifact，配合醒目标注“未经过 adjudication”。
- 所有 replay artifacts 继续存放在 `Research/experiments/` 中，并通过本 doc 附带的说明进行组织；如需要在 `docs/README.md` 或章节索引中引用，请链接到本条以保持 Research 里 admitted vs demo 资产的清晰边界。
