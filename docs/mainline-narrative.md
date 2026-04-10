# 主线叙事与项目总览

这份文档用于承接当前 DiffAudit 的研究叙事、答辩/PPT 话术和对外说明素材。

它不是 `ROADMAP`，也不是运行态单一来源。阶段、gate、owner 仍以路线图和各条线的工作区文档为准；这里的职责是把“我们到底在做什么、三条线如何配合、当前能主张什么”讲清楚。

## 当前一句话

DiffAudit 正在构建一个面向扩散模型成员推断风险的分层审计框架：黑盒 `recon` 证明风险真实存在，灰盒 `PIA` 提供当前最成熟的攻击-防御主讲闭环，白盒 `GSA + W-1` 则用于量化风险上界和探索更严格的同协议防御比较。

## 项目整体理解

这个项目不是单篇论文复现，也不是单一攻击脚本集合。

当前真正要交付的是三件事：

- 一条能稳定讲清楚的算法主线：`PIA + provisional G-1(all_steps)`
- 一套结构化证据链：`summary.json -> intake manifest -> unified attack-defense table`
- 一条能被系统消费的研究结果读链：`Project -> Local-API -> Platform`

因此，研究与工程是耦合的：实验结果不能只存在于临时日志里，而要能够进入 admitted 结果、被 `Local-API` 查询、被平台解释，并保持跨文档口径一致。

## 三条研究线如何配合

### 黑盒：风险证据线

- 主方法：`recon`
- 当前角色：证明在最弱权限假设下也能观测到成员泄露风险
- 当前价值：作为申报和答辩中的“风险确实存在”证据，而不是攻防闭环主讲线

### 灰盒：主讲闭环线

- 主方法：`PIA`
- 当前角色：最成熟、最适合讲“攻击信号 + 防御原型 + 系统消费”闭环
- 当前价值：项目的算法主讲线
- 当前关键口径：
  - 攻击信号是 `epsilon-trajectory consistency`
  - 防御原型是 `stochastic-dropout`
  - strongest claim 仍是 `workspace-verified + adaptive-reviewed`

### 白盒：深度与上界线

- 主方法：`GSA`
- 防御候选：`W-1 = DPDM`
- 当前角色：提供最坏情况下的风险上界，以及更强的机制解释空间
- 当前价值：不是申报阶段唯一主讲成果，而是“我们已经打通更深权限层攻击/防御分析”的技术厚度

## 当前可主张

### 1. 分层威胁模型审计框架

项目已经稳定形成 `black-box / gray-box / white-box` 三条权限线，并通过统一指标与 admitted 结果表来组织结果。这使得审计不是“单论文单攻击”的局部复现，而是“按权限层级逐步增强”的系统性比较。

### 2. `PIA` 的可解释灰盒主讲闭环

当前灰盒最强线已经不是 smoke，而是真实资产 mainline。`PIA` 的攻击信号可以被解释为推理轨迹一致性，而 `stochastic-dropout` 的当前机理解释是打散这一致性信号。这构成了当前最完整的“信号解释 + 防御方向 + 系统读链”闭环。

但这条线当前仍有一个必须显式带出的边界：

- strongest claim 仍然只是 `workspace-verified + adaptive-reviewed`
- `paper-aligned` 仍被 `checkpoint/source provenance` 单独阻塞

### 3. 研究结果到系统读链的结构化证据链

当前结果不只是跑出指标，而是被结构化为 `summary.json`、`intake manifest`、`unified attack-defense table`，再进入 `Local-API` 和平台。这使得结论可以被程序消费、被非研究侧解释，而不是停留在零散实验输出。

## 若 bridge 收口后可升级主张

下面这些表述当前还不能写成已成立事实，但如果 white-box bridge 继续收口，可以升级为更强主张：

- `GSA rerun1` 与 `W-1` 在更完整 same-protocol 条件下形成正式 benchmark 对照
- 白盒防御比较从“已有 defended comparator”升级到“已形成可引用 bridge 结果”
- `Finding NeMo` 等白盒延展线从 research-ready 升级到执行层

## 系统三层架构为什么重要

### Project

- 是研究 truth source
- 负责产出实验结果、manifest、统一总表、研究文档

### Local-API

- 是研究结果的受控消费层
- 负责把 `Project` 的 admitted 结果暴露成稳定接口，而不是让调用方直接理解实验目录结构

### Platform

- 是结果展示和解释层
- 默认消费 admitted 结果，不复制研究逻辑

这三层解耦的意义是：研究可以继续迭代，系统接口保持稳定，展示层不需要理解实验细节就能拿到可靠口径。

## 当前最值得讲的项目故事

目前最合理的对外叙事是：

1. 扩散模型存在可观测的成员泄露风险，黑盒 `recon` 已经给出主证据。
2. 灰盒 `PIA` 是当前最成熟的主讲闭环，既能解释攻击信号，也已有防御原型。
3. 白盒 `GSA` 给出近乎上界的攻击强度，`W-1 = DPDM` 提供方向上有效的防御比较。
4. 当前白盒同协议 bridge 已经从训练链阻塞推进到 `batch_size = 32` 可恢复 `shadow-02/03` checkpoint，并产出第一份 diagnostic comparator summary；随后该 route 已正式以 `保持冻结` 收口，但这仍只是治理与资源排序决策，不是 benchmark 完成。

## 当前 4-8 周执行顺序

未来 `2026-04-09` 到 `2026-06-04` 的真实顺序不是“多开新题”，而是：

1. 将 [2026-04-09-pia-provenance-dossier](../workspaces/gray-box/2026-04-09-pia-provenance-dossier.md) 固定为 `remain long-term blocker`
2. 用 [2026-04-10-recon-explanation-layer](../workspaces/black-box/2026-04-10-recon-explanation-layer.md) 吸收 `CopyMark` 现实边界与频域解释层
3. 用 [future-phase-e-intake](future-phase-e-intake.md) 固定 `Phase E` 候选池排序，并在执行层默认先看 `Finding NeMo + local memorization + FB-Mem`
3.1 用 [2026-04-10-finding-nemo-mechanism-intake](../workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md) 固定其 `intake/eligibility` gate，而不是提前把它写成机制线已开跑
3.2 用 [2026-04-10-finding-nemo-protocol-reconciliation](../workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md) 固定“当前 admitted 白盒资产与原始 NeMo 协议面并不对齐”的边界
3.3 用 [2026-04-10-finding-nemo-observability-smoke-contract](../workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md) 固定未来 smoke 的入口合同，而不是默认把候选写成已获 GPU release
3.4 用 [2026-04-10-finding-nemo-activation-only-canary-sketch](../workspaces/white-box/2026-04-10-finding-nemo-activation-only-canary-sketch.md) 固定实现边界，再用 [2026-04-10-finding-nemo-activation-export-adapter-review](../workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md) 正式固定 `zero-GPU hold / run 未放行`
4. 用 [2026-04-10-recon-explanation-layer](../workspaces/black-box/2026-04-10-recon-explanation-layer.md) 把 `CopyMark` 与频域论文固定在 boundary / explanation 层，不让它们漂移成新黑盒执行线
5. 在 provenance 条件未变化前，不放行 `PIA paper-aligned confirmation`

## 当前不应过度主张

- 不应写“白盒 same-protocol benchmark 已完成”
- 不应写“`Finding NeMo` 已进入当前执行主线”
- 不应写“`Finding NeMo` 已 execution-ready、benchmark-ready 或已获得 GPU release”
- 不应把任何尚未进入 admitted 合同的诊断结果写成系统正式主结果
- 不应再把黑盒写成当前第一优先执行线
- 不应把 `CopyMark` 直接写成“当前主证据已失效”
- 不应把频域论文直接写成新的执行主线
- 不应把 `PIA` 的 provenance blocker 当成可以后补的文书尾项
- 不应把本次 `保持冻结` 决定写成新的研究结果或 benchmark 结论

## 关联文档

- [comprehensive-progress.md](comprehensive-progress.md)
- [reproduction-status.md](reproduction-status.md)
- [../ROADMAP.md](../ROADMAP.md)
- [../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md](../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md)
