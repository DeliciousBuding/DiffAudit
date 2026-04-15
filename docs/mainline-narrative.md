# 主线叙事与项目总览

这份文档用于承接当前 DiffAudit 的研究叙事、答辩/PPT 话术和对外说明素材。

它不是 `ROADMAP`，也不是运行态单一来源。阶段、gate、owner 仍以路线图和各条线的工作区文档为准；这里的职责是把“我们到底在做什么、三条线如何配合、当前能主张什么”讲清楚。

## 当前一句话

DiffAudit 正在构建一个面向扩散模型成员推断风险的分层审计框架：黑盒 `recon` 证明风险真实存在，灰盒 `PIA` 提供当前最成熟的攻击-防御主讲闭环，白盒 `GSA + W-1` 用于量化风险上界；而在 `2026-04-13` 的最新控制后，算法主线正式固定为双线并列：`PIA + GSA/W-1` 继续承担成熟主线，`SMP-LoRA` 则被降回探索主线，其最近的 `T06 optimizer/lr frontier` 已经以 `closed-mixed-no-go` 收口。这是因为 `SMP-LoRA O02/O03/O04` 的三条稳定化路线都已经收口失败：`no-TF32` 复验落成 `0.3957 / 0.3838 / 0.5306` 的混合结果，`seed7` 回退到 `AUC=0.5188`，新的 `epoch40` 更回退到 `AUC=0.6349`。这说明修复默认 seed 合同后，当前仍没有找到能够把 `batch14 throughput` 升成默认模板的稳定化答案。`batch14 legacy=0.6485` 继续证明收益依赖 `throughput_mode`。当前 active GPU question 保持 `none`，当前不放行新的 GPU question；若未来继续，也只能先完成 `baseline vs SMP-LoRA vs W-1 comparator` packet，再决定是否申请资源。

## 当前主讲口径

1. 黑盒风险证据线固定为 `recon`，用于证明在最弱权限下成员泄露风险已可观测，后续文档应继续把它写成 `main evidence / best single metric reference / secondary track` 三层口径，不跨界为防御主讲线。
2. 灰盒算法主线固定为 `PIA`，当前最成熟的攻击-防御闭环必须写成 `workspace-verified + adaptive-reviewed`，同时显式标记 `paper-aligned blocked by checkpoint/source provenance`；`SecMI` 现应写成同一资产线上的独立 corroboration line，而不是 blocked placeholder；`PIA + SecMI` 的 simple fusion 已被 latest disagreement verdict 否决。
3. 白盒深度补充线固定为 `GSA + W-1`，提供风险上界、诊断 bridge 与 defended comparator，不得把当前 admitted assets 直接写成 final paper-level benchmark。
4. `SMP-LoRA` 仍只是受控探索线，当前已经以 `T06 optimizer/lr frontier = closed-mixed-no-go` 收口，不应写成当前的 admitted 或待放行 GPU question；若未来还要推进，必须先把 `baseline vs SMP-LoRA vs W-1` 之类的 comparator packet 写出来，再决定是否申请资源。
5. 当前默认不放行新的 GPU question，直到出现新的 decisive admission 结果；研究重点应放在把上述 admitted 口径（including recon / PIA / GSA+W-1 boundaries, Developer metadata, Leader summary) 固定住，而不是启动新的 run。

## 项目整体理解

这个项目不是单篇论文复现，也不是单一攻击脚本集合。

当前真正要交付的是三件事：

- 一条成熟主讲线：`PIA + GSA/W-1`
- 一条受控探索线：`SMP-LoRA -> comparator or no-go`
- 一套结构化证据链：admitted 面是 `summary.json -> intake manifest/index -> unified attack-defense table`，候选治理面是 `future-phase-e-intake -> phase-e-candidates.json`
- 一条能被系统消费的研究结果读链：`Research -> Local-API -> Platform`

因此，研究与工程是耦合的：实验结果不能只存在于临时日志里，而要能够进入 admitted 结果、被 `Local-API` 查询、被平台解释，并保持跨文档口径一致。

## 三条研究线如何配合

### 黑盒：风险证据线

- 主方法：`recon`
- 当前角色：证明在最弱权限假设下也能观测到成员泄露风险
- 当前价值：作为申报和答辩中的“风险确实存在”证据，而不是攻防闭环主讲线
- 当前固定包：
  - `main evidence = recon DDIM public-100 step30`
  - `best single metric reference = recon DDIM public-50 step10`
  - `secondary track = variation / Towards (formal local secondary track + blocked real-API assets)`
  - `CopyMark = boundary layer`
  - 频域论文 = `explanation layer`
- 当前黑盒边界：
  - 结论只成立于 `fine-tuned / controlled / public-subset / proxy-shadow-member` 语义下
  - 它证明成员信号在受控协议下可观测
  - 不等于真实预训练模型版权取证已成立

### 灰盒：主讲闭环线

- 主方法：`PIA`
- 当前角色：最成熟、最适合讲“攻击信号 + 防御原型 + 系统消费”闭环
- 当前价值：项目的算法主讲线
- 当前关键口径：
  - 攻击信号是 `epsilon-trajectory consistency`
  - 防御原型是 `stochastic-dropout`
  - strongest claim 仍是 `workspace-verified + adaptive-reviewed`，并且必须同时携带 `paper-aligned blocked by checkpoint/source provenance`

### 白盒：深度与上界线

- 主方法：`GSA`
- 防御候选：`W-1 = DPDM`
- 当前角色：提供最坏情况下的风险上界，以及更强的机制解释空间
- 当前价值：不是申报阶段唯一主讲成果，而是“我们已经打通更深权限层攻击/防御分析”的技术厚度

## 当前可主张

### 1. 分层威胁模型审计框架

项目已经稳定形成 `black-box / gray-box / white-box` 三条权限线，并通过统一指标与 admitted 结果表来组织结果。这使得审计不是“单论文单攻击”的局部复现，而是“按权限层级逐步增强”的系统性比较。

### 2. `PIA` 的可解释灰盒主讲闭环

当前灰盒最强线已经不是 smoke，而是真实资产 mainline。`PIA` 的攻击信号可以被解释为推理轨迹一致性，而 `stochastic-dropout` 的当前机理解释是打散这一致性信号。这构成了当前最完整的“信号解释 + 防御方向 + 系统读链”闭环。与此同时，`SecMI` 已在同一 `CIFAR-10` 资产线上给出 full-split local corroboration，最新 `PIA vs SecMI` disagreement 分析又表明两者高度相关（`Spearman = 0.907588`），因此当前灰盒不应再把 simple fusion 讲成下一条主线。

但这条线当前仍有一个必须显式带出的边界：

- strongest claim 仍然只是 `workspace-verified + adaptive-reviewed`
- strongest claim 在执行层必须同时读作 `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`
- `paper-aligned` 仍被 `checkpoint/source provenance` 单独阻塞

### 3. 研究结果到系统读链的结构化证据链

当前结果不只是跑出指标，而是被结构化为 admitted 面的 `summary.json`、`intake manifest/index`、`unified attack-defense table`，以及候选治理面的 `future-phase-e-intake + phase-e-candidates.json`。这使得 admitted 结果和未放行候选不会混层，系统消费也不会误把 candidate 当成 contract。

## 若 bridge 收口后可升级主张

下面这些表述当前还不能写成已成立事实，但如果 white-box bridge 继续收口，可以升级为更强主张：

- `GSA rerun1` 与 `W-1` 在更完整 same-protocol 条件下形成正式 benchmark 对照
- 白盒防御比较从“已有 defended comparator”升级到“已形成可引用 bridge 结果”
- `Finding NeMo` 等白盒延展线从 research-ready 升级到执行层

## 系统三层架构为什么重要

### Research

- 是研究 truth source
- 负责产出实验结果、manifest、统一总表、研究文档

### Local-API

- 是研究结果的受控消费层
- 负责把 `Research` 的 admitted 结果暴露成稳定接口，而不是让调用方直接理解实验目录结构

### Platform

- 是结果展示和解释层
- 默认消费 admitted 结果，不复制研究逻辑

这三层解耦的意义是：研究可以继续迭代，系统接口保持稳定，展示层不需要理解实验细节就能拿到可靠口径。

## 当前最值得讲的项目故事

目前最合理的对外叙事是：

1. 扩散模型存在可观测的成员泄露风险，黑盒 `recon` 已经给出主证据。
2. 灰盒 `PIA` 是当前最成熟的主讲闭环，既能解释攻击信号，也已有防御原型；`SecMI` 则作为独立 corroboration line 证明灰盒泄露并不依赖 `PIA` 单一目标，但它和 `PIA` 的 naive fusion 当前是 `no-go`。
3. 白盒 `GSA` 给出近乎上界的攻击强度，`W-1 = DPDM` 提供方向上有效的防御比较。
4. 当前白盒同协议 bridge 已经从训练链阻塞推进到 `batch_size = 32` 可恢复 `shadow-02/03` checkpoint，并产出第一份 diagnostic comparator summary；随后该 route 已正式以 `保持冻结` 收口，但这仍只是治理与资源排序决策，不是 benchmark 完成。

## 当前 4-8 周执行顺序

未来 `2026-04-13` 起的真实顺序不是“多开新题”，而是：

1. 继续把 `PIA + GSA/W-1` 固定为成熟主线：
   - `PIA` 只做 CPU 侧 provenance、paper-alignment、quality/cost/adaptive 叙事收口
   - `GSA + W-1` 继续保持 admitted 深度线，不重开 bridge
2. `SMP-LoRA` 及其 `T06 optimizer/lr frontier` 已被 `closed-mixed-no-go` 裁定，此时它只保留为受控探索线；若未来要继续推进，也只能先完成 `baseline vs SMP-LoRA vs W-1 comparator` packet，再根据 comparator 结果决定是否重新启动优化/训练策略。当前这一 gate 已以 `Research/workspaces/intake/2026-04-14-baseline-smp-lora-w1-comparator-admission-packet.md` 固化，明确了 comparator 推演所需的假设、资产、预算与停止条件。
3. 按照当前 evidence pack 的口径，仍默认不放行新的 GPU question；未来若要动 GPU，先写 comparator packet 并经过 hypothesis/budget review，之后再决定是否申请资源。
4. 用 [2026-04-09-pia-provenance-dossier](../workspaces/gray-box/2026-04-09-pia-provenance-dossier.md) 与后续 provenance notes 把 `PIA` 长期 blocker 固定为 CPU sidecar，而不是 GPU gate
5. 用 [2026-04-10-recon-decision-package](../workspaces/black-box/2026-04-10-recon-decision-package.md) 固定黑盒五件套，并把 [recon-artifact-mainline-public-100-step30-reverify-20260410-round28](../experiments/recon-artifact-mainline-public-100-step30-reverify-20260410-round28/summary.json) 保持为一致性复核，而不是 admitted upgrade
6. 继续保持 `Finding NeMo + local memorization + FB-Mem = adapter-complete zero-GPU hold`
7. 用 [2026-04-15-pia-vs-secmi-graybox-comparison.md](../workspaces/gray-box/2026-04-15-pia-vs-secmi-graybox-comparison.md) 与 [2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md](../workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md) 固定 `SecMI = corroboration line`、`PIA + SecMI simple fusion = no-go`
8. 用 [2026-04-10-tmia-dm-intake-decomposition](../workspaces/gray-box/2026-04-10-tmia-dm-intake-decomposition.md) 固定 `TMIA-DM` 仍是协议/资产拆解候选
9. 在 provenance 条件未变化前，不放行 `PIA paper-aligned confirmation`

## 当前不应过度主张

- 不应写“白盒 same-protocol benchmark 已完成”
- 不应写“`Finding NeMo` 已进入当前执行主线”
- 不应写“`Finding NeMo` 已 execution-ready、benchmark-ready 或已获得 GPU release”
- 不应把任何尚未进入 admitted 合同的诊断结果写成系统正式主结果
- 不应再把黑盒写成当前第一优先执行线
- 不应把 `CopyMark` 直接写成“当前主证据已失效”
- 不应把频域论文直接写成新的执行主线
- 不应把本轮 `non-GPU artifact-mainline reverify` 写成新的 admitted upgrade
- 不应把 `PIA` 的 provenance blocker 当成可以后补的文书尾项
- 不应把本次 `保持冻结` 决定写成新的研究结果或 benchmark 结论

## 关联文档

- [comprehensive-progress.md](comprehensive-progress.md)
- [reproduction-status.md](reproduction-status.md)
- [../ROADMAP.md](../ROADMAP.md)
- [../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md](../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md)
