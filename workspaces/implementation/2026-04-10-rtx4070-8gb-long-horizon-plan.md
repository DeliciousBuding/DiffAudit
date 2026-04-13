# 2026-04-10 RTX4070 8GB Long-Horizon Plan

## Status Panel

- `owner`: `boss + research_leader + gpu_scheduler_leader`
- `hardware`: `RTX4070 laptop 8GB only`
- `execution_mode`: `single-mainline single-gpu + cpu sidecars`
- `current_mainline`: `PIA + provisional G-1(all_steps)`
- `current_risk_evidence`: `recon DDIM public-100 step30`
- `current_depth_line`: `GSA + W-1 strong-v3 full-scale`
- `active_gpu_question`: `none`
- `white-box_bridge`: `closed-frozen`
- `current_blocker`: `PIA checkpoint/source provenance`
- `finding_nemo_state`: `adapter-complete zero-GPU hold`
- `updated_at`: `2026-04-10 21:36 +08:00`

## 目的

这份计划用于把接下来 `2 周 / 4 周 / 8 周` 的研究推进、GPU 队列、CPU 并行面、论文吸收面和系统消费面统一写成一份可续跑的长期执行计划。

它追求：

1. `high-information-density GPU utilization`
2. `single-GPU question discipline`
3. `full evidence collection under 8GB reality`
4. `no diagnostic-to-benchmark drift`

它不授权：

1. 自动释放新的 `Phase E` GPU 题
2. 重新打开 `white-box same-protocol bridge`
3. 把 `Finding NeMo` 从 `zero-GPU hold` 偷换成 execution-ready
4. 把 `PIA` 的非 provenance 问题写成 `paper-aligned confirmation`

## 固定边界

### 当前必须保持

- `PIA` strongest claim = `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`
- `recon DDIM public-100 step30` = `black-box main evidence`
- `GSA epoch300 rerun1` = `admitted white-box attack main result`
- `W-1 strong-v3 full-scale` = `admitted defended main rung`
- `Finding NeMo` = `adapter-complete zero-GPU hold`
- `batch32 diagnostic comparator` = `runtime-smoke / diagnostic`

### 当前绝对不做

- 不并行开第二条主 GPU 问题
- 不重开 `closed-frozen` bridge
- 不机械重跑已冻结的 `PIA GPU128/256/512`
- 不机械重跑 admitted `GSA 1k-3shadow`
- 不把 `phase-e-candidates.json` 混成 promoted contract surface
- 不把 `Finding NeMo` canary 写成 validation-smoke released

## 总体执行原则

### 1. GPU 利用率从属于研究边界

目标不是让 GPU 永远 100% 占满，而是让每次占用都回答一个尚未解决的问题。

任何 GPU run 入队前都必须先回答：

1. 它改变哪一个未决判断？
2. 它失败时也会留下什么有用 artifact？
3. 它为什么必须用 GPU，而不是 CPU？
4. 它为什么不能被已有结果替代？

只要其中一项答不清，就不入 GPU 队列。

### 2. 单卡 8GB 只跑短段、可恢复、可中断任务

任何 GPU run 都必须切成：

1. `probe`
2. `preview`
3. `mainline-small`
4. `optional-expand`

恢复只能从最近一个成功阶段继续，不允许半猜测续跑。

### 3. CPU 队列负责“把下一个 GPU 问题准备成熟”

GPU 只跑已经准备好的问题。

CPU 并行面负责：

- provenance / intake / comparability / explanation / boundary
- 论文阅读与笔记
- config、manifest、summary、图表、统一表
- 失败样本与成本分析

## 时间规划

## 2 周目标

### A. 灰盒主讲线继续加固

目标：

- 把 `GPU512 admitted line + CPU32 portable pair` 收成同一套 gray-box strongest-claim 证据包
- 在 `RTX4070 8GB` 现实下找到一组 `PIA` 的 `8GB-safe supporting setting`

任务：

1. `PIA adaptive portability ladder`
2. `PIA defense-cost frontier`
3. `PIA provenance dossier` 继续只做 CPU 侧闭环补件

预期产物：

- `summary.json` x 2-6 组
- 一份 `8GB-safe gray-box supporting frontier note`
- 一组可直接复述的 `adaptive / quality / cost` 结论

### B. 黑盒主证据冻结

目标：

- 把 `recon` 收口成高层可直接复述的固定包

任务：

1. 固定 `main evidence / best single metric / secondary track`
2. 固定 `CopyMark = boundary layer`
3. 固定频域论文 = `explanation layer`
4. 只在 GPU 空档做极小 `recon` micro-sanity windows

预期产物：

- 一份 decision-grade black-box package
- 对 `recon` 的 error slice / external-validity boundary 补充说明

Round-28 update:

- 已完成 [2026-04-10-recon-decision-package](../black-box/2026-04-10-recon-decision-package.md)
- 已新增 [recon-artifact-mainline-public-100-step30-reverify-20260410-round28](../../experiments/recon-artifact-mainline-public-100-step30-reverify-20260410-round28/summary.json)
- 当前状态固定为：
  - `writing-only / non-GPU`
  - `admitted_change = none`
  - `gpu_release = none`
  - `queue_state = not-requestable`

### C. 白盒不放行，但继续补零 GPU packet

目标：

- 把 `Finding NeMo` 现有 canary 固化为可复用 observability packet

任务：

1. 整理 `summary.json + records.jsonl + tensor artifacts`
2. 固化 selector / sample binding / output schema
3. 不新增任何 GPU run

预期产物：

- 一份 `Finding NeMo observability packet`
- 一份 `release-review required before any GPU` 的固定锚点

## 4 周目标

### A. 单卡只准备一个新 GPU 候选

目标：

- 只准备一条 `Phase E` 中真正可能吃 GPU 的候选，而不是多开题

当前建议：

- `DP-LoRA comparability ladder`

理由：

1. `Finding NeMo` 仍是 `zero-GPU hold`
2. `SecMI` 仍缺真实资产
3. `TMIA-DM` 还没有最小可执行路径
4. `DP-LoRA` 最适合在 8GB 上先做 `comparability`，不是直接替代 `W-1`

预期产物：

- 一份 `DP-LoRA release-review / budget-review packet`
- 一份 `comparability yes/no` 裁决

### B. 系统消费面继续加固

目标：

- `Local-API / Platform` 能稳定消费 admitted 结果和解释字段

任务：

1. 保持 `unified attack-defense table` 为 admitted 唯一总出口
2. 固定 `PIA / GSA` contract-best 读链
3. 不把 candidate surface 混成 promoted surface

预期产物：

- 更稳定的 `catalog / evidence / report` 视图
- 更少的“靠文档口头解释”的系统层依赖

## 8 周目标

### A. 收口唯一新 GPU 问题

目标：

- 对唯一被放行的新 GPU 问题形成 `go / not-yet / no-go`

要求：

1. 不新增第二条 GPU 线
2. 不让 filler 任务积累成隐藏主线
3. 不因为 GPU 空着就机械扩规模

### B. 完成 3 条主线 + 候选面 + 系统面的统一叙事

目标：

- `recon / PIA / GSA+W-1` admitted 面稳定
- `Phase E` 候选面继续保持单线 release discipline
- 系统读链与研究口径一致

预期产物：

- 一套更完整的 admitted + explanation + boundary + candidate governance 包

## GPU 队列

## Stage 0: 空档填充规则

只有当未来 `2h` 内没有已通过 preview 的更高优先 GPU run 时，才允许投放 filler。

filler 只允许以下两类：

1. `recon micro-sanity window`
2. `PIA tiny-budget portability calibration`

它们只允许回答：

- portability
- cost frontier
- explanation delta

不允许回答：

- new mainline
- benchmark
- release

## Stage 1: 当前主线支持 GPU 队列

### G1-A `PIA adaptive portability ladder`

- `track`: `gray-box`
- `command`: `run-pia-runtime-mainline`
- `question`: 在 8GB 单卡上，`provisional G-1(all_steps)` 是否还能稳定压低 `adaptive` 指标
- `budget`: `3 rungs max; small -> medium -> max-safe-on-8GB`
- `stop_condition`:
  - 任一 rung 已稳定给出同向结果，则不必继续盲目扩档
  - 若结果波动大、方向翻转或时长超过预估 2 倍，则停止
- `expected_artifact`:
  - paired `summary.json`
  - one `frontier note`
- `admission_effect`: `none`

Round-26 update:

- execution packet:
  - `workspaces/gray-box/2026-04-10-pia-8gb-portability-ladder-execution-packet.md`
- frontier note:
  - `workspaces/gray-box/2026-04-10-pia-8gb-supporting-frontier-note.md`
- completed rungs:
  - `GPU128 adaptive pair`
  - `GPU256 adaptive pair`
- current decision:
  - `GPU128 = quickest portable pair`
  - `GPU256 = decision rung with cost warning`
  - `stop before any GPU512 rerun`

### G1-B `PIA defense-cost frontier`

- `track`: `gray-box`
- `command`: `run-pia-runtime-mainline`
- `question`: 当前 8GB 环境下最省预算、最可复述的 defended setting 是什么
- `budget`: `<= 4 runs`
- `stop_condition`:
  - 找到 1-2 个 Pareto-frontier 点即停
  - 若只重复旧结论，不继续
- `expected_artifact`:
  - `cost / adaptive / quality` 对照表
  - one `recommended 8GB-safe preset`
- `admission_effect`: `none`

Round-27 update:

- current verdict:
  - `no-go`
- queue state:
  - `not-requestable`
- rationale:
  - `GPU128/GPU256` 已经回答 portability frontier
  - `GPU256 defense` 成本告警已出现
  - 当前没有新的 bounded low-cost question 被写成 hypothesis/budget note
- reopen condition:
  - only after a new single-question hypothesis/budget/expected-artifact note exists

## Stage 2: 条件性下一题 GPU 候选

### G2-A `DP-LoRA comparability ladder`

- `track`: `white-box defended-track candidate`
- `command`: `run-dpdm-w1-target-only` / `run-dpdm-w1-shadow-comparator` / `run-dpdm-w1-multi-shadow-comparator`
- `question`: `DP-LoRA` 是否值得保留为 `W-2` comparability candidate
- `budget`: `max128 -> max256 -> max512`, 逐 rung 决策
- `stop_condition`:
  - 一旦确认协议不可比或信号过弱，立即停止
  - 只有前一 rung 有明确增量，才进下一 rung
- `expected_artifact`:
  - `comparability yes/no/unclear`
  - minimal comparable configuration
- `admission_effect`: `none`

## Stage 3: 默认 blocked 的 GPU 候选

### G3-A `Finding NeMo minimal validation-smoke`

- `state`: `blocked`
- `why_blocked`: `adapter-complete zero-GPU hold; separate release review missing`

### G3-B `SecMI unblock baseline`

- `state`: `blocked`
- `why_blocked`: `real flagfile + checkpoint root missing`

### G3-C `TMIA-DM minimal executable smoke`

- `state`: `blocked`
- `why_blocked`: `minimal executable path undefined`

## CPU 并行面

### C1 `PIA provenance closure`

- 目标：继续补 `checkpoint/source provenance`
- 产物：
  - upstream release identity note
  - split provenance mapping
  - paper protocol mapping delta

Round-29 update:

- 已完成 [2026-04-10-pia-provenance-split-protocol-delta](../gray-box/2026-04-10-pia-provenance-split-protocol-delta.md)
- 已新增 [next-run-20260410-round29-strict-redo/manifest.json](../gray-box/assets/pia/next-run-20260410-round29-strict-redo/manifest.json)
- 当前新结论固定为：
  - `split shape aligned locally`
  - `paper-faithful random-four-split / four-model tau-transfer protocol still open`
  - `strict redo currently dirty, so any future strict review must first restore repo cleanliness`

### C2 `Recon explanation / boundary package`

- 目标：保持冻结后的黑盒主证据包与外围文档一致
- 产物：
  - `recon decision package`
  - `CopyMark` 边界说明
  - 频域解释层
  - `non-GPU artifact-mainline reverify`

### C3 `Finding NeMo observability packet`

- 目标：把现有 canary 固化为可复用 packet
- 产物：
  - tensor audit
  - selector / sample binding note
  - zero-GPU hold packet

### C4 `DP-LoRA comparability dossier`

- 目标：让 `DP-LoRA` 在进 GPU 前就有清晰 comparability 审查包
- 产物：
  - protocol overlap note
  - minimal config candidate
  - no-go triggers

### C5 `SecMI asset unblock`

- 目标：把 `SecMI` 从资产阻塞拉到可判断状态
- 产物：
  - asset checklist
  - unblock memo

### C6 `TMIA-DM decomposition`

- 目标：只做协议/资产拆解
- 产物：
  - minimal executable path definition
  - intake decision

### C7 `System read-chain hardening`

- 目标：让系统稳定消费 admitted 面和解释字段
- 产物：
  - unified table / contract-best / report field sync

## 失败恢复与恢复链

### 统一恢复顺序

1. 降 batch
2. 降样本数
3. 降 worker
4. 缩短 step/window
5. 回到 CPU probe

### 统一 writeback 最小要求

每个阶段至少写出：

- `summary.json`
- resolved config
- `last_completed_stage`
- checkpoint pointer
- sample binding / seed

### 夜间纪律

- 只允许跑已通过 preview 的配置
- 未通过 preview 的配置不允许 overnight

## 哪些方向必须继续延后

1. `PIA paper-aligned confirmation`
2. `Finding NeMo` GPU smoke
3. `SecMI` 真正执行
4. `TMIA-DM` 真正执行
5. `variation real API`
6. `white-box same-protocol bridge` 重开
7. admitted `PIA GPU128/256/512` 机械复跑
8. admitted `GSA 1k-3shadow` 机械复跑

## Immediate Next Three

1. 继续补 `PIA provenance` 的 `release/source identity + split provenance + paper protocol mapping` CPU 闭环
2. 保持 `recon decision package` 冻结，并把外围主文档、状态页和控制态全部对齐；不新开黑盒 run
3. 在 `Finding NeMo` 继续保持 `zero-GPU hold` 的前提下，准备 `DP-LoRA comparability ladder` 的 release-review packet，形成下一次 GPU 准入审查包
