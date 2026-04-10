# DiffAudit Research Roadmap

## 目标

这份路线图只面向 `Project` 研究仓库。

统一目标：

- 让黑盒、灰盒、白盒三条攻击线进入同一套研究与证据规划
- 让攻击、资产、manifest、summary、防御结果都能被统一记录
- 在当前阶段形成“至少一条可辩护攻击主线 + 至少一条可比较防御原型 + 一张 admitted 统一总表”

## 当前研究快照

截至 `2026-04-09`：

- `Current Mainline`: `PIA + provisional G-1(all_steps)`
- `Current Risk Evidence`: `recon DDIM public-100 step30`
- `Current Depth Line`: `GSA + W-1 strong-v3 full-scale`
- `Active GPU Question`: `none (bridge closed-frozen; next queue not yet released)`
- `Blocked Candidates`: `SecMI`, `variation real API`, `W-2`, `G-2`
- `Research-Ready Candidates`: `TMIA-DM`, `Finding NeMo`, `DP-LoRA`

当前研究判断：

- `PIA` 仍是当前最成熟、最适合打成“攻击 + 防御”主讲闭环的一条线
- `PIA` 当前 strongest claim 仍是 `workspace-verified + adaptive-reviewed`
- `PIA` 当前唯一 `paper-aligned` blocker 是 `checkpoint/source provenance`
- `recon` 仍是黑盒风险证据主线，而不是当前主讲闭环
- `GSA epoch300 rerun1` 已是当前白盒攻击主结果，并已写回 admitted 总表、intake 与系统读链
- `batch_size = 32` 已让 `shadow-02 / shadow-03` checkpoint 恢复可运行
- 当前 same-protocol bridge 已完成 `保持冻结` 收口
- 当前 `batch32 diagnostic comparator` 仍停留在 `runtime-smoke / diagnostic` 级别，未改 admitted 合同
- `PIA provenance dossier` 当前已 closed 为 `remain long-term blocker`

## 当前阶段与 Gate

默认研究顺序：

1. `Phase A: Freeze Current Admitted Mainlines`
2. `Phase B: Harden Mainline Read Chain And Blocker Narratives`
3. `Phase C: Close Active GPU Strengthening Run`
4. `Phase D: Same-Protocol Benchmark Bridge`
5. `Phase E: Open Next Research Question`

当前主阶段：

- `Phase D: Same-Protocol Benchmark Bridge`

当前尾项：

- `Phase B: Harden Mainline Read Chain And Blocker Narratives`

当前 `2026-04-09` 到 `2026-06-04` 的执行顺序固定为：

1. `PIA provenance` dossier 与书面裁决
2. `recon` 主证据冻结，同时吸收 `CopyMark` 现实边界与频域解释层
3. `Phase E` 候选池 intake 与排序
4. 下一条 GPU 问题的准入验证

当前阶段判定补充：

- bridge 已完成 `保持冻结` 收口
- 当前没有新的 active 主 GPU 问题
- 当前最值得推进的唯一目标切换为：
  - `Finding NeMo + local memorization + FB-Mem` 的 `activation-export adapter` 书面收口

进入 `Phase E` 的 gate：

1. `PIA` 的 strongest claim、defended mainline、paper-alignment blocker 都已固定
2. `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 same-protocol bridge 合同已经固定
3. bridge 启动入口已 portable，且不依赖本地 scheduler
4. bridge 已产出第一份 decisive artifact，并已形成“继续扩大、保持冻结、或收口失败模式”的书面决策

补充 gate 解释：

- 当前 `batch32 diagnostic comparator` 仍是 `runtime-smoke / diagnostic`
- `Phase D` 已完成 frozen-closed 收口
- `Phase E` 当前固定排序为：
  1. `PIA paper-aligned confirmation`
  2. `Finding NeMo + local memorization + FB-Mem`
  3. `DP-LoRA`
  4. `SecMI unblock`
  5. `TMIA-DM intake`
- 执行层当前默认放行顺序为：
  1. `Finding NeMo + local memorization + FB-Mem`
  2. `DP-LoRA`
  3. `SecMI unblock`
  4. `TMIA-DM intake`

## GPU 研究规则

- 当前允许主 GPU 任务
- 但研究侧同一时段只允许一个主线 GPU 问题
- 当前 active GPU 问题：
  - `none`

本地 scheduler 边界：

- `LocalOps/paper-resource-scheduler` 是本地治理工具，不是研究仓合同
- 外部协作者必须可以在没有 scheduler 的前提下：
  - 建环境
  - 跑 `probe / dry-run / smoke / mainline`
  - 在自己的 CPU/GPU 上复现
  - 提交 PR
- 因此研究仓文档、命令、校验链都不得把 scheduler 设成硬依赖

当前问题收口后的可用 GPU 问题只允许从下面三类中选一条：

1. `PIA paper-aligned confirmation`
2. `SecMI unblock`
3. `TMIA-DM or Finding NeMo intake`

补充边界：

- `Finding NeMo + local memorization + FB-Mem` 当前只获得了 `eligibility-gated for one minimal validation-smoke only` 的定义
- 当前 `paper-faithful NeMo on admitted white-box assets = no-go`
- 当前 `portable observability smoke` 已进入 `read-only contract-probe + cpu-only activation-export adapter implemented`
- 当前仍未获 run release
- 在 intake 文档未补齐 `hook 点 / 资产需求 / compute budget / stop conditions / expected artifact` 前，不得申请 GPU

研究侧明确不做：

- 不继续重跑已冻结的 `PIA GPU128/256/512`
- 不继续重跑 admitted `GSA 1k-3shadow`
- 不在 `variation` 和 `SecMI` 资产未到位时烧 GPU

## Phase A: Freeze Current Admitted Mainlines

- 目标：
  - 固定 `recon / PIA / GSA / W-1` 的 admitted 主结果
- 当前状态：
  - `mostly complete`
- 当前 owner / 固定角色：
  - `research_leader`
- 进入条件：
  - 三条攻击线都已形成至少一份可辩护结果
- 完成标准：
  - admitted 主结果在总表、manifest、研究文档里不互相冲突
- 本阶段不做：
  - 不扩新论文
  - 不把 `blocked / smoke / secondary` 写成主结果

## Phase B: Harden Mainline Read Chain And Blocker Narratives

- 目标：
  - 把当前主讲线写成“可讲、可读、可阻塞解释”一致的正式主线
- 当前状态：
  - `closure completed`
- 当前 owner / 固定角色：
  - `research_leader`
- 进入条件：
  - `Phase A` 基本完成
- 完成标准：
  - `PIA` strongest claim 固定为 `workspace-verified + adaptive-reviewed`
  - `all_steps` 固定为 defended mainline
  - `late_steps_only` 明确为消融，不替代主线
  - `PIA` 的 `paper-aligned` blocker 固定为 `checkpoint/source provenance`
  - `GSA` 的 live intake/canonical summary 与 admitted `1k-3shadow` 主结果一致
- 本阶段不做：
  - 不因为 GPU 已放开就重跑 `PIA`
  - 不把 blocker narrative 混成 benchmark 已完成

## Phase C: Close Active GPU Strengthening Run

- 目标：
  - 收口当前已经启动的 `GSA epoch300 rerun1`，并完成 admitted 决策
- 当前状态：
  - `closure completed`
- 当前 owner / 固定角色：
  - `research_leader`
- 进入条件：
  - 已存在 active GPU 长任务
- 完成标准：
  - 形成新的 rerun runtime `summary.json`
  - 并完成当前 admitted `GSA` 主结果的决策与回写
  - 当前不再以 rerun promotion 作为 active GPU 问题
- 本阶段不做：
  - 不并行再开第二条白盒长任务
  - 不在 admission review 未完成时改 admitted 主结果

## Phase D: Same-Protocol Benchmark Bridge

- 目标：
  - 让主线从“能讲”推进到更强的同协议对比面
- 当前状态：
  - `closed-frozen`
- 当前 owner / 固定角色：
  - `research_leader`
- 进入条件：
  - `GSA rerun1` 已稳定成为 admitted 白盒攻击主结果
- 完成标准：
  - 白盒 `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 same-protocol 路线明确
  - bridge 启动入口已 portable
  - batch32 训练链已证明可以恢复 `shadow-02 / shadow-03` checkpoint
  - bridge 已产出第一份 diagnostic comparator summary，并明确它是否继续扩大或保持冻结
  - 灰盒 `PIA` 是否值得做 `paper-aligned` 确认有清晰 gate
- 本阶段不做：
  - 不把 same-protocol bridge 写成 benchmark 已完成
  - 不把新研究问题提前塞进这阶段

## Phase E: Open Next Research Question

- 目标：
  - 在当前主线收口后开启真正新的问题
- 当前状态：
  - `intake-only`
- 当前 owner / 固定角色：
  - `research_leader`
  - `总管理 Agent`
- 进入条件：
  - `Phase D` 完成
- 完成标准：
  - 新问题有明确资产条件、评价指标、退出条件
  - 同时只允许一条新主 GPU 问题
- 本阶段不做：
  - 不为了“GPU 空着”就机械复跑旧结论
  - 不同时开启两条新主线

## 研究执行纪律

- admitted 结果优先写 machine-readable `manifest / summary / table`
- `PIA` 是当前算法主讲线，不是灰盒补充线
- `recon` 是当前风险证据线，不是当前主讲防御闭环
- 白盒 `GSA + W-1` 当前价值在深度与上界，不是申报阶段唯一主讲成果
- 只要 `LocalOps` scheduler 不存在，研究仓也必须能被别人用来跑实验、提 PR

## 需要同步的研究文档

每次阶段切换或运行态变化，至少同步：

- `D:\Code\DiffAudit\Project\ROADMAP.md`
- `D:\Code\DiffAudit\Project\docs\reproduction-status.md`
- `D:\Code\DiffAudit\Project\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Project\docs\local-api.md`
