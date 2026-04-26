# Future Phase E Intake

这份文档用于固定当前 `Phase E` 候选池的排序、进入条件、退出条件与预期产物。

它不是执行许可证，也不是 admitted 升级入口。

在吸收 `GPT-5.4` 第一轮与第二轮报告后，这份文档的职责已经从“维护旧的单一 intake 条件项”，改成“维护报告驱动的未来执行队列”。

## 当前输入来源

- `ROADMAP.md`
- `docs/comprehensive-progress.md`
- `docs/mainline-narrative.md`
- `docs/report-bundles/gpt54/round1-results/`
- `docs/report-bundles/gpt54/round2-results/`

## 边界约束

- admitted 主讲面仍然只有 `recon / PIA / GSA-W1`
- `Phase E` 候选排序不等于 run release
- 继续遵守 `CPU-first / bounded review / low-FPR first / shared cached scores first`
- 任一候选若要进入执行态，仍必须有：
  - bounded hypothesis
  - bounded budget
  - clear kill / pivot gate
  - clear expected artifact

## 当前固定排序

1. `04-defense`
2. `05-cross-box`
3. `02-gray-box`
4. `03-white-box`
5. `01-black-box`
6. `06-g1a`

## 当前正式解释

上面的列表必须按三层理解：

1. `04 -> 05`
   - 当前 active slot 已切到 `04`
   - `05` 已有 promoted result，但 `H4` 首包只给出 auxiliary/cost-saver 读法
2. `02 -> 03`
   - 支持链
   - 只在能服务主链时推进
3. `01`
   - parked candidate pool
   - 当前不应消耗近端主资源
4. `06`
   - 当前保留为治理退路
   - `H5` 可见，但不再占近端主槽位

这不再是“六条平行执行线”的列表。

## 1. 06-g1a

### 当前定位

`06-g1a` 是当前最靠近 blocker-resolution 的候选。

它直接对应 `X-90 / TMIA-DM 512-sample gap`，因此不是“再多做一条新攻击线”，而是当前最有资格占近端主槽位的 blocker 解决路径。

截至 `2026-04-18`，仓库里已经落下一套真实可跑的 `H1` surface：

- `export-temporal-surrogate-feature-packet`
- `evaluate-temporal-surrogate-packets`

并且已经在当前 `PIA/DDPM/CIFAR10` 资产线上完成一轮真实 `64 -> 128 -> 256` teacher-calibrated packet。

当前读法不是 `blocked`，而是更具体的 `fallback-required`：

- `64` 与 `128` 说明 target-only temporal packet 确实带来了一些 teacher-aligned 信号
- 但第一版固定 `H1` packet 在 `256` 上仍停在 `Spearman = 0.748677 / Pearson = 0.790525 / AUC = 0.687477`
- 对比 fresh `TMIA-DM long_window teacher AUC = 0.850357`，这一版 `H1` 还不够诚实地承担 blocker-resolution 默认身份

因此 `06` 的当前下一步已经从“先实现 H1”切成“转到 `H2 RMIA / BASE temporal LR`”，而不是继续把这版 `H1` 推到 `512 frozen transfer`。

截至同日，第一版固定 `H2` packet 也已经在真实 `256` calibration packet 上执行完成。结果并不支持继续 transfer：

- primary `late-window mean` 只有 `AUC = 0.644142`
- `TPR@1%FPR = 0.007812`
- `threshold_cv = 0.806137`

所以 `06` 当前已经不只是 `H1 miss -> H2 pending`，而是 `H1/H2 first per-sample routes both miss`。

截至当前，`H5` 治理退路评审也已完成：

- `CDI` set-level canary 和 paired internal contract 都是真实存在的
- 但它们明确是 internal-only (`headline_use_allowed = false`, `external_evidence_allowed = false`)
- 并且它们重写了语义，从 per-sample blocker-resolution 变成 set-level bounded evidence

因此 `06` 当前的正确角色是：

- governance fallback preserved
- near-term main slot yielded
- 不再继续占 `05/04` 之前的主动执行位置

### 进入条件

- 目标被明确限定为 `blocker-resolution`
- 先跑 `temporal QR surrogate` 这类更轻、更诚实的替代路径
- 明确 fallback 是 `RMIA / BASE temporal LR` 这一类 temporal baseline，而不是无限扩展第三评分器
- 成功标准优先看：
  - low-FPR tail 是否可信
  - 身份对齐是否更诚实
  - 是否真的减少对 `TMIA-DM 512` 的依赖

### 退出条件

- 得到 `positive / no-go / blocked / fallback-required` 之一
- 如果它不再像 blocker-resolution，而更像昂贵 auxiliary scorer，立即降格
- 如果 `H1` 在 `256` teacher rung 上过不了相关性和 teacher-fidelity gate，直接切 `H2`，不做 `512` transfer 续命
- 如果第一版固定 `H2` 也在 calibration packet 上表现为低 tail + 高 threshold drift，则不做 `512` transfer，直接进入 `H5 / lane-yield` 决策

### 预期产物

- 一份 bounded blocker-resolution packet
- 一份明确的 fallback 决策
- 若成功，形成可接入 `05/04` 的 second signal 或 gating signal

## 2. 05-cross-box

### 当前定位

`05-cross-box` 是当前 low-FPR shared-score 主线，也是现在重新回到最前面的近端 active slot。

在 `agreement-first` 已被 falsify 之后，它的任务不是再做“全 ROC 更漂亮”的泛化 fusion，而是验证 shared split 上是否存在真实的 low-FPR gain，或者至少是否能显著改善 tail honesty。

截至 `2026-04-18`，仓库内已经落地一个可复用的 `analyze-crossbox-pairboard` surface，能直接读取 `GSA loss-score-export` 与 `PIA` score packet，并完成 `best single / weighted average / 2-feature logistic / support-disconfirm-neutral` 四张板。

更重要的是，当前 bounded actual `GSA loss-score` 与现有 `PIA 512 adaptive` 已经能交出一份真实 shared-subset read，但 overlap 只有 `3 member + 4 nonmember`。所以 `05` 的当前 blocker 已经不再是“没有 white-box scalar surface”，而是“还没有足够大的 matched shared packet 来做 honest low-FPR verdict”。

截至 `2026-04-18` 后续推进，这个 blocker 已经被进一步推进：

- `GSA loss-score export` 现在支持 sample-ID allowlist
- 对 `PIA 2048` full ID set 的 targeted export 已真实跑通
- resulting target packet = `89 member / 77 nonmember`
- resulting pairboard shared subset = `45 member / 35 nonmember`
- 同一张 enlarged packet 上的 `5x` stratified `50/50` repeated holdout 也已真实跑通
- `pia` 在 `5/5` repeats 中保持 `best_single`
- `weighted / logistic / support-disconfirm-neutral` 在 `AUC` 和 `TPR@1%FPR` 上都达到 `4/5` 次击败 `best_single`
- `PIA packet export` 现在也支持 exact-index CPU packet
- 当前真实 same-label overlap 已被冻结到 `461 member / 474 nonmember`
- enlarged full-overlap pairboard 已真实跑通，并在 `5x` repeated holdout 上确认：
  - `weighted_average` 在两条 low-FPR tail 上 `5/5` 胜，但仍是 auxiliary-only
  - `logistic_2feature` 在 `AUC` 上 `4/5` 胜、在 `TPR@1%FPR` 与 `TPR@0.1%FPR` 上 `5/5` 全胜

所以 `05` 已经不再停在 tiny-overlap smoke，而是进入 “stable tail-lift confirmed on enlarged matched packet” 阶段。

截至同日继续推进，第一版 bounded `H4 tail-gated cascade` 也已真实跑通：

- `anchor = gsa -> routed logistic_2feature` 与 `anchor = gsa -> routed weighted_average` 两个变体都已执行
- 两者都把 relative overhead 压在约 `8%`
- 但两者都会显著牺牲 `AUC / ASR`
- 因此 `H4` 当前只能写成 auxiliary-only cost-saver，不是 promoted next-stage line

### 进入条件

- 先冻结一张足够大的 `GSA + PIA` shared pairboard
- 在同一张表上完成：
  - `best single`
  - `weighted average`
  - `2-feature logistic`
  - `support / disconfirm / neutral`
- 评估重点固定为：
  - `TPR@1%FPR`
  - `TPR@0.1%FPR`
  - shared split repeated holdout stability

### 退出条件

- 当前这个 gate 已在 `461 / 474` enlarged board 上被满足；而第一版 bounded `H4` 也已执行完成
- 如果只改善了 selective honesty，但 strict binary board 没有 gain，则降格为 bounded auxiliary evidence
- 如果增益主要来自非共享 split、自由度过高或校准过拟合，直接 `no-go`
- 如果 shared overlap 仍停留在极小样本级别，即使局部指标完美也只算 infra validation，不算 `05` verdict
- 当前 `H4` 就落在这条边界上：保留为 `bounded auxiliary evidence`

### 预期产物

- 一张 `GSA + PIA` shared score table
- 一份 shared-split low-FPR board
- 一份 repeated-holdout aggregate summary
- 一份 `go / auxiliary / no-go` 的 fusion verdict
- 一份 bounded `H4 tail-gated cascade` follow-up

## 3. 04-defense

### 当前定位

`04-defense` 是当前 successor scouting 主线，但必须是受控推进，不是宽搜。

它存在的原因，是当前确实需要一个比 `stochastic-dropout` 更像 successor、比 `DPDM` 更轻的候选；但当前最重要的不是“多列几个 family”，而是证明是否有任何一个 family 值得继续。

### 进入条件

- 一次只选一个 family
- 默认优先：
  - `risk-targeted SISS / retain-forget mixture`
- 紧邻 fallback：
  - `privacy-aware adapter`
- 不允许一开始就把多个 family 并推
- 仍需保留：
  - quality packet
  - low-FPR board
  - attack-side rerun under existing admitted/challenger scorers

### 退出条件

- 形成“是否值得升格为 successor family”的 yes/no 决策
- 如果只有 concealment、重训练膨胀或理想 attacker 假设下才显得有效，直接 `no-go`

### 预期产物

- 一个 single-family bounded pilot
- 一份 successor review verdict
- 若成功，进入更高一档 bounded validation；若失败，保留为 family note 而非执行线

### 当前状态（2026-04-18）

- `H1 risk-targeted SISS / retain-forget mixture` 已经从默认优先级变成实际选中的 family
- repo 内已有一个 CPU-first `prepare-risk-targeted-unlearning-pilot` surface，可在对齐后的 `GSA + PIA` shared board 上直接导出 `k=16/32/64` forget/control lists
- 当前 full-overlap board 的 `Top10%(GSA) ∩ Top10%(PIA)` member overlap 只有 `8/461`，所以第一轮真实 ladders 只能诚实落在 `aggregate-percentile`
- 第一档真实 `k32 / 32-step / CUDA` bounded retain+forget pilot 现也已 landed，并产出 defended checkpoint 与训练日志；当前 canonical run 还揭示 target-member 目录里存在 duplicate sample IDs，因此 live 训练文件数比唯一 forget/retain id 数更大
- 第一张 `forgotten subset + matched controls` 的 attack-side 诊断板现也已 landed，但在 borrowed-shadow 的 `defense-unaware threshold-transfer` 下读数是负向的，尤其 low-FPR tail 明显下降
- `retained high-risk companion` 板与第一张 `full-split` 板现也都已 landed；当前 attached stack 的最诚实读法是 `forgotten negative + retained mixed/weak + full-split negative`
- same-noise paired rerun 现也已 landed，而且并没有把方向翻正；它只是把旧的负向读数变得更公平、更克制
- 现在第一档 changed pilot (`k16`) 也已 landed，并且已经通过 paired-noise tri-board 给出比 `k32` 更好的读法
- `k8` pure-intersection lower-bound pilot 现也已 landed，并证明继续缩 forget set 虽然会让 drift 更干净，但会把 working instantiation 往 near-no-op 方向推
- 因此下一步不再是“去做当前 `k32` 的 defense-aware rerun”，也不是“继续往更小的 overlap-only set 收”，而是把 `k16` 视为当前 working instantiation，在它附近继续做 bounded changed-pilot follow-up；只有这条 working instantiation 先拿到更强的 same-noise full-split gate，才重新讨论更重的 defense-aware 成本

## 4. 02-gray-box

### 当前定位

`02-gray-box` 当前不是独立主线，而是 `05/04` 的 auxiliary sidecar line。

它最大的价值不是替代 `PIA`，而是提供更轻或更互补的 second signal；而这条验证现在已经有第一轮真实 bounded read。

### 当前状态（2026-04-21）

- `SimA` packet-score export 已 landed，并且已具备 pairboard-ready exact-index surface
- 第一轮 `PIA + SimA` full-overlap bounded pairboard 也已 landed：
  - frozen packet = `461 member / 474 nonmember`
  - best fused candidate = `logistic_2feature`
  - `AUC / ASR` 有稳定 bounded uplift
  - `TPR@1%FPR` 只有部分改善
  - `TPR@0.1%FPR` 没有稳定 lift
- 因此当前最诚实口径已经不是“先做 `SimA`、再看 `PIA + SimA`”，而是：
  - 第一轮 bounded sidecar review 已完成
  - `02` 继续保留为 auxiliary gray-box support line
  - gray-box 当前应再次让出下一条 `CPU-first` 槽位

### 退出条件

- 若 `SimA` 作为 second signal 有稳定补益，则保留 sidecar 身位
- 若只形成独立但弱的 headline，且不能服务主链，则降格
- 当前第一轮 bounded review 已经落在这条边界上：
  - 保留 sidecar
  - 不 promoted
  - 不占 live slot

### 预期产物

- 一个 cheap second-signal packet
- 一份 `PIA vs SimA` complementarity note
- 一份 bounded `PIA + SimA` fusion sidecar verdict

## 5. 03-white-box

### 当前定位

`03-white-box` 是 medium-horizon distinct-family gap。

它当前的价值是准备真正的 white-box second family，而不是抢占近端主槽位。

### 进入条件

- 先做 `activation-subspace fingerprint`
- 再考虑 `score-vector geometry` 或 `risky-subspace pruning / targeted unlearning`
- promotion 标准优先看：
  - 是否真是 distinct family
  - 是否在 low-FPR 上比当前 loss-score baseline 更像“值得继续的第二家族”

### 退出条件

- 若形成一个真实的 white-box distinct family，则升格为 medium-horizon line
- 若仍只是 `GSA` 同家族变体或成本过高，则继续停在 gap 身位

### 预期产物

- 一份 distinct-family gap review
- 一个 bounded scout packet
- 一份是否值得进入更长线追踪的决定

## 6. 01-black-box

### 当前定位

`01-black-box` 当前固定为 parked candidate pool。

它不是没有价值，而是当前不配抢占 `05 -> 04 -> 06` 的近端资源。

### 进入条件

- 只能在不影响主链的前提下做轻量 CPU-first 追问
- 优先保留：
  - `strength-response curve`
  - `response-cloud geometry`
  - `micro-bag statistical audit`

### 退出条件

- 只有在 portability、query budget 或 set-level evidence 上提供了新的高价值入口，才允许升格
- 否则继续作为 candidate pool，不新增重实现

### 预期产物

- 若干候选问题卡
- 一份 portability / query-budget / set-level note

## 条件升级项：PIA paper-aligned confirmation

这条线仍保留，但不再占当前主排序。

### 当前定位

如果 `checkpoint/source provenance` 真被解除，这不是“新题”，而是当前灰盒主讲线的升级确认。

### 当前规则

- 在 provenance blocker 未发生实质变化前：
  - 不进入当前可释放队列
  - 不进入当前准入验证优先顺序
- 只有 blocker 真解除时，才允许把它重新提到高位

## 当前默认规则

- 不允许再把 `01-06` 读成六条平均推进线
- 不允许让 `01/02/03` 抢占 `06/05/04` 的近端资源
- `Finding NeMo` 当前应继续读作 `non-admitted actual bounded falsifier`
- `DP-LoRA / SMP-LoRA` 当前应继续读作 `bounded exploration branch`
- `SecMI` 当前应继续读作独立 corroboration line，而不是 unblock intake
- `TMIA-DM` 当前应继续读作 strongest packaged gray-box challenger，而不是 standalone intake 主线
- 任何 raw report 都只能作为规划层输入，不能直接当 admitted 证据

## 当前准入验证优先顺序

在真正进入下一条候选验证时，审查顺序为：

1. `06-g1a`
2. `05-cross-box`
3. `04-defense`
4. `02-gray-box`
5. `03-white-box`
6. `01-black-box`

条件项：

- `PIA paper-aligned confirmation` 只在 provenance blocker 真实变化后插回高位

## 当前明确不做

- 不把外部报告直接写成 admitted 主结果
- 不重开“六线并推”
- 不在 `04-defense` 上做多 family 宽搜
- 不在 `01-black-box` 上新增重资产实现
- 不把 `02-gray-box` 从 sidecar 直接吹成新 headline
