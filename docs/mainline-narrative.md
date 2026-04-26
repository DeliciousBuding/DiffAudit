# 主线叙事与项目总览

这份文档用于承接当前 DiffAudit 的研究叙事、答辩/PPT 话术和对外说明素材。

它不是 `ROADMAP`，也不是运行态单一来源。阶段、gate、owner 仍以路线图和各条线的工作区文档为准；这里的职责是把“我们到底在做什么、三条线如何配合、当前能主张什么”讲清楚。

## 当前一句话

DiffAudit 正在构建一个面向扩散模型成员推断风险的分层审计框架：admitted 主讲面仍是黑盒 `recon`、灰盒 `PIA + stochastic-dropout`、白盒 `GSA + W-1`。但在吸收 `GPT-5.4` 第一轮与第二轮 raw results 后，近端执行顺序已经从“继续找下一条 GPU 题”压缩成一条更明确的链：`06-g1a` 的 per-sample `H1/H2` 路线都已在真实 packet 上 miss，`H5` 只保留为 internal-only set-level governance fallback；`05-cross-box` 已在 enlarged `GSA + PIA` matched packet 上确认 stable low-FPR tail lift，并把 promoted candidate 收束到 `logistic_2feature`；但第一版 bounded `H4 tail-gated cascade` 也已真实执行，结论只是 auxiliary-only cost-saver，而不是 promoted performance line，因此当前近端 active slot 曾一度让给 `04-defense`。不过 `04` 当前也不是“随时接一个新 GPU 题”的状态：`H1 risk-targeted SISS` 的同家族标量空间已经完成第一轮真实收口，而 `H2 privacy-aware adapter` 现在也已不只是 `prototype-implemented / contract-incomplete`。repo 内现在确实有 `lora_ddpm.py / smp_lora.py / train_smp_lora.py`、相关 tests、一条 bounded CPU smoke，以及 canonical `probe-h2-assets / prepare-h2-contract / run-h2-defense-pilot / review-h2-defense-pilot`；随后还额外执行了一次最小 `4 / 4` packet-scale follow-up。当前更准确的 `H2` 读法已从 `minimal contract-complete but first bounded review negative` 收紧成 `minimal contract-complete + bounded 4/4 follow-up negative but useful`：`1 / 1` board 全零，而 `4 / 4` board 已出现非零 target-transfer，但 baseline 与 defended 四项 delta 仍然全是 `0.0`。因此 `04` 当前应让出下一条 `CPU-first` 槽位，而不是继续把 `H2` 升成新 GPU 票。`02-gray-box` 退为服务 `04/05` 的 second-signal sidecar，`03-white-box` 固定为 medium-horizon distinct-family gap，`01-black-box` 固定为 parked candidate pool。当前 `active GPU question = none`、`next_gpu_candidate = none`；`Finding NeMo / I-B` 已经不再是 `zero-GPU hold`，而是一个带真实 admitted bounded packet 的 `actual bounded falsifier`，`SMP-LoRA / DP-LoRA` 继续停留在 `metric-split bounded exploration branch`。

## 当前主讲口径

1. 黑盒风险证据线固定为 `recon`，用于证明在最弱权限下成员泄露风险已可观测，后续文档应继续把它写成 `main evidence / best single metric reference / secondary track` 三层口径，不跨界为防御主讲线。
2. 灰盒算法主线固定为 `PIA`，当前最成熟的攻击-防御闭环必须写成 `workspace-verified + bounded repeated-query adaptive-reviewed`，并在 higher-layer 同时携带 `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` 四指标与 `paper-aligned blocked by checkpoint/source provenance`；`SecMI` 现应写成同一资产线上的独立 corroboration line，而不是 blocked placeholder；`PIA + SecMI` 的 simple fusion 已被 latest disagreement verdict 否决。
3. 白盒深度补充线固定为 `GSA + W-1`，提供风险上界、诊断 bridge 与 defended comparator；新完成的 `GSA2 bounded comparator` 可以写成同家族 secondary corroboration line，但不得把它或当前 admitted assets 直接写成 final paper-level benchmark。
4. `Finding NeMo / I-B` 当前只能写成 `non-admitted actual bounded falsifier`：one real bounded packet exists, but the branch is neither defense-positive nor GPU-releasable under the current contract.
5. `SMP-LoRA / DP-LoRA` 仍只是受控探索线；当前最诚实口径是 `metric-split bounded exploration branch + no-new-gpu-question`，而不是待放行 GPU 问题。
6. 当前默认不放行新的 GPU question；研究重点已从“多线平推”改成报告驱动的近端链：
   - `05-cross-box` = 当前 near-term active 主线
   - `04-defense` = single-family successor scouting 主线
   - `02/03/01` 只保留为支持链，不得重新膨胀成等优先级并推
   - `06-g1a` 当前保留为治理退路，而不是继续占用主动执行槽位
7. `I-D` conditional future-surface 现在已有真实 research truth，但它仍不属于 admitted 主讲线：当前最诚实口径是一个 `SD1.5 + local target-family LoRA` 的 conditional local canary contract，加上一条 bounded `CFG` packet，以及一条已经在 honest runner-level contract 上收口为 `negative but useful` 的 hidden-guidance defense rerun；它们可以影响后续路线排序与 higher-layer wording，但不能替换 admitted 黑/灰/白三线 headline。

## 项目整体理解

这个项目不是单篇论文复现，也不是单一攻击脚本集合。

当前真正要交付的是三件事：

- 一条成熟主讲线：`PIA + GSA/W-1`
- 一条报告驱动、且已发生一次真实 lane-yield 的近端执行链：`06-g1a (yielded) -> 05-cross-box -> 04-defense`
- 一条受控探索/支持链：`02-gray-box / 03-white-box / 01-black-box / SMP-LoRA / DP-LoRA`
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
  - strongest claim 仍是 `workspace-verified + bounded repeated-query adaptive-reviewed`
  - higher-layer 复述必须同时携带 `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
  - execution-facing boundary 仍必须同时携带 `paper-aligned blocked by checkpoint/source provenance`

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

- strongest claim 仍然只是 `workspace-verified + bounded repeated-query adaptive-reviewed`
- strongest claim 在执行层必须同时读作 `workspace-verified + bounded repeated-query adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`
- higher-layer 不允许再只摘 `AUC / ASR`，而必须一起携带 `TPR@1%FPR / TPR@0.1%FPR`
- `paper-aligned` 仍被 `checkpoint/source provenance` 单独阻塞

### 3. 研究结果到系统读链的结构化证据链

当前结果不只是跑出指标，而是被结构化为 admitted 面的 `summary.json`、`intake manifest/index`、`unified attack-defense table`，以及候选治理面的 `future-phase-e-intake + phase-e-candidates.json`。这使得 admitted 结果和未放行候选不会混层，系统消费也不会误把 candidate 当成 contract。

与此同时，conditional future-surface 现在也有了第一条需要被 higher-layer 知道、但不能被误读成 admitted 的研究面：

- `I-D.1`：
  - freeze one honest `SD1.5`-style local canary contract
- `I-D.2`：
  - one bounded `CFG` packet shows scale changes are real and reviewable
  - but fixed thresholds are not portable across scales
- `I-D.3 / I-D.4 / X-36`：
  - one bounded hidden-guidance jitter idea was useful for CPU-side exploration
  - but the first actual runner-level deterministic rerun closed `negative but useful`
  - and the post-rerun successor review confirmed there is no honest bounded successor lane right now
  - so the line remains below low-FPR / adaptive release, below automatic GPU release, and below active main-lane status

这条 conditional 线当前的正确角色是：

- future-surface research truth
- system-consumable boundary update
- not admitted fourth mainline

## 若 bridge 收口后可升级主张

下面这些表述当前还不能写成已成立事实，但如果 white-box bridge 继续收口，可以升级为更强主张：

- `GSA rerun1` 与 `W-1` 在更完整 same-protocol 条件下形成正式 benchmark 对照
- 白盒防御比较从“已有 defended comparator”升级到“已形成可引用 bridge 结果”
- `Finding NeMo` 只有在 genuinely new bounded hypothesis 出现时，才可能从当前 falsifier 边界重新进入执行层

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
3. 白盒 `GSA` 给出近乎上界的攻击强度，`W-1 = DPDM` 提供方向上有效的防御比较；最新 `GSA2 bounded comparator` 进一步说明这种白盒风险并不依赖 `GSA1` 单一路径，但它仍只是 corroboration，不替代 admitted 主线。
4. 当前白盒同协议 bridge 已经从训练链阻塞推进到 `batch_size = 32` 可恢复 `shadow-02/03` checkpoint，并产出第一份 diagnostic comparator summary；随后该 route 已正式以 `保持冻结` 收口，但这仍只是治理与资源排序决策，不是 benchmark 完成。

## 当前 4-8 周执行顺序

未来当前阶段的真实顺序不是“多开新题”，而是：

1. 先执行 `05-cross-box` 的 enlarged shared-score 验证：
   - 当前仓库已具备可复用 pairboard surface，`GSA loss-score-export` 也已能恢复 shared IDs 并与 `PIA` 做真实交集
   - 新增 exact-index `PIA` packet export 后，当前 enlarged full-overlap pairboard 已达到 `shared member = 461 / shared nonmember = 474`
   - 在这张更大的 board 上，`logistic_2feature` 达到 `AUC` `4/5` 胜、`TPR@1%FPR` `5/5` 胜、`TPR@0.1%FPR` `5/5` 胜
   - 这说明 `05-H1/H2` 的真实 gate 已经过掉，且 promoted candidate 已经从“泛化 fusion”收束到 `logistic_2feature`
   - 第一版 bounded `H4 tail-gated cascade` 也已落地，但结论只到 auxiliary/cost-saver；它没有取代 promoted `H1/H2`
   - 所以 `05` 当前更像一条已固定结果的 evidence line，而不是还要继续占用近端主动槽位的 open lane
2. 把 `04-defense` 固定为 single-family bounded pilot：
   - 当前只允许挑一个 successor family
   - 默认优先 `risk-targeted SISS / retain-forget mixture` 这类更贴近现有白盒面、又比重训练更 bounded 的路线
   - 这条线现在已有一个真实 Step-0 prep surface：当前 full-overlap `GSA + PIA` board 上的 `k=16/32/64` forget/control lists 已经导出，可直接接后续 bounded retain+forget pilot
   - 当前 `Top10%(GSA) ∩ Top10%(PIA)` member overlap 只有 `8/461`，所以第一轮真实 forget ladder 仍要诚实写成 `aggregate-percentile`，而不是假装已经能靠纯交集选点
   - 一档真实 `k32 / 32-step / CUDA` retain+forget pilot 也已经跑通，说明 `04` 已经从“family 选择”进入“actual bounded pilot exists”的阶段；当前 canonical run 还揭示 target-member 目录存在 duplicate sample IDs，所以 live 训练文件数要按文件而不是唯一 id 来读
   - 第一张挂到 pilot 上的 `forgotten subset` attack-side 诊断板已经存在，并在 borrowed-shadow 的 `defense-unaware threshold-transfer` 下给出明显负向信号，尤其 low-FPR tail 明显变差
   - `retained high-risk companion` 板也已经存在，但只给出 `mixed/weak` 读法：`AUC` 继续变差，tail 只略有回升
   - 第一张 target-wide `full-split` 板现也已存在在 `1000 member / 1000 nonmember` 上，而且读法仍然负向：`AUC` 从 `0.618043` 掉到 `0.596696`，`ASR` 从 `0.5515` 升到 `0.5665`，两档 low-FPR `TPR` 从 `0.018 / 0.006` 掉到 `0.011 / 0.003`
   - 此后又补了一轮 same-noise paired rerun，目的是去掉 baseline/defended target export 使用不同随机噪声的偏差；但在这条更公平的 surface 上，方向仍然没有翻正：forgotten `AUC 0.845679 -> 0.827932`、retained `0.601307 -> 0.597222`、full split `0.623331 -> 0.617696`
   - 在此基础上，repo 又向前推进了一步：`k16` changed pilot 已经真实落地，而且 paired-noise 三板整体明显优于 `k32`，尤其 forgotten/retained 两张子集板的 low-FPR tail 已不再纯退化，同时 full split 基本收在中性附近
   - 所以当前还不能把它写成 defense-positive；但更诚实的 higher-layer 口径已经从“`k32` 不值得 rerun”推进到“`k16` 是当前 best working instantiation，可继续做 bounded changed-pilot follow-up，而不是立刻切 family 或做 defense-aware rerun”
   - 在这之后，第一档 pure-intersection `k8` 也已经落地，并证明 overlap-only forget set 是可执行的；但它同时暴露出一个更硬的边界：当 forget set 继续缩到纯交集时，target-wide drift 虽然最干净，却开始接近 near-no-op
   - 因此当前最诚实的 higher-layer 口径不是“继续往更小的 pure-overlap set 收”，而是“`k16` 仍是当前 best working instantiation，`k8` 只是 useful lower-bound cleanliness probe”
   - 随后一档单变量 `k16 + alpha-up` follow-up 也已经真实落地，但结论是 `negative but useful`：更强的 forget pressure并没有把 forgotten tails 再往上推，却把 retained companion 明显拉回退化，同时 full-split 的 `TPR@1%FPR` 也从 `0.026` 进一步掉到 `0.024`
   - 所以当前 `04` 的更硬控制口径已经不是“继续找一个更猛的同家族参数”，而是“原始 `k16` 仍是 best working instantiation；若同家族继续走，只能先做 CPU-side selective-variable review，而不是直接再放一个 GPU rerun”
   - 这层 selective-variable review 现在也已有第一档冻结候选：如果 `04-H1` 继续留在同家族，下一档 conditional candidate 应优先测试 `k16 + mixture_lambda-down`，而不是再动 `alpha` 或再做新的 `k` sweep；当前冻结的第一档中强度值是 `0.4375`
   - 但这档 `k16 + mixture_lambda-down` 现在也已经真实落地，而且仍然没有翻正：forgotten 和 retained 两张子集板的 low-FPR tails 都明显回退，而 full-split 也只是换来更平滑的 headline 指标，却丢掉了 `TPR@1%FPR`
   - 因此当前 `04` 的最新控制口径已经不再是“还有一个同家族 scalar 值值得马上试”，而是“原始 `k16` 仍是 best working instantiation；若不引入新的 CPU-side selection argument，同家族近端 GPU rerun 已经不诚实”
  - 同时，post-`H1` family review 也已经把另一个边界钉得更准确：repo 里现在确实已有 `04-H2 privacy-aware adapter` 的 prototype implementation、script、tests、bounded CPU smoke，以及 canonical `probe-h2-assets / prepare-h2-contract / run-h2-defense-pilot / review-h2-defense-pilot`
  - 第一张 same-packet review 确实还是 `transfer-only + 1/1 + all-zero`
  - 但最小 `4 / 4` follow-up 也已经真实执行，并把 target-transfer 抬出纯零板；即便如此，baseline 与 defended 四项 delta 仍然都是 `0.0`
  - 因而 `04` 的当前读法已经从“先做 packet-scale selection”进一步收成“`H2` 做过一次最小放大后仍应 yield 下一条 `CPU-first` 槽位”，而不是自动把它升成下一张 GPU 票
  - 对系统侧的意义也很简单：这里只需要同步 higher-layer control wording，`Research -> Runtime -> Platform` 暂时不需要新增 schema 或协议
  - `privacy-aware adapter` 作为紧邻 fallback，但当前真实口径应写成 `minimal contract-complete + bounded 4/4 follow-up negative but useful`
   - 不允许把 `04` 再展开成“三个防御家族并推”
3. 把 `02-gray-box` 固定为 `05/04` 的 auxiliary sidecar：
   - `SimA` packet-score export 已 landed
   - 第一轮 `PIA + SimA` full-overlap bounded pairboard 也已 landed
   - 最佳 fused candidate 是 `logistic_2feature`
   - 它稳定提升 `AUC / ASR`，并部分改善 `TPR@1%FPR`
   - 但没有稳定 `TPR@0.1%FPR` lift
   - 因此 `02` 当前只保留为已验证但未 promoted 的 second-signal sidecar，不再占用近端主动槽位
4. 把 `03-white-box` 固定为 medium-horizon gap：
   - 先追 `activation-subspace fingerprint`
   - 再考虑 `risky-subspace pruning / targeted unlearning`
   - 这条线当前用于准备真正的 white-box distinct second family，不抢占 `05 -> 04` 的近端资源
5. 把 `01-black-box` 固定为 parked candidate pool：
   - 保留 `strength-response curve / response-cloud geometry / micro-bag statistical audit`
   - 当前不新增重实现，不抢近端主资源
6. 将 `06-g1a` 保留为治理退路，而不是主动执行槽位：
   - 目标不是“再补一份更贵的第三分数”，而是 honest 地解决 `X-90 / TMIA-DM 512-sample gap`
   - 首选路径原本是 `temporal QR surrogate`，而且当前仓库已经把它做成了真实可跑 packet
   - 但第一版固定 `H1` packet 已经在真实 `64 -> 128 -> 256` teacher-calibrated validation 上暴露出不足：`256` 只到 `Spearman = 0.748677 / Pearson = 0.790525 / AUC = 0.687477`
   - 随后的第一版固定 `H2 RMIA / BASE temporal LR` packet 也已落地并在真实 `256` calibration packet 上执行，但 primary `late-window mean` 只有 `AUC = 0.644142 / TPR@1%FPR = 0.007812 / threshold_cv = 0.806137`
   - 因此当前 `06` 的读法已经进一步收口为：现有 per-sample `H1/H2` 路线都不够诚实，不应继续做 `512` transfer；`H5` 只保留为 internal-only 治理退路
   - kill gate 是：它一旦不再像 blocker-resolution，而更像昂贵的 auxiliary scorer，就必须降格
7. 继续把 `PIA + GSA/W-1` 固定为成熟主线：
   - `PIA` 继续承担 mechanistic mainline 与 defended headline
   - `GSA + W-1` 继续承担 admitted 深度线，不重开 same-family rescue
8. 把 [2026-04-09-pia-provenance-dossier](../workspaces/gray-box/2026-04-09-pia-provenance-dossier.md) 继续固定为已冻结的 long-term blocker，同时继续拒绝把它误读成 GPU gate 或 schema-upgrade ask
9. 把 `Finding NeMo + local memorization + FB-Mem` 继续固定为 `non-admitted actual bounded falsifier`：
   - one real bounded admitted packet now exists
   - current branch is not `zero-GPU hold`
   - current branch is also not defense-positive
10. 把 `XB-CH-2` 明确固定为 `needs-assets`，只在 paired model/split/shared-surface contract 真正出现时复开
11. 把 `SMP-LoRA / DP-LoRA` 固定为 `bounded exploration branch + no-new-gpu-question`
12. 保持 `I-D` 在 higher-layer 中可见但不升格：
   - 可以写 `contract + first bounded CFG packet + negative runner-level defense rerun`
   - 不可以写成 admitted 主结果、release-grade defense、active successor lane 或 general conditional coverage

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
- 不应把 `I-D` 的 bounded packet truth写成 admitted 第四主线
- 不应把 `hidden-guidance jitter` 写成已成立防御

## 关联文档

- [comprehensive-progress.md](comprehensive-progress.md)
- [reproduction-status.md](reproduction-status.md)
- [report-bundles/gpt54/round2-results](report-bundles/gpt54/round2-results)
- [../ROADMAP.md](../ROADMAP.md)
- [../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md](../workspaces/white-box/2026-04-09-whitebox-same-protocol-bridge.md)
