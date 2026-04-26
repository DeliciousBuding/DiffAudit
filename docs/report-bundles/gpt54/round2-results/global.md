According to a document from 2026-04-17, 我对这包的主判断是：不要把 `04/05/06` 当成三条平均推进线，而应改成“`06` 先解 blocker，`05` 立刻验证 low-FPR 增益，`04` 只做受控 successor scouting；`01/02/03` 以高价值追问保留，不再并推”。原因很直接：当前主线已经从“继续找下一条 GPU 题”转成 `CPU-first / bounded review / no waste`，`active_gpu_question = none`，根级 blocker 是 `X-90 / TMIA-DM 512-sample gap`；同时规划门槛要求优先 low-FPR、优先解决 blocker、优先低成本互补证据、优先 shared cached scores，而不是新一轮重实验。

## 1. Global prioritization

1. `06-g1a`
   我会放第一。它直接对应当前明确 blocker，最符合“解决当前 blocker”这条 gate；而且六方向摘要已经把它收敛成 `temporal QR surrogate / RMIA-BASE temporal LR / set-level pivot` 三个问题，不再是泛泛扩张。

2. `05-cross-box`
   我会放第二。当前已经明确 `agreement-first` 被 falsify，下一步应是更诚实的 fusion / selective route；同时六方向摘要也把问题压缩成 `GSA + PIA calibrated late fusion`、`support/disconfirm/neutral`、`tail-gated low-FPR cascade`，这类工作天然更接近 shared-score / CPU-first 验证，而不是大规模新训练。

3. `04-defense`
   我会放第三，而且是“受控继续”，不是“停”。它之所以还热，是因为当前确实需要一个“比 `dropout` 更强、比 `DPDM` 更轻”的 successor；但从包内信息看，`04` 目前更像候选 family 池，而不是已经出现单一最短实现 brief，所以推进方式应是“只选一个 family 做 bounded pilot”，而不是宽搜三条线并行。

4. `02-gray-box`
   我会放第四，作为 sidecar 候选，不开 standalone 主线。理由不是它不重要，而是它现在最大的价值，是服务 `05/06`：它仍可能提供比 `PIA` 更轻或更互补的第二信号，但不值得在这个窗口单独吃掉主资源。

5. `03-white-box`
   我会放第五，保留为长线缺口。包里已经明确白盒 distinct second family 仍未真正成立，所以它不能被当成“已经收官”；但同样地，它也不像未来 30 天最短产出线，更适合作为中期研究缺口来追问。

6. `01-black-box`
   我会放第六，并把它定义为“冻结新增实现、保留固定包与 portability 问题池”。不是因为它没价值，而是因为当前规划明确说 `01/02/03` 应以“高价值追问”而不是“全量并推”为主，而黑盒又最受 query 成本与环境约束影响。

我的分类就是：**立即继续 = `06`、`05`；受控继续 = `04`；候选 sidecar = `02`、`03`；暂停新增实现 = `01`。** 唯一最接近可交换的位置是 `04` 和 `02`；我把 `04` 放前，主要因为 prompt 明确要求优先尊重当前最接近真实执行的 `04/05/06` 三包。

## 2. 30-day plan

第一件事，不是立刻开更多实验，而是先冻结一个共同评估合同：`DDPM/CIFAR10` admitted surface、low-FPR 优先、bounded pilot 优先、adaptive attacker 必须考虑、能用 shared cached scores 的先用 shared cached scores。否则你会在未来 30 天里不断拿不可比的 headline metric 互相“证明”。

第二件事，做 `06-H1` 的最小 blocker-resolution pilot。它的目标不是“证明任何 surrogate 都行”，而是验证：有没有一条更便宜、身份对齐更好、仍能保住 low-FPR tail 可信度的替代路线。若这个 pilot 很快暴露出不可信，就尽早转 `H2 = RMIA / BASE temporal LR`，不要把一个月烧在“继续硬补 TMIA-DM 512”上。

第三件事，同月启动 `05` 的 shared-score 验证：先做 `GSA + PIA calibrated late fusion`，然后做 `support/disconfirm/neutral`。判断标准不是 “AUC 漂不漂亮”，而是 shared split 上 `TPR@1%FPR / TPR@0.1%FPR` 是否有稳定正增益，或至少是否用很小成本显著改善尾部决策诚实性。

第四件事，`04-defense` 只允许选一个 family 做 bounded pilot。我会在 `risk-targeted SISS unlearning / privacy-aware adapter fine-tuning / sensitivity-conditioned noise` 里只挑一个最接近当前仓库的方案；本月不做“三线并推”，因为那会直接违反 `no waste` 和 `bounded pilot` 的原则。

第五件事，在 30 天末给出一张统一的 go/kill 表：`06` 是否真在解 `X-90`；`05` 是否真改善 low-FPR；`04` 是否出现一个值得升格的 successor family。到这一步还不能过 gate 的，直接降为 candidate 或收口，不留“差一点就能成”的模糊叙事。

## 3. 90-day plan

**第 1 个月** 的目标不是“多做题”，而是把三条近端线的真假分出来：`06` 要么证明 blocker 有 honest 解法，要么明确进入 fallback；`05` 要么证明 shared-table 的 low-FPR gain 存在，要么退回 bounded auxiliary evidence；`04` 要么冒出一个真正可跑的 successor candidate，要么承认当前还没有。

**第 2 个月**，只放大第 1 个月已经过 gate 的东西。若 `06-H1` 过关，再去做更完整的 512-surface 整合；若 `05` 过关，再看是更需要 portability 还是 efficiency，决定走 `Recon + PIA` 还是 `tail-gated cascade`；若 `04` 的单一 family pilot 有信号，再升到下一档 bounded validation。没有过 gate 的方向，不进入“再给一次机会”的惯性循环。

**第 3 个月**，目标应从“继续试”切换成“形成一个能讲清楚的阶段性地图”：最好情况是得到“一个 blocker-resolved 的 `06`、一个 validated 的 `05`、一个有诚实希望的 `04`”；次优情况也应该是得到“哪些方向值得收口、为什么收口”的硬结论。到那时，`02/03` 再被重开，也必须是为了解决这张地图上已经暴露出的具体缺口，而不是重新发散。

## 4. Parallelization

可以并行的，主要是 CPU-first 的东西：`06` 的 blocker-resolution pilot、`05` 的 shared-score calibration / routing 设计、`04` 的 family 选择与评估合同准备、以及 `02` 那些能复用缓存输出的轻量追问。这些都符合“能 CPU-first 的先 CPU-first，能用 shared cached scores 验证的不要先开新重实验”。

必须串行的有三类。第一，`06` 的 fallback 链必须串行：先看最短 honest 路线，再决定是否 pivot。第二，`05` 的更重分支必须串行：先证实 shared-table 上的两盒增益，再决定是否值得上 portability / cascade。第三，`04` 的实现必须串行：先选一个 family，再做 pilot，不能在“还没出现第一条真希望”的时候把资源铺在三条防御线上。

## 5. Kill / pivot gates

全局 kill gate 我会直接沿用规划包：没有改善 `TPR@1%FPR` 或 `TPR@0.1%FPR`、没有形成 distinct family、没有用很低成本提供新的互补证据、也没有解决明确 blocker 的，就不该继续；同样，只改善 `AUC`、需要不成比例的 GPU / 工程改造、或只在理想 attacker 下有效的，也应该停。

对 `06`，最关键的 kill gate 是：它一旦不再像“blocker-resolution”，而更像“又一个贵的第三分数”，就应该 pivot，而不是继续包装成 blocker 在被解决。对 `05`，最关键的 kill gate 是：如果 shared split 上没有稳定 low-FPR 正增益，或者收益只来自非共享 split/过拟合堆叠，那就降格为 bounded auxiliary evidence。对 `04`，最关键的 kill gate 是：如果某 candidate 只能通过 concealment、重训练膨胀、或理想化 attacker 假设才显得有效，就该直接关掉。

## 6. Other directions worth asking

对 `01-black-box`，我只会追两个问题：
其一，`strength-response curve` 能不能把 portability 和 query budget 的 trade-off 量化成一个可比较前沿；其二，`micro-bag statistical audit` 能不能在不新增重资产的前提下，给出更便宜的 set-level corroboration。

对 `02-gray-box`，我会追：
其一，`SimA single-query score norm` 能不能成为比 `PIA` 更轻的第二信号；其二，`mid-frequency residual` 或 `tail-calibrated tri-signal fusion` 能不能直接服务 `05/06` 的 low-FPR 目标，而不是变成另一个独立方向。

对 `03-white-box`，我会追：
其一，`activation-subspace fingerprint` 到底是不是 distinct second family，而不是 `GSA` 的同家族变体；其二，`risky-subspace pruning / targeted unlearning` 有没有机会成为比当前更 bounded 的白盒干预线索。

一句话收尾：**未来 90 天最值钱的，不是多开几条 GPU 题，而是把 `06` 的 blocker honest 地解掉，把 `05` 的 low-FPR gain 做成 shared-table 结果，再决定 `04` 里哪一条真配得上成为 successor。**
