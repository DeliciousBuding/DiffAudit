# Comprehensive Progress

这份文档是 `Research` 研究仓库的综合进度入口。

它不替代 [reproduction-status.md](reproduction-status.md) 的逐线细节，也不替代 [mia-defense-research-index.md](mia-defense-research-index.md) 的文献整理；它的职责是把“当前最能讲的攻击线、最缺的防御线、最短执行路径”放到一页里。

## 当前一句话

当前仓库已从“继续找下一条 GPU 题”切到“报告驱动的长期主线收敛”。`PIA + GSA/W-1` 仍是当前成熟主线，但最新真实 packet 已经让近端优先级再次收口：`06-g1a` 的 per-sample `H1/H2` 都已在真实 `256` packet 上 miss，`H5` 只保留为 internal-only set-level governance fallback；`05-cross-box` 已在 enlarged `GSA + PIA` matched packet 上完成更强的 full-overlap repeated holdout，并确认 stable tail-lift；随后第一版 bounded `H4` 也已落地，但只给出 auxiliary/cost-saver 读法，因此 near-term active slot 现已从 `05` 继续让给 `04-defense`。但 `04` 当前也已经把 `H2 privacy-aware adapter` 的 packet-scale 问题走完了第一轮最小验证：canonical `probe-h2-assets`、`prepare-h2-contract`、`run-h2-defense-pilot`、`review-h2-defense-pilot` 已在 admitted CIFAR10 资产上 landed，冻结了 `checkpoint-9600/model.safetensors`、`1000 / 1000` `32 x 32 x 3` `RGB` 资产身份；随后最小 `4 / 4` CPU follow-up 也已落地。当前更准确的 `H2` 读法已从 `minimal contract-complete but first bounded review negative` 收紧成 `minimal contract-complete + bounded 4/4 follow-up negative but useful`：`1 / 1` transfer-only board 全零，而 `4 / 4` board 已出现非零 target-transfer（`AUC = 0.5 / ASR = 0.375 / TPR@1%FPR = 0.5 / TPR@0.1%FPR = 0.5`），说明它不只是纯退化板；但 baseline 与 defended 四项 delta 仍然都是 `0.0`，所以它依然不能写成 execution-ready successor，也不能升为 `next_gpu_candidate`。`02` 退到为 `04/05` 输送 second signal 的 sidecar 位置，而它的 enabling gap 与第一轮 fusion review 也已进一步收口：`SimA` packet-score export 已 landed，且 `PIA + SimA logistic_2feature` 在 frozen `461 / 474` packet 上给出稳定 `AUC / ASR` 增益与部分 `TPR@1%FPR` 改善，但没有稳定 `TPR@0.1%FPR` lift，因此 gray-box 现在应再次让出下一条 `CPU-first` 槽位。`03` 固定为 white-box distinct second family 的 medium-horizon gap，`01` 固定为 parked black-box candidate pool。当前 `active GPU question = none`，`04-H2` 也已经做完一次最小 packet-scale 放大并再次关停；因此当前 live lane 已切到 `X-141 non-graybox next-lane reselection after X-140 stale-entry sync`，而不是继续机械扩 `H2`。

## 进度总览

| 维度 | 当前判断 | 备注 |
| --- | --- | --- |
| 黑盒攻击 | `主证据稳定，近端后置` | `recon` 是当前最强证据线；black-box 当前更像候选池而不是近端主槽位 |
| 灰盒攻击 | `最成熟 + sidecar可扩展` | `PIA` 已进入 real-asset runtime mainline；`SimA` 已 execution-feasible but weak，且 `PIA + SimA logistic_2feature` 已给出 bounded `AUC / ASR` 增益，但最严 `TPR@0.1%FPR` 没有稳定 lift，因此当前仍只应读作 auxiliary sidecar |
| 白盒攻击 | `主结果稳定，distinct second family 仍缺` | `GSA` 已写回 admitted 主结果；`activation-subspace fingerprint` 是当前最干净的 medium-horizon 缺口 |
| 黑盒防御 | `基本未落地` | `B-1 / B-2` 仍在设计层 |
| 灰盒防御 | `已有当前 defended story，下一步应更受控` | `stochastic-dropout` 仍是当前 defended story；报告更支持 selective successor，而不是继续 blanket 变体 |
| 白盒防御 | `已有 defended comparator，下一步转 post-training family` | `DPDM` 仍是当前 defended comparator；更值得长期看的是真正的 subspace edit / unlearning successor |
| 统一评估表 | `已有第一版` | 已新增 admitted main results 的跨盒总表 |

当前阶段追加判断：

- `white-box same-protocol bridge` 已完成 `保持冻结` 收口
- 当前 active 主 GPU 问题已回到 `none`
- 当前 `PIA provenance dossier` 已 closed 为 `remain long-term blocker`
- `PIA 8GB portability ladder` 已完成 `probe + preview + GPU128/GPU256 adaptive pair`，当前 frontier 固定为 `GPU128 = quickest portable pair`、`GPU256 = decision rung with cost warning`
- `Finding NeMo + local memorization + FB-Mem` 不再是 `decision-grade zero-GPU hold`：
  - 当前已经有一个 real bounded admitted packet
  - 当前最诚实口径是 `actual bounded falsifier`
  - same-family GPU rescue rerun 继续低于 release
- 白盒 defense breadth 的第一轮 shortlist 也已经收口为负结论：
  - 当前 repo 只有 `DPDM / W-1` 这一条可执行 defended family
  - `Finding NeMo` 仍是 observability 路线
- `I-D` 当前也已收口到更硬边界：
  - `local conditional canary contract + bounded CFG packet + negative actual runner-level defense rerun`
  - 当前没有 honest bounded successor lane
  - 只有 genuinely new bounded hypothesis 出现时才允许重开
  - `Local Mirror` 不提供第二防御家族
- [2026-04-10-recon-decision-package](../workspaces/black-box/2026-04-10-recon-decision-package.md) 已把黑盒五件套固定为 decision-grade package，本轮 [recon-artifact-mainline-public-100-step30-reverify-20260410-round28](../experiments/recon-artifact-mainline-public-100-step30-reverify-20260410-round28/summary.json) 又在 CPU 上复算到相同 headline metrics，且不改 admitted 结果
- [2026-04-10-pia-provenance-split-protocol-delta](../workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md) 已把 `split shape aligned locally / random-four-split protocol still open / strict redo currently dirty` 三点固定为新的 provenance supplement
- `X-86→X-89` 序列已完成：`X-86` 识别 G1-A blocker resolution，`X-87` 把 G1-A/X-90 deferred 为 `needs-assets`（TMIA-DM 512-sample gap），`X-88` 重定向到 root ROADMAP B-M0 Candidate A（bounded shadow-LR white-box loss-score follow-up），B-M0 Candidate A bounded GPU review 决策为 `hold-review-only`（CPU-bound 离线评估），`X-89` 在 B-M0 window 关闭后（两条候选均 CPU-bound，无 GPU release）返回 I-A CPU sidecar。当前 `active_gpu_question = none`，`next_gpu_candidate = none`，继续 I-A higher-layer boundary maintenance。
- issue #10 已关闭为 `positive hardening`：`recon` 现在有独立的 strict Stage 0 paper gate（`check-recon-stage0-paper-gate`），它会在当前公开 bundle 只证明 `proxy-shadow-member / local-semantic-chain-ready` 时明确返回 `blocked`，防止把 local-ready 误读成 paper-aligned `Attack-I`。
- GPT-5.4 round-2 报告已完成 long-horizon 收敛，而最近真实 packet 又把 active slot 进一步收口为：
  - `05-cross-box` = 当前 near-term active 主线
  - `04-defense` = 当前受控 successor scouting 主线
  - `02-gray-box` = second-signal sidecar
  - `03-white-box` = medium-horizon gap
  - `01-black-box` = parked candidate pool
  - `06-g1a` = governance fallback preserved after per-sample miss
- `05-cross-box` 现在也已有 enlarged full-overlap pairboard：
  - shared packet = `461 member / 474 nonmember`
  - `logistic_2feature` 在 `AUC` 上 `4/5` 胜，在 `TPR@1%FPR` 与 `TPR@0.1%FPR` 上 `5/5` 全胜
  - `weighted_average` 仍只适合保留为 tail-only auxiliary fusion
  - 当前最诚实口径是 `stable low-FPR tail-lift confirmed on enlarged matched packet`
  - 第一版 bounded `H4` 现已完成，但只落在 auxiliary/cost-saver 边界上
  - 因此 `05` 当前保留 promoted `H1/H2` 结果，active slot 则继续让给 `04`
- 这意味着未来 30 天不应再把 `01-06` 读成六条平行执行线，而应读成 `05 -> 04` 的 active chain + `02/03/01` 的支持链 + `06` 的治理退路

## 攻击主线

### 黑盒

- 主线：`recon`
- 次主线候选：`variation`（对应 `Towards Black-Box`）
- 当前能说的话：
  - 公开资产上的 black-box 风险已经有可引用主证据
  - `recon` strict paper-faithful `Attack-I` 入口现在有可执行 Stage 0 gate；当前正确结果仍是 `blocked / paper_aligned_semantics = false`
  - `variation` 已能在本地 CPU 上重复跑 synthetic smoke
  - `variation` 的真实 API 资产 probe 已确认 blocked，当前缺 query image root；但这条线现在已经是 `contract-ready blocked`：
    - 第一硬门槛是 `query_image_root / query images`
    - 后续复开仍必须补齐 `endpoint/proxy + query budget + frozen parameters`
  - `CLiD` 当前边界已从泛化的“local bridge”进一步收紧到 `evaluator-near local clip-only corroboration`：
    - 目标侧本地 rung 的两个输出文件在跳过首行后可解析成 `100 x 5` 数值矩阵，接近 released `cal_clid_th.py` 的输入形状
    - 但 full threshold-evaluator 仍缺 shadow train/test pair，且已执行 rung 的文件头仍暴露旧 user-cache `diff_path`
    - 这条判断现在还有 machine-readable 审计锚点：
      - `workspaces/black-box/runs/clid-threshold-compatibility-20260416-r1/summary.json`
  - 新归档 `TMIA-DM` 已证明时间相关噪声 / 梯度信号也是正式文献方向，但它当前不属于严格黑盒执行面
- 当前不能说的话：
  - 还不能把 black-box 防御讲成已有结果
  - 还不能把 `variation` 写成真实 API 闭环
  - 还不能把 `TMIA-DM` 写成黑盒新主线
- 当前用途：
  - 作为申报和答辩里的“风险存在”主证据
  - `variation` 适合作为第二黑盒候选线补充进申报叙事
  - 黑盒最终口径现在应区分 `main evidence`、`best single metric reference` 和 `secondary track`
  - 当前高层固定包应同时带出：
    - `main evidence = recon DDIM public-100 step30`
    - `best single metric reference = recon DDIM public-50 step10`
    - `secondary track = variation / Towards`
    - `CopyMark = boundary only`
    - 频域论文 = `explanation only`

### 灰盒

- 主线：`PIA`
- corroboration：`SecMI`
- 当前能说的话：
  - `PIA` 已经不是 smoke，而是真实资产 mainline
  - `PIA GPU128 / GPU256 / GPU512` 已拿到同口径 baseline + defense 对照，且 defense 指标连续三档都低于 baseline
  - `PIA GPU512` 同档 repeat 也继续维持 defense 优于 baseline
  - round-26 的 `GPU128 / GPU256 adaptive portability pair` 又在 `RTX4070 8GB` 上复现了同向下降，其中 `GPU128` 是当前 quickest portable pair，`GPU256` 则因 defense cost 升高而保留为 decision rung with cost warning
  - `pia_next_run --strict` 已通过，当前 asset line 已可写成 `workspace-verified`
  - 当前 `PIA` 攻击分数可以明确解释为 `epsilon-trajectory consistency` 信号，而不是泛化的 reconstruction score
  - `stochastic-dropout` 当前最可辩护的作用机理，是在推理时打散这一致性信号
  - 当前 gray-box 新一轮重点已从“多开 run”切到 `off / all_steps / late_steps_only + repeated-query adaptive review + structured quality/cost`
  - `SecMI` 已完成 full-split local execution，当前应写成独立 corroboration line，而不是 `blocked baseline`
  - `TMIA-DM` 已不再只是 intake 候选：
    - 现在是当前最强的 packaged gray-box challenger
    - 在 attack-side operating-point comparison 中对 `PIA` 构成真实竞争
    - 在 defended side 也保留了 `TMIA + temporal-striding` 这一条 challenger reference
  - `Noise as a Probe` 已不再只是 paper-side备选：
    - 当前 local `SD1.5 + celeba_partial_target/checkpoint-25000` 路径已经跑通
    - `8 / 8 / 8` 与 `16 / 16 / 16` 两档都已 repeat-positive
    - 当前应写成 `strengthened bounded challenger candidate`
  - `CDI` 当前已不再只是 paper-side collection idea：
    - first internal canary 已落盘
    - repaired `PIA + SecMI` paired `2048` surface 已落盘
    - `control-z-linear` 已冻结为 default internal paired scorer
    - 但它仍只应写成 internal audit-shape extension，而不是 headline scorer 或外部版权级证据
  - 新整理的 `PIA / TMIA-DM / SimA / MoFit` 文献轴已经统一到“时间 / 噪声 / 条件信号”叙事上
  - 当前最适合把防御压到这条线上做正式比较
- 当前不能说的话：
  - 还不能说灰盒防御已经验证有效
  - 还不能说 `Noise as a Probe` 已经取代 `TMIA-DM` 的 packaged challenger 位置
  - 还不能说 `Noise as a Probe` 已经可以替换 `PIA` 的 headline 地位
- 当前用途：
  - 作为当前算法主讲线
  - `TMIA-DM` 作为当前最强 packaged gray-box challenger
  - `Noise as a Probe` 作为新 latent-diffusion challenger candidate 的有界补充线
  - 作为 `Runtime-Server` contract-specific best summary 的首要 admitted 消费对象
  - 当前只允许写成 `workspace-verified + paper-alignment blocked by checkpoint/source provenance`
  - 截至 `2026-04-10`，`PIA provenance dossier` 已 closed 为 `remain long-term blocker`

### 白盒

- 主线：`GSA`
- 扩展：`Finding NeMo (executed bounded packet -> non-admitted actual bounded falsifier)`
- 当前能说的话：
  - 白盒闭环已经打通
  - 资产根、checkpoint-*、bucket 已进入规范结构
  - `DPDM` 已从环境阻塞推进到真实 CUDA checkpoint
  - 当前白盒防御的主要技术问题是评估桥接，不是训练缺失
  - `GSA` 已跑出第一版强白盒结果
  - 一条更强配置的 `GSA epoch300 rerun1` 已完成 runtime，并在同协议下显著强于旧 `20260408 1k-3shadow`
  - `DPDM` target-only comparator 当前接近随机，方向上支持防御有效
  - `DPDM` multi-shadow comparator 当前也接近随机，方向上继续支持防御有效
  - `DPDM` 在 defended target-member checkpoint 上仍接近随机，白盒防御信号更明确
  - `DPDM` 的 defended-target + defended-shadows `strong-v2` comparator 为 `AUC = 0.541199`，仍显著弱于 `GSA rerun1 = 0.998192`
  - `DPDM` 的 `strong-v2 max512` comparator 为 `AUC = 0.537201`，说明更大评估规模下趋势仍未反转
  - `DPDM` 的 `strong-v2 3-shadow max512` comparator 为 `AUC = 0.462799`，这是当前最接近 defended `1k-3shadow` 结构的本地结果
  - `DPDM` 的 `strong-v2 3-shadow full-scale` comparator 为 `AUC = 0.490813`，仍明显弱于 `GSA` 主线
  - `DPDM` 的 `strong-v3 3-shadow max128` comparator 为 `AUC = 0.537048`，说明 stronger training rung 已经能在 GPU 上稳定出第一条 defended 结果
  - `DPDM` 的 `strong-v3 3-shadow max256` comparator 为 `AUC = 0.522339`，说明这条更强训练 rung 已经推进到中规模 GPU defended 结果
  - `DPDM` 的 `strong-v3 3-shadow max512` comparator 为 `AUC = 0.5`，说明 stronger training rung 已推进到更大规模 GPU defended 结果
  - `DPDM` 的 `strong-v3 3-shadow full-scale` comparator 为 `AUC = 0.488783`，说明 stronger training rung 已完成 full-scale defended 结果
  - `DP-LoRA / SMP-LoRA` 当前已经不是 intake-only 候选：
    - 它先拿到了一张 same-asset local comparator board
    - 随后在 hardened evaluator 下又得到一张 harmonized local board
    - 但这张 harmonized board 不是 clean dominance：
      - frozen `SMP-LoRA` 仍然优于本地 `W-1`
      - 但 `baseline` 在本地 `AUC` 上优于 frozen `SMP-LoRA`
    - 因此当前最诚实口径是：
      - `successor lane alive`
      - `metric-split bounded local evidence`
      - `no-new-gpu-question`
  - 当前 same-protocol bridge 的关键训练阻塞已经从“`shadow-02` 无法落盘”收缩到“较高训练规模不稳定”；在清理 orphan `multiprocessing-fork` 后，`batch_size = 32` 已让 `shadow-02 / shadow-03` checkpoint 重新可得
  - 基于这组 batch32 checkpoint，新的 same-protocol diagnostic comparator 已经产出 [dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409](../workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409/summary.json)，指标为 `auc=0.541199 / asr=0.515625 / tpr@1%fpr=0.0 / tpr@0.1%fpr=0.0`
  - 这份 batch32 comparator 当前仍是 `runtime-smoke` 级 bridge 诊断结果，不应直接写成新的 admitted 白盒防御主结果
  - 当前 same-protocol bridge 已正式以 `保持冻结` 收口；这只是治理与资源排序决策，不是新的 benchmark 结果
  - 系统侧对白盒 `GSA` 的 live intake 现在应与 admitted `1k-3shadow` 主结果对齐，而不是继续停在早期 CPU closed-loop
  - 新的 [2026-04-10-finding-nemo-mechanism-intake](../workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md) 现在只应被读作历史 intake gate；当前 branch 已经越过 intake-only 阶段，不能再把它当作当前 `Phase E` 候选
  - 新的 [2026-04-10-finding-nemo-protocol-reconciliation](../workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md) 已明确：当前 admitted 白盒资产与 `Finding NeMo` 原始 `Stable Diffusion v1.4 / cross-attention value layers` 协议面不兼容；这条边界仍然有效，但它现在约束的是 future reconsideration，而不是“当前仍只允许 observability / zero-GPU hold”
  - 新的 [2026-04-10-finding-nemo-observability-smoke-contract](../workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md) 已把未来 smoke 的 `checkpoint_root / layer selector / sample binding / output schema / scheduler gate` 写成可审查合同；本轮又把它落实成 `read-only contract-probe`
  - `src/diffaudit/attacks/gsa_observability.py` 与 `probe-gsa-observability-contract` 已在 `Research` 内实现零 GPU 的合同解析入口，并已在真实 admitted 资产上返回 `status = ready`
  - 本轮新增 `export-gsa-observability-canary` 与 `export_gsa_observability_canary`，已在 `Research` 内实现 CPU-only 的 sample-pair activation export，并在 [finding-nemo-observability-canary-20260410-round24](../workspaces/white-box/runs/finding-nemo-observability-canary-20260410-round24/summary.json) 写出 `summary.json + records.jsonl + tensor artifacts`
  - 新的 [2026-04-10-finding-nemo-activation-export-adapter-review](../workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md) 现在只应被读作历史 adapter boundary；当前 branch 的更强 truth 已经是“一条真实 bounded admitted packet exists”
  - 新的 [2026-04-17-finding-nemo-first-truly-bounded-admitted-intervention-review-verdict](../workspaces/white-box/2026-04-17-finding-nemo-first-truly-bounded-admitted-intervention-review-verdict.md) 与 [2026-04-17-finding-nemo-post-first-actual-packet-boundary-review](../workspaces/white-box/2026-04-17-finding-nemo-post-first-actual-packet-boundary-review.md) 已把 `Finding NeMo` 当前最强诚实口径冻结为 `non-admitted actual bounded falsifier`：
    - one actual bounded admitted packet now exists
    - current branch is not `zero-GPU hold`
    - current branch is not defense-positive
  - [2026-04-10-finding-nemo-activation-only-canary-sketch](../workspaces/white-box/2026-04-10-finding-nemo-activation-only-canary-sketch.md) 继续保留为边界文档，但当前不再能写成“尚未开始 activation export”
- 当前不能说的话：
  - 还不能说白盒论文级复现成功
  - 还不能说白盒 defense 比较已经完成
  - 还不能把当前 batch32 bridge diagnostic 写成 benchmark 已完成或 admitted summary 已更新
  - 还不能把 `DPDM` target-only comparator写成同口径白盒攻击结果
  - 还不能把当前 `DPDM strong-v2 defended-target multi-shadow comparator` 写成最终白盒 defense benchmark
  - 还不能把 `Finding NeMo` 写成当前执行主线、execution-ready 或 benchmark-ready
- 当前用途：
  - 作为技术深度补充线

## 防御主线

### 当前建议

| 轨道 | 当前最合理防御路线 | 当前判断 |
| --- | --- | --- |
| 黑盒 | `B-1 / B-2` | 设计方向成立，但还没有正式实现 |

当前补充判断：

- 第一条更像真实部署层缓解的黑盒 mitigation 已经试过：
  - `served-image-sanitization = JPEG quality 70 + resize 512 -> 448 -> 512`
  - 在本地 `CLiD clip` bridge 上没有压低攻击指标
- 因此黑盒防御当前应继续写成 `not-yet-landed`，而不是“完全没试过”
| 灰盒 | `G-1` | 已进入 provisional 形态，并出现三档同口径下降信号与一次同档 repeat；新的 adaptive review 仍支持 `all_steps`，`late_steps_only` 则保留为质量优先消融 |
| 白盒 | `W-1 = DPDM` | 已拿到 strong-v2 主结果，也拿到 strong-v3 的 full-scale GPU defended 结果；当前主讲口径冻结为 `strong-v3 full-scale` |

### 当前不建议优先做

- `G-2` 知识蒸馏代理模型
- `W-2` 成员信号对抗训练

原因：

- 它们设计空间太大
- 当前仓库还没有稳定的 attack-defense 对比表
- 申报阶段更需要可运行、可对比、可讲清楚的路线

## 当前最重要的偏差

### 1. 文档路线不等于仓库真实状态

- `mia-defense-document.docx` 可以指导防御方向
- 但不能直接当作当前执行进度表

### 2. 黑盒优先不等于黑盒是当前最适合主讲的攻击-防御闭环

- 黑盒 `recon` 证据最强
- 但灰盒 `PIA` 更适合打成“攻击 + 防御”主讲闭环

### 3. 白盒价值在深度，不在当前申报阶段的稳定结果

- `GSA` 很重要
- 但当前它更适合作为“我们已经打通白盒闭环”的证明，而不是唯一主讲成果

## 当前最短执行顺序

1. 继续把 `PIA + GSA/W-1` 固定为成熟主线，并保持 admitted/system narrative 不漂移
2. 将 `06-H1 temporal QR surrogate` 固定为当前 `X-90` blocker-resolution 默认路线；只有当它过不了 teacher-calibrated gate 时，才切到 `06-H2 RMIA/BASE temporal LR`
   - 当前 repo 已经落地 `06-H1` 的真实 packet surface，并在 `64 -> 128 -> 256` 上完成 first actual teacher-calibrated validation
   - 当前固定 `H1` packet 在 `256` 上停在 `Spearman = 0.748677 / Pearson = 0.790525 / AUC = 0.687477`，相对 `TMIA-DM long_window teacher AUC = 0.850357` 仍有明显差距
   - 第一版固定 `H2` packet 现也已在真实 `256` calibration packet 上执行，primary `late-window mean` 仅到 `AUC = 0.644142 / TPR@1%FPR = 0.007812 / threshold_cv = 0.806137`
   - 因此 `06` 的当前读法已经从 `H1 miss -> H2 fallback` 进一步收敛成 `per-sample H1/H2 both miss`；下一步是 `H5` 治理退路或 lane-yield，而不是继续对这两版 packet 做 `512` transfer
3. 将 `05-H1/H2` 固定为当前 cross-box 主执行包：
   - 一张 canonical `GSA + PIA` shared score table
   - `best single / weighted average / 2-feature logistic`
   - `support / disconfirm / neutral`
   - repo 内 pairboard infra 已经落地，且 `GSA loss-score-export` 已可直接进入 shared-index intersection
   - 当前真实 `PIA 512 adaptive x GSA bounded actual loss-score` shared subset 只有 `3 member + 4 nonmember`，因此现在的 next gate 是更大的 matched shared packet，而不是提前写 fusion 结论
   - `tail-gated cascade` 只在前两者有正增益后再开
4. 将 `04` 固定为受控 successor scouting：
   - 默认只开 `H1 risk-targeted SISS`
  - `H2 privacy-aware adapter` 作为紧邻 fallback，但当前真实状态已更新为 `prototype-implemented / contract-incomplete`
   - `H1` 现在已经不只是口头优先级：repo 内已落地一个 CPU-first `prepare-risk-targeted-unlearning-pilot` surface，并在当前 full-overlap `GSA + PIA` shared board 上导出了 `k=16/32/64` forget/control lists
   - 当前 `Top10%(GSA) ∩ Top10%(PIA)` member overlap 只有 `8/461`，所以第一轮真实 ladder 还不能走纯交集，只能走 `aggregate-percentile`
   - 第一档真实 `k32 / 32-step / CUDA` retain+forget pilot 也已经执行，说明这条线不是“只能写 prep”；当前 canonical run 还额外暴露出 target-member 目录的 duplicate-id 事实，因此 live 训练文件数是 `33 forget / 967 retain`，而不是简单的 `32 / 933`
  - 第一张挂到 pilot 上的 attack-side `forgotten subset` 诊断板也已落地，但读数并不乐观：在 borrowed-shadow 的 `defense-unaware threshold-transfer` 下，`AUC` 从 `0.774691` 掉到 `0.755401`，两档 low-FPR `TPR` 都从 `0.222222` 掉到 `0.027778`
  - `retained high-risk companion` 板也已经存在，但读法仍只是 `mixed/weak`：`AUC` 从 `0.703431` 掉到 `0.670752`，两档 low-FPR `TPR` 只从 `0.083333` 回到 `0.111111`
  - 第一张 full-split board 现也已落地在 `1000 member / 1000 nonmember` 上，且 target-wide 读法仍然负向：`AUC` 从 `0.618043` 掉到 `0.596696`，`ASR` 从 `0.5515` 升到 `0.5665`，两档 low-FPR `TPR` 从 `0.018 / 0.006` 掉到 `0.011 / 0.003`
  - 因此 `04` 当前最诚实的 attached-read stack 已经不是“只看一个负向 forgotten subset”，而是 `forgotten negative + retained mixed/weak + full-split negative`
  - 现在 repo 还多了一层更公平的 target-side control：`GSA` review export 支持 same-noise paired rerun
  - 在这条更强 surface 上，三张板仍然没有翻正：
    - forgotten：`AUC 0.845679 -> 0.827932`
    - retained：`AUC 0.601307 -> 0.597222`
    - full split：`AUC 0.623331 -> 0.617696`
  - paired-noise full-split 的 score shift 也没有显示出强烈的 forgotten-targeted 效应，而更像 broad global shift
  - 所以当前不只是“还没有 defense-aware rerun”，而是“当前这个 `k32` instantiation 本身就还不值得去吃 defense-aware rerun 的成本”
  - 随后第一档 pure-intersection lower-bound pilot `k8` 也已真实落地：这是当前第一条不靠 `aggregate-percentile`、而是完全靠 `Top10%` overlap 的 forget set
  - `k8` 的读法是 `cleaner but too weak`：
    - forgotten subset 基本完全持平
    - retained companion 不再保留 `k16` 那种 tail 改善
    - full split 依然近中性，但不足以抵消它的过度收紧
  - 所以当前 `04` 的最诚实 lead 不是 `k8`，而仍然是 `k16`
  - 此后又补了一档最小 changed pilot：`k16`，即只把 forget set 从 `32` 收到 `16`，其他训练超参数不动
  - 这档 `k16` 在 paired-noise 三板上的读法明显优于 `k32`：
    - forgotten：`AUC` 仍略降，但 low-FPR tails 从 `0.315789` 升到 `0.368421`
    - retained：`AUC` 持平，tails 从 `0.235294` 升到 `0.294118`
    - full split：接近中性，`AUC 0.623331 -> 0.622141`
  - paired-noise full-split 的 shift 也从 `k32` 的 `~+0.0075` global drift 收到 `k16` 的 `~+0.0018` level
  - 因此当前 `04` 的最诚实读法不再是“只有一个弱 pilot”，而是“`k32` 已基本判弱，`k16` 是当前 best working instantiation，但仍未到 defense-positive”
  - 此后又补了一档单变量 `k16 + alpha-up` follow-up（`alpha = 0.75`，其余不动），但这档结果是 `negative but useful`：
    - forgotten subset 没有比原始 `k16` 再变好，只是把 `AUC` 从 `0.885965` 继续压到 `0.883041`
    - retained companion 明显回退，原先 `k16` 保留下来的 tails 改善直接消失，`AUC` 也从 `0.781046` 掉到 `0.774510`
    - full split 虽然 `AUC` 形式上更接近中性，但 `TPR@1%FPR` 从 `0.026` 掉到 `0.024`，`ASR` 也略变差
  - 因此当前 `04` 的控制读法再次收紧：原始 `k16` 仍是 best working instantiation，而“继续加 forget pressure”已经不再是 open lever；如果同家族继续走，只能先做 CPU-side selective-variable review，而不是立刻再放一个 GPU rerun
  - 这一层 selective-variable review 现在也已被进一步收成一个具体的 conditional candidate：若 `04-H1` 后续还要在同家族内继续，第一档 honest 候选不再是 `alpha` 或 `k`，而是 `k16 + mixture_lambda-down`；当前冻结的第一档中强度参数是 `mixture_lambda = 0.4375`
  - 但这档 `k16 + mixture_lambda-down` 现也已经真实执行，而且同样落成 `negative but useful`：
    - forgotten subset 的 low-FPR tails 从原始 `k16` 的 `0.368421` 掉到 `0.263158`
    - retained companion 的 tails 也从 `0.294118` 掉到 `0.176471`
    - full split 虽然 `AUC` 略升到 `0.624224`、`ASR` 略降到 `0.5550`，但 `TPR@1%FPR` 反而从 `0.026` 掉到 `0.021`
  - 所以当前 `04` 的更硬结论已经进一步收口为：原始 `k16` 仍是唯一保得住的 working instantiation，而同家族 scalar tuning 现在不再是 honest immediate GPU path
  - post-`H1` family review 也已经补齐：
    - repo 当前只有 `04-H1` 的 canonical `diffaudit` 级实现、CLI 与 review contract
    - `04-H2 privacy-aware adapter` 现在已有 prototype implementation / script / tests / bounded CPU smoke，且 canonical `probe-h2-assets`、`prepare-h2-contract`、`run-h2-defense-pilot`、`review-h2-defense-pilot` 都已 landed
    - 它的第一张 same-packet review 确实是 `transfer-only + 1/1 + all-zero`
    - 但最小 `4 / 4` follow-up 也已经真实执行：target-transfer 不再纯零，却仍然没有任何 defended-vs-baseline delta
    - 因此 `04` 的当前读法已经变成 `H2 should yield after one minimal packet-scale follow-up`，而不是继续自动放大或提 GPU
  - 这对 `Runtime/Platform` 的含义也已经固定：
    - 当前 sharper `04` 边界只需要 higher-layer wording 同步
    - `Research -> Runtime -> Platform` 不需要新增 schema 或协议
   - `H3` 只允许作为 cheap sidecar，不许与 `H1/H2` 并推
5. 将 `02` 固定为 sidecar enabling line，而不是独立主槽位：
   - `SimA` 当前只保留为 `execution-feasible but weak` scorer
   - 不直接重开 plain `SimA` scorer rerun
   - `SimA` packet-score export 已 landed，`member_scores / nonmember_scores / indices` 已可导出
   - 第一轮 `PIA + SimA` bounded full-overlap pairboard 已 landed：
     - `logistic_2feature` 在 `5 / 5` repeated holdout 上稳定提升 `AUC / ASR`
     - `TPR@1%FPR` 只给出部分改善
     - `TPR@0.1%FPR` 没有稳定 lift
   - 因此 `PIA + SimA` 仍保留为 auxiliary gray-box sidecar，而不是 promoted next lane
   - 最后才看 `suspicion-gated late-step perturbation`
6. 将 `03` 固定为 medium-horizon gap：
   - `activation-subspace fingerprint` 是当前最干净的白盒 second family
   - `risky-subspace pruning` 是当前最值得保留的 post-training defense 候选
   - 但它不应抢占 `05 -> 04 -> 06` 的近端槽位
7. 将 `01` 固定为 parked black-box candidate pool：
   - 先冻结 `recon` comparator rung
   - 再保留 `response-cloud geometry / strength-response / micro-bag statistical audit`
   - 当前不允许它消耗近端主资源
8. 保持 `SecMI = independent corroboration line`、`TMIA-DM = strongest packaged gray-box challenger`、`Noise as a Probe = bounded challenger candidate` 这些既有真值不漂移
9. 基于统一表继续补质量 / 成本 / boundary 列，并保持 low-FPR 与 adaptive 解释优先
10. 若 `06/05/04` 任一方向改变 exported fields / packet contract / summary logic / runner requirement，允许 `Researcher` 对接 `Platform / Runtime-Server`，但默认仍先做 note-level handoff

## 申报 / PPT 应该怎么讲

当前最合理的讲法是：

1. 扩散模型存在成员泄露风险
2. 我们已经在黑盒、灰盒、白盒三种权限下建立了攻击验证能力
3. 当前最成熟的是灰盒 `PIA`
4. 我们已经拿到一个 `provisional G-1` 灰盒防御闭环
5. 白盒 `GSA + W-1` 已经进入“强攻击结果已出、full-scale defended comparator 已有、same-protocol bridge 已产出第一份 diagnostic summary”的阶段

## 关联文档

- 逐线状态：[reproduction-status.md](reproduction-status.md)
- 主线叙事：[mainline-narrative.md](mainline-narrative.md)
- GPT-5.4 原始结果包：[report-bundles/gpt54/round2-results](report-bundles/gpt54/round2-results)（第二轮） / [report-bundles/gpt54/round1-results](report-bundles/gpt54/round1-results)（第一轮）
- 防御文档索引：[mia-defense-research-index.md](mia-defense-research-index.md)
- 防御执行清单：[mia-defense-execution-checklist.md](mia-defense-execution-checklist.md)
- 研究仓路线图：[../ROADMAP.md](../ROADMAP.md)
