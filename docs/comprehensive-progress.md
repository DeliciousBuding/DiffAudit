# Comprehensive Progress

这份文档是 `Research` 研究仓库的综合进度入口。

它不替代 [reproduction-status.md](reproduction-status.md) 的逐线细节，也不替代 [mia-defense-research-index.md](mia-defense-research-index.md) 的文献整理；它的职责是把“当前最能讲的攻击线、最缺的防御线、最短执行路径”放到一页里。

## 当前一句话

当前仓库已经具备三条攻击线的基本骨架；而 `2026-04-13` 这一轮的最新控制进一步把算法主线固定成双线并列：成熟主线是 `PIA + GSA/W-1`，探索主线是 `SMP-LoRA`。后者已经把 `O02/O03/O04` 的三条稳定化尝试全部收口：`no-TF32` 三次结果最终落成 `0.3957 / 0.3838 / 0.5306`，`O04 seed7 run1` 回退到 `AUC=0.5188`，新的 `O03 epoch40 run1` 更进一步回退到 `AUC=0.6349`。因此下一条唯一值得放行的 GPU 问题已收敛为 `T06 optimizer/lr frontier`，而当前 active GPU question 仍保持 `none`，直到 admission packet 被单独放行。

## 进度总览

| 维度 | 当前判断 | 备注 |
| --- | --- | --- |
| 黑盒攻击 | `较成熟` | `recon` 是当前最强证据线 |
| 灰盒攻击 | `最成熟` | `PIA` 已进入 real-asset runtime mainline |
| 白盒攻击 | `已冻结 admitted 主结果` | `GSA` 的 `epoch300 rerun1` 已写回 admitted 主结果，AUC 为 `0.998192` |
| 黑盒防御 | `基本未落地` | `B-1 / B-2` 仍在设计层 |
| 灰盒防御 | `已进入 provisional G-1 + adaptive gate completed` | `PIA GPU128/GPU256/GPU512` 三档与一次 `GPU512` 同档 repeat 都显示 `stochastic-dropout` 压低指标；新的 `GPU512` adaptive-reviewed baseline + `all_steps / late_steps_only` 已落地；round-26 的 `GPU128/GPU256 adaptive portability pair` 又在 `RTX4070 8GB` 上给出同向结果，其中 `GPU128` 是 quickest portable pair，`GPU256` 带 cost warning |
| 白盒防御 | `已有 full-scale 主结果，bridge diagnostic 已产生` | `DPDM` 已完成 `strong-v3 full-scale` defended comparator，并额外拿到 batch32 same-protocol diagnostic summary，但尚未进入 admitted 合同 |
| 统一评估表 | `已有第一版` | 已新增 admitted main results 的跨盒总表 |

当前阶段追加判断：

- `white-box same-protocol bridge` 已完成 `保持冻结` 收口
- 当前 active 主 GPU 问题已回到 `none`
- 当前 `PIA provenance dossier` 已 closed 为 `remain long-term blocker`
- `PIA 8GB portability ladder` 已完成 `probe + preview + GPU128/GPU256 adaptive pair`，当前 frontier 固定为 `GPU128 = quickest portable pair`、`GPU256 = decision rung with cost warning`
- `Finding NeMo + local memorization + FB-Mem` 的 intake/eligibility note 已建立，且 `activation export adapter` 已固定为 `decision-grade zero-GPU hold`
- 白盒 defense breadth 的第一轮 shortlist 也已经收口为负结论：
  - 当前 repo 只有 `DPDM / W-1` 这一条可执行 defended family
  - `Finding NeMo` 仍是 observability 路线
  - `Local Mirror` 不提供第二防御家族
- [2026-04-10-recon-decision-package](../workspaces/black-box/2026-04-10-recon-decision-package.md) 已把黑盒五件套固定为 decision-grade package，本轮 [recon-artifact-mainline-public-100-step30-reverify-20260410-round28](../experiments/recon-artifact-mainline-public-100-step30-reverify-20260410-round28/summary.json) 又在 CPU 上复算到相同 headline metrics，且不改 admitted 结果
- [2026-04-10-pia-provenance-split-protocol-delta](../workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md) 已把 `split shape aligned locally / random-four-split protocol still open / strict redo currently dirty` 三点固定为新的 provenance supplement
- 当前最值得推进的唯一目标切到：把双主线口径压实，并把 `SMP-LoRA` 的下一题固定成 `T06 optimizer/lr frontier`；`PIA provenance` 继续作为 CPU sidecar blocker；`recon` 当前进入 frozen maintenance，而不是继续扩 run

## 攻击主线

### 黑盒

- 主线：`recon`
- 次主线候选：`variation`（对应 `Towards Black-Box`）
- 当前能说的话：
  - 公开资产上的 black-box 风险已经有可引用主证据
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
- baseline：`SecMI`（当前已判定为 `blocked baseline`）
- 当前能说的话：
  - `PIA` 已经不是 smoke，而是真实资产 mainline
  - `PIA GPU128 / GPU256 / GPU512` 已拿到同口径 baseline + defense 对照，且 defense 指标连续三档都低于 baseline
  - `PIA GPU512` 同档 repeat 也继续维持 defense 优于 baseline
  - round-26 的 `GPU128 / GPU256 adaptive portability pair` 又在 `RTX4070 8GB` 上复现了同向下降，其中 `GPU128` 是当前 quickest portable pair，`GPU256` 则因 defense cost 升高而保留为 decision rung with cost warning
  - `pia_next_run --strict` 已通过，当前 asset line 已可写成 `workspace-verified`
  - 当前 `PIA` 攻击分数可以明确解释为 `epsilon-trajectory consistency` 信号，而不是泛化的 reconstruction score
  - `stochastic-dropout` 当前最可辩护的作用机理，是在推理时打散这一致性信号
  - 当前 gray-box 新一轮重点已从“多开 run”切到 `off / all_steps / late_steps_only + repeated-query adaptive review + structured quality/cost`
  - 新归档的 `TMIA-DM` 说明时间相关噪声 / 梯度信号也是灰盒成员推断的正式文献方向，但它当前仍只是 research-ready 候选
  - 新整理的 `PIA / TMIA-DM / SimA / MoFit` 文献轴已经统一到“时间 / 噪声 / 条件信号”叙事上
  - 当前最适合把防御压到这条线上做正式比较
- 当前不能说的话：
  - 还不能说灰盒防御已经验证有效
  - 当前用途：
  - 作为当前算法主讲线
  - 作为 `Local-API` contract-specific best summary 的首要 admitted 消费对象
  - 当前只允许写成 `workspace-verified + paper-alignment blocked by checkpoint/source provenance`
  - 截至 `2026-04-10`，`PIA provenance dossier` 已 closed 为 `remain long-term blocker`

### 白盒

- 主线：`GSA`
- 扩展：`Finding NeMo (intake-gated)`
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
  - 当前 same-protocol bridge 的关键训练阻塞已经从“`shadow-02` 无法落盘”收缩到“较高训练规模不稳定”；在清理 orphan `multiprocessing-fork` 后，`batch_size = 32` 已让 `shadow-02 / shadow-03` checkpoint 重新可得
  - 基于这组 batch32 checkpoint，新的 same-protocol diagnostic comparator 已经产出 [dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409](../workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409/summary.json)，指标为 `auc=0.541199 / asr=0.515625 / tpr@1%fpr=0.0 / tpr@0.1%fpr=0.0`
  - 这份 batch32 comparator 当前仍是 `runtime-smoke` 级 bridge 诊断结果，不应直接写成新的 admitted 白盒防御主结果
  - 当前 same-protocol bridge 已正式以 `保持冻结` 收口；这只是治理与资源排序决策，不是新的 benchmark 结果
  - 系统侧对白盒 `GSA` 的 live intake 现在应与 admitted `1k-3shadow` 主结果对齐，而不是继续停在早期 CPU closed-loop
  - 新的 [2026-04-10-finding-nemo-mechanism-intake](../workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md) 已把 `Finding NeMo + local memorization + FB-Mem` 固定为 intake/eligibility-only 候选，并把任何 future `validation-smoke` 压回 `separate release-review reconsideration` 的条件性上限
  - 新的 [2026-04-10-finding-nemo-protocol-reconciliation](../workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md) 已明确：当前 admitted 白盒资产与 `Finding NeMo` 原始 `Stable Diffusion v1.4 / cross-attention value layers` 协议面不兼容；当前只允许继续做 zero-GPU 的 observability 规划
  - 新的 [2026-04-10-finding-nemo-observability-smoke-contract](../workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md) 已把未来 smoke 的 `checkpoint_root / layer selector / sample binding / output schema / scheduler gate` 写成可审查合同；本轮又把它落实成 `read-only contract-probe`
  - `src/diffaudit/attacks/gsa_observability.py` 与 `probe-gsa-observability-contract` 已在 `Research` 内实现零 GPU 的合同解析入口，并已在真实 admitted 资产上返回 `status = ready`
  - 本轮新增 `export-gsa-observability-canary` 与 `export_gsa_observability_canary`，已在 `Research` 内实现 CPU-only 的 sample-pair activation export，并在 [finding-nemo-observability-canary-20260410-round24](../workspaces/white-box/runs/finding-nemo-observability-canary-20260410-round24/summary.json) 写出 `summary.json + records.jsonl + tensor artifacts`
  - 新的 [2026-04-10-finding-nemo-activation-export-adapter-review](../workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md) 已把这批实现正式固定为 `zero-GPU hold / queue not-requestable`
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
2. 将 [2026-04-13-smp-lora-t06-optimizer-lr-frontier-admission-packet](../workspaces/intake/2026-04-13-smp-lora-t06-optimizer-lr-frontier-admission-packet.md) 固定为 `SMP-LoRA` 的唯一下一题 packet
3. 将 [2026-04-09-pia-provenance-dossier](../workspaces/gray-box/2026-04-09-pia-provenance-dossier.md) 固定为 CPU sidecar blocker，并保持 `workspace-verified + paper-alignment blocked by checkpoint/source provenance` 不漂移
4. 保持 [2026-04-10-recon-decision-package](../workspaces/black-box/2026-04-10-recon-decision-package.md) 作为当前黑盒固定包，并明确它继续是 `writing-only / non-GPU / no admitted change`
4. `variation / Towards` 继续保留为 formal local secondary track，并明确 real-API assets blocked
5. 在统一表和叙事材料里补齐 `threat model / asset semantics / evidence level / external-validity boundary`
6. 用 [future-phase-e-intake](future-phase-e-intake.md) 与 [2026-04-10-phase-e-intake-ordering-review](../workspaces/intake/2026-04-10-phase-e-intake-ordering-review.md) 固定 `Phase E` 候选池排序，并只允许进入准入验证
5.1 用 [2026-04-10-intake-registry-phase-e-boundary-review](../workspaces/intake/2026-04-10-intake-registry-phase-e-boundary-review.md) 与 [phase-e-candidates.json](../workspaces/intake/phase-e-candidates.json) 把 machine-readable candidate ordering 从 `index.json.entries[]` 的 promoted contract 面里剥离出来
7. 用 [2026-04-10-secmi-unblock-decision](../workspaces/gray-box/2026-04-10-secmi-unblock-decision.md) 把 `SecMI` 固定为 `not-yet / remain blocked baseline`
8. 用 [2026-04-10-tmia-dm-intake-decomposition](../workspaces/gray-box/2026-04-10-tmia-dm-intake-decomposition.md) 把 `TMIA-DM` 固定为 `gray-box protocol / asset decomposition intake only`
9. 用 [2026-04-10-finding-nemo-mechanism-intake](../workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md)、[2026-04-10-finding-nemo-protocol-reconciliation](../workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md) 与 [2026-04-10-finding-nemo-observability-smoke-contract](../workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md) 固定 `Finding NeMo + local memorization + FB-Mem` 的 intake gate，并把它保留为当前最完整的 intake dossier
10. `PIA paper-aligned confirmation` 继续保留文档层条件性首位，但执行层视为 `no-go`
11. 基于第一版统一总表继续补质量 / 成本列，并保持灰盒机理说明与 adaptive gate 一致

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
- 防御文档索引：[mia-defense-research-index.md](mia-defense-research-index.md)
- 防御执行清单：[mia-defense-execution-checklist.md](mia-defense-execution-checklist.md)
- 研究仓路线图：[../ROADMAP.md](../ROADMAP.md)
