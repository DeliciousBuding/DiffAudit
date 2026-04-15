# 2026-04-12 SMP-LoRA GPU Expansion Program

## Status

- `candidate_key`: `white-box/dp-lora/cifar10-ddpm`
- `program_status`: `single-item historical runtime / O02 no-TF32 replicate closed`
- `queue_state`: `no active release; remainder blocked`
- `gpu_release`: `none`
- `admission`: `single-item provisional runtime release granted (2026-04-12 override)`
- `active_gpu_question`: `none`
- `next_question_packet`: `T06 optimizer/lr frontier`
- `current_mainline_not_changed`: `SMP-LoRA O02 no-TF32 replicate is now closed after repairing the default seed contract; no-TF32 rerun1/rerun2/rerun3 produced 0.3957 / 0.3838 / 0.5306, which is mixed and not a stabilization answer; batch14 throughput still remains the strongest candidate with variance note, not the default template; PIA provenance remains CPU-side blocker`
- `program_goal`: `preserve the no-TF32 mixed-result verdict and force the next GPU question to earn a new admission packet`

## Current Evidence Snapshot

- 当前真实 sweep 只有 6 组，不是 14 组
- 当前本地最优配置：
  - `lambda=0.1, rank=4, ep=10, AUC=0.34375, Acc=0.39473684210526316`
- 当前无防御基线：
  - `AUC=0.5565217391304348, Acc=0.5263157894736842`
- 当前 `100 epochs` 长训练结果：
  - `AUC=0.3784615384615384`
- 当前 `rank=1, lambda=0.1, ep=10, batch=12, throughput` 结果：
  - `AUC=0.40555555555555556, Acc=0.4473684210526316`
- 当前 `rank=2, lambda=0.1, ep=10, batch=12, throughput` 结果：
  - `AUC=0.4847222222222222, Acc=0.4473684210526316`
- 当前 `rank=4, lambda=0.1, ep=10, batch=12, throughput` 结果：
  - `AUC=0.5531746031746032, Acc=0.5`
- 当前 `rank=4, lambda=0.1, ep=10, batch=8, throughput` 结果：
  - `AUC=0.6159420289855072, Acc=0.5263157894736842`
- 当前 `rank=4, lambda=0.1, ep=10, batch=8, legacy replay1` 结果：
  - `AUC=0.4872159090909091, Acc=0.5526315789473685`
- 当前 `rank=4, lambda=0.1, ep=10, batch=8, legacy rerun2` 结果：
  - `AUC=0.4944444444444444, Acc=0.47368421052631576`
- 当前 `rank=1, lambda=0.1, ep=10, batch=12, throughput rerun2` 结果：
  - `AUC=0.42640692640692646, Acc=0.39473684210526316`
- 当前 `lambda=0.05` 状态：
  - `O01 rerun stalled after checkpoint step_10200`
  - `no final`
  - `salvage evaluation AUC=0.5770308123249299, Accuracy=0.5`

## Hard Boundary

- `This SMP-LoRA long-horizon list remains program-plan only / not released.`
- `queue_effect = O02 only; gpu_release = O02 only; admission_effect = O02 only; every non-O02 item remains blocked and non-released.`
- `execution-layer status = candidate only / not-requestable / not in current releasable queue.`
- `Local execution evidence exists, but comparability/intake hardening remains; it is not admitted, not queue released, not GPU released, and not a W-1 replacement.`
- `Any 336-item GPU-heavy backlog here is future candidate branches only; todolist ordering does not equal execution permission or GPU admission.`
- `Only one main GPU question may remain released at a time; the current released item is O02.`
- `diagnostic / intake / handover wording must not be written as execution complete or release complete.`

## Freeze Decision

- `freeze_round`: `round-80`
- `frozen_backlog`: `T01-T336`
- `future_numeric_expansion`: `forbidden by default`
- `T337+ policy`: `must not be added unless a separate governance review explicitly rewrites the gate and explains why the existing 336-item canon is insufficient`
- `reading_mode`: `prefer shortlist + audit canon + freeze charter + resume bundle before reading the full T01-T336 canon`
- `release_interpretation`: `historical O02 only; there is no active GPU release now, and freeze still applies to every non-O02 item`

## Admission Gate Before Any GPU Release

下列条件缺一不可；未满足前，所有 GPU heavy item 都保持 `blocked`：

1. 锁定单一 canonical comparator：
   - `SMP-LoRA under DDPM/CIFAR10 local protocol` vs `baseline` vs `W-1 strong-v3`
2. 明确当前允许的 headline 口径：
   - `under the current DDPM/CIFAR-10 local protocol, SMP-LoRA reduced observed GSA AUC from 0.5565 to 0.3438 on the best tested configuration`
3. 锁定单一评估协议：
   - 攻击器
   - 数据划分
   - 指标
   - seed
   - output schema
4. 对任何未来 GPU 项都写清：
   - `hypothesis`
   - `asset requirement`
   - `compute budget`
   - `stop conditions`
   - `expected artifact`
5. 除 `O02` 的单条 release 放行外：
   - 其余条目不得从 `blocked` 升到 `requestable`

## Released Runtime Window

- `released_item`: `O01 unattended low-lambda rescue run`
- `release_reason`: `2026-04-12 user-directed GPU utilization override; later closed as stalled-salvaged`
- `launch_pid`: `38028`
- `launch_output_dir`: `D:/Code/DiffAudit/Research/outputs/smp-lora-o01-lambda005-rerun-20260412-202122`
- `launch_stdout`: `D:/Code/DiffAudit/Research/outputs/smp-lora-o01-lambda005-rerun-20260412-202122/stdout.log`
- `launch_stderr`: `D:/Code/DiffAudit/Research/outputs/smp-lora-o01-lambda005-rerun-20260412-202122/stderr.log`
- `release_scope`: `historical O01 and O02 only; there is no current live GPU release`
- `runtime_tuning_status`: `future-launch tuning landed and verified on 2026-04-12; O02 is now sealed after batch14 throughput x7, batch14 legacy, workers4/6, no-bench, no-TF32, and seed123/42 comparators. The current verdict is: batch14 throughput is the strongest candidate with variance note; batch14 legacy proves the gain is throughput-dependent; batch13 is no-gain; batch15 is unstable no-go; batch16 is degraded; workers4/6/no-bench/seeded are not stabilization answers; no-TF32 has only one positive sample`
- `runtime_tuning_scope`: `recommended_num_workers helper + DataLoader persistent_workers/prefetch + intermediate checkpoints skip full training_log + throughput_mode defaults (bf16 autocast, TF32, cudnn benchmark, non_blocking transfers, save_every=500)`

## Execution Policy

- 同一时段最多只允许 1 条 `SMP-LoRA` 任务处于 released active-runtime
- 当前没有 released active item
- 其余条目全部留在 blocked queue
- 下一条 GPU item 在单独准入 packet 写完前不得放行
- 以下任一条件成立即停：
  - OOM 或显存需求超出预算 2 倍
  - 无法在规定 rung 内产生 `summary.json + config + seed + checkpoint pointer`
  - 指标没有给出方向性增量
  - 只是在重复旧结论
  - comparability 进一步恶化

## Future Shortlist

下面 12 项是未来若要做 release-review，只应先看的一页 shortlist。它们仍然全部是 `blocked`，这里只是为了避免后续继续在 `T337+` 上机械扩表：

1. `T01`: `lambda=0.05` 短程完整重跑
2. `T02`: `lambda=0.05` 100 epochs 长训练完整重跑
3. `T03`: 更低 `lambda` 前沿扫描：`0.01 / 0.02 / 0.05`
4. `T04`: `rank` 扫描：`1 / 2 / 4 / 8`
5. `T07`: 多随机种子稳定性重跑
6. `T10`: 自适应 GSA 攻击评估
7. `T11`: 图像质量面板：`FID / IS / PRD`
8. `T12`: `W-1 strong-v3` 与 `SMP-LoRA` 同协议统一比较
9. `T13`: 第二 target checkpoint 可迁移性验证
10. `T14`: 8GB 可迁移训练链与 batch/accum frontier
11. `T15`: 防御强度-质量-算力三维 frontier 汇总重跑
12. `T16`: 多 target checkpoint 交叉验证批次

shortlist 解释：

- 它只压缩未来最可能有信息增量的 blocked 候选
- 它不改变 `queue_state = not-requestable`
- 它不产生 `gpu_release`
- 它不构成 release queue
- 它不替代完整的 `T01-T336` 审计 canon

## Audit Canon

这份 program 文档当前必须始终被读成下面 8 条固定事实：

1. 当前真实 sweep 只有 6 组，不是 14 组
2. 当前最优本地配置是 `lambda=0.1 / rank=4 / epochs=10 / AUC=0.34375`
3. 当前无防御基线是 `AUC=0.5565217391304348`
4. 当前 `100 epochs` 长训练结果劣于 `10 epochs`
5. 当前 `lambda=0.05` 的 O01 rerun 已 stalled 于 `step_10200`，且 salvage evaluation 为 `AUC=0.5770 / Accuracy=0.5`，差于 baseline
6. 当前最合理口径只到 `comparability/intake hardening`
7. 当前 `SMP-LoRA` 不是 admitted 结果，不是 `W-1` 替代，也不是 release-ready object
8. 当前主线 GPU 问题已回到 `none`，而 `PIA provenance release/source identity unresolved CPU closure` 继续是 sidecar blocker

## Freeze Charter

- 不得把 `Wave 0-29 / Tier A-D / H1-H13 / Campaign A-T / R0 / SC-1..SC-5` 写成 queue opening
- 不得把 shortlist 写成 execution permission
- 不得把 `planning-only`、`handover`、`intake`、`diagnostic` 写成 execution complete
- 不得因 `GPU 空闲` 或用户持续催促而越过 release-review / admission gate
- 不得继续扩写 `T337+` 来制造“已经在推进模型主线”的错觉
- 若未来真的要提名单个 GPU rung，先补齐 `hypothesis / asset requirement / compute budget / stop conditions / expected artifact`

## Resume Bundle

后续接手优先读下面 6 项，而不是先从 `T01` 顺序读到 `T336`：

1. `docs/autonomous-research-director/state.json`
2. `docs/autonomous-research-director/latest.md`
3. `docs/autonomous-research-director/rounds/2026-04-12-1613-round-80.md`
4. `Research/workspaces/GPU_TRAINING_HANDOVER.md`
5. `Research/workspaces/intake/2026-04-11-dplora-comparability-note.md`
6. 本文件的 `Status / Current Evidence Snapshot / Hard Boundary / Future Shortlist / Audit Canon / Freeze Charter`

## Strategic Long-Horizon GPU Campaign Layer

这个 section 专门回应“需要十几个完整、可长期无人值守阅读的 GPU heavy 任务”的要求，但它不是新 queue，也不是 `T337+` 扩表。下面 16 个 campaign 只是在 `T01-T336` 与 `M01-M336` 之上再加一层长期管理视图，全部继续维持：

- `blocked`
- `not-requestable`
- `gpu_release = none`
- `admission_effect = none`

### C01. low-lambda recovery and completion campaign

- `status`: `blocked`
- `mapped_pack`: `T01-T03 + M01-M03`
- `goal`: 把 `lambda=0.05` interrupted residual 和更低 `lambda` 前沿压成第一组可比较完整结果
- `hypothesis`: 当前最优点可能仍在 `lambda=0.1` 以下，但必须先补齐 `lambda=0.05` 完整闭环才能判断
- `asset_requirement`: existing SMP-LoRA checkpoints、baseline evaluation schema、low-lambda configs
- `compute_budget`: `<= 28 GPUh`
- `stop_conditions`: `lambda=0.05` 仍无法稳定完成；更低 `lambda` 只复制旧结论；artifact 不完整
- `expected_artifact`: low-lambda completion packet with sweep table, final checkpoints, unified evaluation
- `why_blocked_now`: `release-review not passed; canonical comparator still not locked`

### C02. rank-capacity frontier campaign

- `status`: `blocked`
- `mapped_pack`: `T04 + T29-T30 + M04 + M20-M21`
- `goal`: 锁定 `rank / parameter budget / layer placement` 对隐私收益和显存包络的联合影响
- `hypothesis`: 当前收益可能由容量配置而非单一 `lambda` 主导
- `asset_requirement`: rank configs、layer placement variants、capacity accounting schema
- `compute_budget`: `<= 30 GPUh`
- `stop_conditions`: frontier 形态不稳定；容量解释不可复核；显存需求超预算
- `expected_artifact`: capacity frontier packet with memory envelope and best-rung rationale
- `why_blocked_now`: `capacity and placement contracts are not frozen`

### C03. epoch-length and early-stop campaign

- `status`: `blocked`
- `mapped_pack`: `T05 + T41-T42 + M05 + M32-M33`
- `goal`: 统一回答“短训最优是否稳定成立、长训是否只是在过拟合”
- `hypothesis`: `10 epochs` 最优可能是 narrow-window optimum，需与 `40/100/200 epochs` 和 early-stop 策略一起审查
- `asset_requirement`: long-run checkpoints、checkpoint scoring rule、early-stop templates
- `compute_budget`: `<= 44 GPUh`
- `stop_conditions`: 长训只复制过拟合；early-stop 规则不稳定；曲线无法支持决策
- `expected_artifact`: training-length and early-stop packet with curve summaries and decision rung
- `why_blocked_now`: `long-horizon stress and early-stop contracts are not approved`

### C04. seed and stability campaign

- `status`: `blocked`
- `mapped_pack`: `T07 + T39 + M06 + M30`
- `goal`: 证明当前 best rung 不是 seed 偶然值，并给出 top-rung 统一复核
- `hypothesis`: 当前 `lambda=0.1 / rank=4 / ep=10` 方向可能稳定，但离 decision-grade 还缺跨 seed 方差面
- `asset_requirement`: multi-seed configs、top-rung shortlist、stability summary schema
- `compute_budget`: `<= 32 GPUh`
- `stop_conditions`: mean/std 不稳定；top-5 互相矛盾；只能复现单次偶然结果
- `expected_artifact`: stability packet with seed statistics and top-rung audit canon
- `why_blocked_now`: `stability packet and shortlist review are not approved`

### C05. enlarged-evaluation and threshold campaign

- `status`: `blocked`
- `mapped_pack`: `T08 + T37 + M07 + M28`
- `goal`: 把当前小评估集与阈值曲线扩成更可审查的 comparability 面
- `hypothesis`: 当前 headline AUC 可能在更大评估集和不同 thresholding 下仍成立
- `asset_requirement`: larger eval set bindings、threshold curve schema、paired checkpoint panel
- `compute_budget`: `<= 24 GPUh`
- `stop_conditions`: enlarged eval 反转结论；阈值曲线不稳定；评估协议不一致
- `expected_artifact`: enlarged-eval packet with threshold panels and confidence summary
- `why_blocked_now`: `evaluation protocol and threshold interpretation are not locked`

### C06. adaptive-attack stress campaign

- `status`: `blocked`
- `mapped_pack`: `T10 + M08`
- `goal`: 在 adaptive threat 下重新审查当前非自适应收益
- `hypothesis`: SMP-LoRA 当前 observed gain 可能在 adaptive GSA 下被削弱甚至消失
- `asset_requirement`: adaptive attack protocol、paired defended/baseline checkpoints、attack logging schema
- `compute_budget`: `<= 26 GPUh`
- `stop_conditions`: adaptive protocol漂移；结果无法与 baseline/W-1 比较；artifact 不可审计
- `expected_artifact`: adaptive-attack packet with best/worst-case analysis
- `why_blocked_now`: `adaptive protocol is not release-reviewed`

### C07. privacy-quality tradeoff campaign

- `status`: `blocked`
- `mapped_pack`: `T11 + T38 + T43 + M09 + M29 + M34`
- `goal`: 把 `FID / IS / PRD / sample-budget` 侧的质量代价压成统一质量面板
- `hypothesis`: 当前隐私收益可能伴随可接受质量损失，但需要更完整 panel 才能说清
- `asset_requirement`: quality metrics pipeline、sample-budget variants、generated image bundles
- `compute_budget`: `<= 30 GPUh`
- `stop_conditions`: 质量协议不稳；sample budget 解释冲突；无法形成统一权衡面
- `expected_artifact`: privacy-quality tradeoff packet with multi-metric quality panels
- `why_blocked_now`: `quality contract and sample-budget policy are unresolved`

### C08. W-1 unified comparator campaign

- `status`: `blocked`
- `mapped_pack`: `T12 + T24 + M10 + M15`
- `goal`: 构造 `baseline vs SMP-LoRA vs W-1 strong-v3` 的单一 comparator 面，并扩到多攻击器
- `hypothesis`: SMP-LoRA 只有在同协议、多攻击面下仍保留优势，才值得准备 release-review
- `asset_requirement`: W-1 comparator assets、multi-attack panel schema、shared evaluation contract
- `compute_budget`: `<= 34 GPUh`
- `stop_conditions`: comparator 不同协议；多攻击器结论分裂；W-1 替代叙事失控
- `expected_artifact`: unified comparator packet across attack panels
- `why_blocked_now`: `canonical comparator is still not locked`

### C09. target portability campaign

- `status`: `blocked`
- `mapped_pack`: `T13 + T16 + T27 + M11 + M14 + M18`
- `goal`: 证明当前 best rung 不依赖单一 target checkpoint
- `hypothesis`: 当前收益可能是 target-specific；需要第二 target 和 cross-target consolidation 才能排除偶然性
- `asset_requirement`: second target checkpoints、cross-target manifest、paired evaluation schema
- `compute_budget`: `<= 40 GPUh`
- `stop_conditions`: 第二 target 不可比；cross-target 反转结论；consolidation 失败
- `expected_artifact`: cross-target portability packet with consolidated candidate summary
- `why_blocked_now`: `target portability scope is not approved`

### C10. split portability campaign

- `status`: `blocked`
- `mapped_pack`: `T17 + M17`
- `goal`: 检查当前结论是否对不同 split 形态仍成立
- `hypothesis`: 当前收益可能受当前 local split 结构强影响
- `asset_requirement`: alternative split definitions、split-bound checkpoint mapping、evaluation manifests
- `compute_budget`: `<= 24 GPUh`
- `stop_conditions`: split semantics 无法对齐；结果只在单 split 成立；artifact 不可比较
- `expected_artifact`: cross-split portability packet
- `why_blocked_now`: `split protocol contract is not frozen`

### C11. 8GB portability and device envelope campaign

- `status`: `blocked`
- `mapped_pack`: `T14 + T44 + M12 + M35`
- `goal`: 锁定单卡 8GB 可迁移训练链，并补跨 GPU 设备包络
- `hypothesis`: 当前结论可能依赖单一设备环境；需要显存档位与设备型号双重验证
- `asset_requirement`: 8GB configs、device inventory、runtime envelope schema
- `compute_budget`: `<= 32 GPUh`
- `stop_conditions`: 8GB rung 不稳；设备差异导致协议漂移；显存包络超预算
- `expected_artifact`: portability packet with 8GB ladder and cross-device envelope report
- `why_blocked_now`: `portable rung and device envelope are not approved`

### C12. cross-environment reproducibility campaign

- `status`: `blocked`
- `mapped_pack`: `T45 + M36`
- `goal`: 验证不同软件栈/驱动/环境下 artifact 是否一致
- `hypothesis`: 当前结果可能部分依赖环境细节，而非方法本身
- `asset_requirement`: environment matrix、repro configs、checksum and artifact schema
- `compute_budget`: `<= 26 GPUh`
- `stop_conditions`: 环境差异导致结果不可复现；artifact schema 不稳定；无法形成复核包
- `expected_artifact`: cross-environment reproducibility packet
- `why_blocked_now`: `environment reproducibility contract is unresolved`

### C13. longrun monitoring and recovery campaign

- `status`: `blocked`
- `mapped_pack`: `T34 + T46 + T50 + M25 + M37 + M41`
- `goal`: 为真正无人值守长训补齐日志、checkpoint、恢复和 smoke 机制
- `hypothesis`: 当前最大缺口不是“还能再跑什么”，而是“长训挂起后如何可审计恢复”
- `asset_requirement`: logging schema、checkpoint retention policy、recovery hooks、smoke ladders
- `compute_budget`: `<= 24 GPUh`
- `stop_conditions`: 日志与 checkpoint 不可对齐；恢复后 artifact 不一致；无人值守 smoke 无法界定
- `expected_artifact`: unattended monitoring and recovery packet
- `why_blocked_now`: `monitoring and recovery contracts are not release-approved`

### C14. budget and duration governance campaign

- `status`: `blocked`
- `mapped_pack`: `T47-T49 + T56-T60 + M31 + M39-M40 + M47-M50 + M331`
- `goal`: 把长时 GPU 项的预算、时长、失败模式和 no-go/go criteria 压成统一治理梯子
- `hypothesis`: 当前最容易失控的是治理层，不是单个 rung 本身
- `asset_requirement`: budget ladders、duration slots、no-go/go criteria、dependency atlas
- `compute_budget`: `<= 16 GPUh planning-side + future bounded checks`
- `stop_conditions`: budget ladder 与 stop rules 冲突；失败模式 taxonomy 不稳；duration slot 不可执行
- `expected_artifact`: budget-duration governance packet
- `why_blocked_now`: `governance ladder is planning only and must not become execution permit`

### C15. checkpoint escrow and watchdog campaign

- `status`: `blocked`
- `mapped_pack`: `M333-M335`
- `goal`: 为未来真正长时 run 预先定义 watchdog、升级树和 checkpoint 托管策略
- `hypothesis`: 没有 watchdog/escrow，任何无人值守长训都会变成不可交接的黑箱
- `asset_requirement`: watchdog schema、escalation tree、checkpoint escrow policy
- `compute_budget`: `<= 10 GPUh planning-side + future smoke checks`
- `stop_conditions`: watchdog 不能覆盖关键失败点；checkpoint escrow 不可恢复；升级树模糊
- `expected_artifact`: watchdog and checkpoint escrow packet
- `why_blocked_now`: `handoff safety layer is not approved`

### C16. final release-review commonwealth campaign

- `status`: `blocked`
- `mapped_pack`: `T330-T336 + M321-M336`
- `goal`: 把所有 freeze/rehearsal/verdict/commonwealth 层压成最终 release-review 前置大包
- `hypothesis`: 即使未来准备放行，也必须先有一份可以只读接手的终版 commonwealth 包
- `asset_requirement`: commonwealth scoreboards、handoff doctrine、freeze charter、resume-state ladder
- `compute_budget`: `<= 14 GPUh planning-side + future bounded audits`
- `stop_conditions`: commonwealth 包仍需扩表；resume-state 不完整；任何 charter 仍能被误读为放行
- `expected_artifact`: final release-review commonwealth bundle
- `why_blocked_now`: `commonwealth remains fail-closed and not release-ready`

### Campaign Hard Rule

- `C01-C16` 只是 `T01-T336 / M01-M336` 的长期战略压缩视图
- 它们的排序只表达未来信息增量优先级，不表达执行许可
- 它们全部仍然是 `blocked GPU-heavy campaigns`
- `campaign_status != released queue item`
- `campaign_presence != admission`
- `campaign_presence != gpu_release`

## Unattended Long-Run Operations Board

这个 section 继续回应“进入无人值守长时间任务状态”的诉求，但仍然只是一层 planning-only 的 operations board。下面 `O01-O16` 把 `C01-C16` 进一步细化成更像真实长时 runbook 的 operation 条目；它们全部仍然是：

- `blocked`
- `not-requestable`
- `gpu_release = none`
- `execution_effect = none`

### O01. unattended low-lambda rescue run

- `status`: `stalled-closed / salvaged-evaluation`
- `maps_to`: `C01`
- `duration_target`: `8-12h`
- `checkpoint_cadence`: `every 250-500 steps`
- `heartbeat_cadence`: `every 30 min`
- `objective`: 补齐 `lambda=0.05` interrupted residual 并完成低 `lambda` 初轮 rescue
- `escalation_trigger`: `2 consecutive failed resumes` or `artifact incomplete at end`
- `stop_conditions`: no final checkpoint; no evaluation; lower lambda only repeats old outcome
- `expected_artifact`: rescue summary, final weights, evaluation packet, resume log
- `runtime_note`: `stalled after checkpoint step_10200; python CPU delta=0 over 8s; stdout/stderr stopped; salvage evaluation from step_10200 produced AUC=0.5770 / Accuracy=0.5 and no final/`
- `why_released_now`: `single-item override release was granted, but O01 closed as failed-stalled evidence and is no longer the active operation-layer item`

### O02. unattended capacity frontier sweep

- `status`: `active-runtime / released-exception`
- `maps_to`: `C02`
- `duration_target`: `12-18h`
- `checkpoint_cadence`: `per rung completion + final merged manifest`
- `heartbeat_cadence`: `every 45 min`
- `objective`: 在 rank/capacity/layer placement 维度上跑出首版显存-隐私 frontier；completed rungs = rank1/rank2/rank4(batch12); current released rung = lambda=0.1, rank=4, epochs=10, batch=8
- `escalation_trigger`: `OOM on 2+ adjacent rungs` or `capacity schema drift`
- `stop_conditions`: capacity explanation unstable; memory envelope exceeds budget; outputs incomparable
- `expected_artifact`: frontier table, memory envelope report, best-capacity shortlist
- `runtime_note`: `completed rungs: rank1 => AUC 0.4056, rank2 => AUC 0.4847, rank4(batch12) => AUC 0.5532; current released rung uses throughput_mode defaults + batch_size=8 + save_every=500`
- `why_released_now`: `O01 low-lambda route failed to beat baseline, so the next single GPU question moved to the first capacity frontier rung`

### O03. unattended epoch stress and early-stop run

- `status`: `blocked`
- `maps_to`: `C03`
- `duration_target`: `16-24h`
- `checkpoint_cadence`: `every epoch plus milestone checkpoints`
- `heartbeat_cadence`: `every 30 min`
- `objective`: 把 `10/40/100/200 epochs` 与 early-stop policy 放到一条长训 stress line 上
- `escalation_trigger`: `overfitting boundary crossed twice` or `checkpoint scoring rule diverges`
- `stop_conditions`: curve provides no new decision signal; longrun only worsens metrics; early-stop policy unstable
- `expected_artifact`: training curves, early-stop verdict, overfitting boundary packet
- `why_blocked_now`: `longrun stress protocol is not approved`

### O04. unattended multi-seed stability batch

- `status`: `blocked`
- `maps_to`: `C04`
- `duration_target`: `10-16h`
- `checkpoint_cadence`: `per seed final checkpoint`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 对 best rung 做多 seed 长时稳定性复核
- `escalation_trigger`: `variance exceeds review threshold` or `seed failures > 25%`
- `stop_conditions`: mean/std unstable; top-rung audit contradicts headline; seed completion rate too low
- `expected_artifact`: seed stability matrix, variance packet, audited best-rung set
- `why_blocked_now`: `seed stability packet is not approved`

### O05. unattended enlarged-eval board

- `status`: `blocked`
- `maps_to`: `C05`
- `duration_target`: `8-14h`
- `checkpoint_cadence`: `per evaluation shard`
- `heartbeat_cadence`: `every 45 min`
- `objective`: 用更大评估面和 threshold panels 审查当前 headline
- `escalation_trigger`: `evaluation shard mismatch` or `threshold panel drift`
- `stop_conditions`: enlarged eval reverses direction; threshold panels unstable; schema mismatch
- `expected_artifact`: enlarged-eval summary, threshold panel bundle, confidence report
- `why_blocked_now`: `evaluation expansion contract is not approved`

### O06. unattended adaptive stress board

- `status`: `blocked`
- `maps_to`: `C06`
- `duration_target`: `12-18h`
- `checkpoint_cadence`: `per adaptive rung`
- `heartbeat_cadence`: `every 45 min`
- `objective`: 跑出 adaptive threat 下的首版 SMP-LoRA stress packet
- `escalation_trigger`: `adaptive protocol mismatch` or `no comparable baseline after 2 rungs`
- `stop_conditions`: adaptive attack setup drifts; results not comparable to baseline/W-1; artifacts incomplete
- `expected_artifact`: adaptive stress packet, worst-case summary, protocol audit note
- `why_blocked_now`: `adaptive runbook is not release-reviewed`

### O07. unattended privacy-quality board

- `status`: `blocked`
- `maps_to`: `C07`
- `duration_target`: `12-20h`
- `checkpoint_cadence`: `per metric family and per sample-budget rung`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 在质量与隐私双面板上给当前防御收益补齐 sidecar
- `escalation_trigger`: `quality pipeline breaks` or `sample-budget schema conflicts`
- `stop_conditions`: tradeoff panel cannot close; quality metrics disagree without explanation; image bundle incomplete
- `expected_artifact`: privacy-quality dashboard packet, generated samples bundle, metric summary
- `why_blocked_now`: `quality sidecar contract is unresolved`

### O08. unattended W-1 comparator board

- `status`: `blocked`
- `maps_to`: `C08`
- `duration_target`: `14-22h`
- `checkpoint_cadence`: `per comparator rung`
- `heartbeat_cadence`: `every 45 min`
- `objective`: 生成 `baseline vs SMP-LoRA vs W-1` 的长期 comparator operation board
- `escalation_trigger`: `protocol mismatch across comparator branches` or `W-1 mapping ambiguity`
- `stop_conditions`: comparator not same-protocol; attack panels diverge without closure; release wording risk appears
- `expected_artifact`: unified comparator packet, multi-attack board, release-review gap note
- `why_blocked_now`: `same-protocol comparator is not approved`

### O09. unattended cross-target board

- `status`: `blocked`
- `maps_to`: `C09`
- `duration_target`: `16-24h`
- `checkpoint_cadence`: `per target and per consolidation stage`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 把第二 target 与 cross-target consolidation 做成可无人值守批次
- `escalation_trigger`: `second target invalid` or `cross-target disagreement exceeds threshold`
- `stop_conditions`: portability does not hold; consolidation collapses; target mapping incomplete
- `expected_artifact`: cross-target packet, consolidation matrix, portability verdict
- `why_blocked_now`: `target portability scope is not release-approved`

### O10. unattended cross-split board

- `status`: `blocked`
- `maps_to`: `C10`
- `duration_target`: `12-18h`
- `checkpoint_cadence`: `per split rung`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 用多 split operation board 检查当前结论是否只依赖单一 split
- `escalation_trigger`: `split semantics mismatch` or `artifact joins fail`
- `stop_conditions`: splits not comparable; outcome only holds on one split; manifests incomplete
- `expected_artifact`: cross-split packet, split-join manifest, portability note
- `why_blocked_now`: `split portability protocol is unresolved`

### O11. unattended 8GB and device ladder board

- `status`: `blocked`
- `maps_to`: `C11`
- `duration_target`: `14-22h`
- `checkpoint_cadence`: `per device rung and per 8GB milestone`
- `heartbeat_cadence`: `every 45 min`
- `objective`: 把 8GB ladder 与 cross-device envelope 做成真正的长期操作板
- `escalation_trigger`: `device-specific OOM cluster` or `runtime envelope drift`
- `stop_conditions`: 8GB rung collapses; device mismatch breaks comparability; cadence logs incomplete
- `expected_artifact`: 8GB ladder packet, device envelope report, runtime audit logs
- `why_blocked_now`: `device-envelope release review has not passed`

### O12. unattended cross-environment reproducibility board

- `status`: `blocked`
- `maps_to`: `C12`
- `duration_target`: `12-18h`
- `checkpoint_cadence`: `per environment completion`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 形成环境矩阵下的长期可复核 operation board
- `escalation_trigger`: `checksum divergence across environments` or `artifact schema mismatch`
- `stop_conditions`: environment results irreconcilable; checksums drift; packet not reproducible
- `expected_artifact`: environment reproducibility matrix, checksum pack, environment verdict
- `why_blocked_now`: `reproducibility board is not approved`

### O13. unattended monitoring and recovery board

- `status`: `blocked`
- `maps_to`: `C13`
- `duration_target`: `18-30h`
- `checkpoint_cadence`: `time-based plus milestone-based`
- `heartbeat_cadence`: `every 20 min`
- `objective`: 为未来真正长时 run 补齐 checkpoint/heartbeat/recovery 操作层
- `escalation_trigger`: `missed heartbeat twice` or `checkpoint gap exceeds tolerance`
- `stop_conditions`: recovery not deterministic; heartbeat gaps unresolved; monitoring schema incomplete
- `expected_artifact`: monitoring-recovery board, heartbeat logs, recovery audit packet
- `why_blocked_now`: `unattended monitoring policy is not release-approved`

### O14. unattended budget-duration governance board

- `status`: `blocked`
- `maps_to`: `C14`
- `duration_target`: `8-12h planning-side plus future bounded checks`
- `checkpoint_cadence`: `per governance rung closure`
- `heartbeat_cadence`: `every 90 min`
- `objective`: 把预算、时长、升级与 no-go/go criteria 变成可执行前的操作治理板
- `escalation_trigger`: `budget ladder conflict` or `duration slot ambiguity`
- `stop_conditions`: go/no-go charters drift; duration slots unworkable; governance packet contradicts stop rules
- `expected_artifact`: budget-duration governance board, escalation ladder, no-go/go packet
- `why_blocked_now`: `governance board is planning only`

### O15. unattended checkpoint escrow board

- `status`: `blocked`
- `maps_to`: `C15`
- `duration_target`: `8-12h planning-side plus smoke checks`
- `checkpoint_cadence`: `every escrow window`
- `heartbeat_cadence`: `every 60 min`
- `objective`: 让 watchdog、escalation tree、checkpoint escrow 具备操作层可交接结构
- `escalation_trigger`: `escrow retention failure` or `watchdog blind spot found`
- `stop_conditions`: escrow unrecoverable; watchdog misses critical failure; escalation path unclear
- `expected_artifact`: escrow board, watchdog packet, escalation map
- `why_blocked_now`: `safety handoff board is not approved`

### O16. unattended final commonwealth board

- `status`: `blocked`
- `maps_to`: `C16`
- `duration_target`: `10-16h planning-side plus bounded audits`
- `checkpoint_cadence`: `per commonwealth sub-bundle`
- `heartbeat_cadence`: `every 90 min`
- `objective`: 把 commonwealth 终版包整理成真正可只读接手的 operation board
- `escalation_trigger`: `resume-state incomplete` or `handoff doctrine conflicts`
- `stop_conditions`: commonwealth bundle still expandable; resume-state ladder incomplete; any charter implies release
- `expected_artifact`: final commonwealth operations board, resume-state bundle, handoff packet
- `why_blocked_now`: `commonwealth remains fail-closed and not release-ready`

### Operations Hard Rule

- `O01-O16` 只是 `C01-C16` 的操作层压缩视图
- 它们的存在只表示“未来若要做无人值守长训，需要哪些 operation fields”；当前只有 `O01` 例外地被单条放行
- `O02-O16` 仍然全部是 `blocked GPU-heavy operations`
- `operation_status != released run` 对 `O02-O16` 仍然成立
- `operation_status != scheduler permit`
- `operation_status != gpu_release`

## Managed Long-Horizon Todolist

这个 section 专门回应“需要一眼能看懂的长时 GPU heavy todolist”。它不新增 `T337+`，只把已有 canon 中最值得未来做 release-review 的 16 个长任务重排成管理视图。除非 gate 被单独改写，否则下面全部维持 `blocked / not-requestable / gpu_release = none`。

### M01. `T01 lambda=0.05` 短程完整重跑

- `status`: `blocked`
- `priority_band`: `highest within SMP-LoRA blocked queue`
- `goal`: 验证当前中断残留是否值得补成首个可比较完整 rung
- `compute_budget`: `<= 4 GPUh`
- `acceptance`: 写出 `summary.json + evaluation.json + final weights + config`
- `why_blocked_now`: `release-review not passed`

### M02. `T02 lambda=0.05` 100 epochs 长训练完整重跑

- `status`: `blocked`
- `priority_band`: `highest within SMP-LoRA blocked queue`
- `goal`: 判断更低 `lambda` 的长训是否优于当前 `100 epochs AUC=0.37846`
- `compute_budget`: `<= 12 GPUh`
- `acceptance`: 写出长训练曲线、最终评估、最优 checkpoint 选择记录
- `why_blocked_now`: `release-review not passed`

### M03. `T03` 更低 `lambda` 前沿扫描：`0.01 / 0.02 / 0.05`

- `status`: `blocked`
- `priority_band`: `highest within SMP-LoRA blocked queue`
- `goal`: 判断最优点是否位于 `lambda=0.1` 以下
- `compute_budget`: `<= 12 GPUh`
- `acceptance`: 形成扫描表与最优点摘要
- `why_blocked_now`: `canonical comparator still not locked`

### M04. `T04 rank` 扫描：`1 / 2 / 4 / 8`

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 锁定 rank frontier 与显存包络
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 形成 rank frontier 表、显存记录、最佳 rank 说明
- `why_blocked_now`: `8GB portability contract not yet frozen`

### M05. `T05 epochs` 前沿扫描：`5 / 10 / 15 / 20 / 40`

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 判断当前最优是否来自窄窗口而非长训练
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 epoch frontier 图与 early-stop 建议
- `why_blocked_now`: `no canonical early-stop rule`

### M06. `T07` 多随机种子稳定性重跑

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 判断 `lambda=0.1 / rank=4 / ep=10` 是否只是 seed 偶然值
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 mean/std 与方向稳定性摘要
- `why_blocked_now`: `stability packet missing`

### M07. `T08` 更大评估集重复测量

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 把当前 `63/63` 评估提升到更可审查的 comparability 面
- `compute_budget`: `<= 10 GPUh`
- `acceptance`: 形成 enlarged-eval summary 与方差说明
- `why_blocked_now`: `eval protocol not locked`

### M08. `T10` 自适应 GSA 攻击评估

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 判断当前非自适应收益在 adaptive threat 下是否保留
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 adaptive attack report 与 best/worst-case packet
- `why_blocked_now`: `adaptive protocol not locked`

### M09. `T11` 图像质量面板：`FID / IS / PRD`

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 给隐私收益补齐质量 sidecar
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 privacy-quality tradeoff summary
- `why_blocked_now`: `quality contract missing`

### M10. `T12` `W-1 strong-v3` 与 `SMP-LoRA` 同协议统一比较

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 构造 `baseline vs SMP-LoRA vs W-1` 的单一 comparator 面
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 写出 unified comparator packet
- `why_blocked_now`: `canonical comparator missing`

### M11. `T13` 第二 target checkpoint 可迁移性验证

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 验证当前收益是否为单一 target 偶然现象
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 形成 portability note 与 cross-target summary
- `why_blocked_now`: `target portability not scoped`

### M12. `T14` 8GB 可迁移训练链与 batch/accum frontier

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 锁定单卡 8GB 可运行的 portable rung
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 8GB portability packet 与 runtime envelope
- `why_blocked_now`: `portable rung not proven`

### M13. `T15` 防御强度-质量-算力三维 frontier 汇总重跑

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把零散单点压成 decision-grade 3D frontier
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 final frontier packet 与 release-review candidate summary
- `why_blocked_now`: `depends on partial closure of M01-M12`

### M14. `T16` 多 target checkpoint 交叉验证批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 给当前 best rung 补跨 checkpoint 可迁移性
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 cross-target packet 与 mean/std 汇总
- `why_blocked_now`: `cross-target scope not locked`

### M15. `T24` 多攻击面联合评估批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 在 loss/gradient/proxy 多攻击面下审查同一批 checkpoint
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 multi-attack evaluation packet
- `why_blocked_now`: `attack panel unification missing`

### M16. `T25` 多 shadow 扩容重跑

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 评估更大 shadow 数量是否显著降低 comparability 方差
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 shadow-expansion packet
- `why_blocked_now`: `shadow asset scope and budget not approved`

### M17. `T26` 跨 split 可迁移性重跑

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 判断当前收益是否对不同 split 形态依然成立
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 cross-split portability packet
- `why_blocked_now`: `split protocol contract not frozen`

### M18. `T27` 双目标 release-candidate consolidation 重跑

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 对最优两条 target 路线做合并式候选复核
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 dual-target consolidation summary
- `why_blocked_now`: `candidate release packet not approved`

### M19. `T28` 最终大包汇总批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把前面多个 frontier 和 portability 结果压成一个 release-review 大包
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 final batch packet with unified schema
- `why_blocked_now`: `depends on closure of earlier management items`

### M20. `T29` LoRA layer placement 扫描

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 检查不同 LoRA 挂载层位是否改变隐私-质量折中
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 layer-placement frontier
- `why_blocked_now`: `placement contract not scoped`

### M21. `T30` LoRA 参数量 frontier

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 检查总参数量变化是否比 rank 更能解释结果
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 parameter-budget frontier
- `why_blocked_now`: `capacity contract not frozen`

### M22. `T31` Mixed precision / full precision 对照

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 排除数值精度路径对当前结论的扭曲
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 precision comparator summary
- `why_blocked_now`: `runtime/precision schema not locked`

### M23. `T32` 训练集规模扩展批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 评估训练集扩大后收益是否稳定
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 train-scale expansion packet
- `why_blocked_now`: `asset scale-up not approved`

### M24. `T33` 训练步数与评估步数联合网格

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 同时锁定 train-epoch 与 eval-step 的联合最优区域
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 joint grid summary
- `why_blocked_now`: `joint protocol grid not frozen`

### M25. `T34` 多次重启恢复鲁棒性测试

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 验证长训中断/恢复后 artifact 是否仍一致
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 restart-resume robustness report
- `why_blocked_now`: `recovery contract not approved`

### M26. `T35` 多 run ensemble 评估

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 检查 ensemble 是否让 SMP-LoRA 表现更稳定或更可解释
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 multi-run ensemble packet
- `why_blocked_now`: `ensemble interpretation not approved`

### M27. `T36` LoRA merge/unmerge 可复现性批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 保证 merge/unmerge 不引入不可见差异
- `compute_budget`: `<= 12 GPUh`
- `acceptance`: 形成 merge reproducibility note with paired eval
- `why_blocked_now`: `artifact reproducibility contract unresolved`

### M28. `T37` 跨攻击器阈值曲线重跑

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 在不同 thresholding 方式下检查曲线形态是否稳健
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 threshold-curve panel
- `why_blocked_now`: `curve interpretation contract unresolved`

### M29. `T38` 多质量指标联合 panel

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 形成比单一 FID 更完整的质量 sidecar
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 形成 multi-quality panel
- `why_blocked_now`: `quality panel contract unresolved`

### M30. `T39` 最佳 5 个 rung 的全量复核批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把最优 5 个候选 rung 做全量统一复核
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 top-5 audit packet
- `why_blocked_now`: `top-rung shortlist not release-approved`

### M31. `T40` 最终长期无人值守候选运行册

- `status`: `blocked`
- `priority_band`: `high once all prior evidence exists`
- `goal`: 形成真正面向未来无人值守执行的 runbook
- `compute_budget`: `<= 8 GPUh planning-side + future decisive runs`
- `acceptance`: 形成 unattended candidate runbook with budgets and stop rules
- `why_blocked_now`: `cannot become active runbook before release-review`

### M32. `T41` 超长训练 200 epochs 过拟合边界批次

- `status`: `blocked`
- `priority_band`: `high for long-horizon stress`
- `goal`: 明确 200 epochs 是否只会加重过拟合，还是能带来新最优点
- `compute_budget`: `<= 28 GPUh`
- `acceptance`: 形成 overfitting-boundary report with checkpoint curve
- `why_blocked_now`: `long-horizon stress not approved`

### M33. `T42` 多阶段 early-stop 策略批次

- `status`: `blocked`
- `priority_band`: `high for unattended long-run control`
- `goal`: 把长时训练的提前停止规则从经验判断收敛成可执行策略
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 形成 multi-stage early-stop policy packet
- `why_blocked_now`: `early-stop contract not approved`

### M34. `T43` 不同 sample budget 下的质量-隐私联合批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 评估 sample budget 变化对质量与隐私收益的联合影响
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 sample-budget tradeoff panel
- `why_blocked_now`: `sample-budget policy not frozen`

### M35. `T44` 跨 GPU 设备包络验证

- `status`: `blocked`
- `priority_band`: `high for portability`
- `goal`: 验证不同 GPU 型号/显存档位上的 runtime 包络和结论稳定性
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 cross-device envelope report
- `why_blocked_now`: `device envelope scope and budget not approved`

### M36. `T45` 跨环境可复现性批次

- `status`: `blocked`
- `priority_band`: `high for unattended reproducibility`
- `goal`: 验证不同软件/驱动/环境下 artifact 是否可重复
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 cross-environment reproducibility packet
- `why_blocked_now`: `environment reproducibility contract unresolved`

### M37. `T46` 训练日志与监控信号对齐批次

- `status`: `blocked`
- `priority_band`: `high for unattended monitoring`
- `goal`: 让无人值守长训的日志、checkpoint、监控信号保持可对齐
- `compute_budget`: `<= 12 GPUh`
- `acceptance`: 形成 logging-monitor alignment packet
- `why_blocked_now`: `monitoring schema not approved`

### M38. `T47` 梯度统计面板批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 补齐梯度分布、范数与不稳定性信号面板
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 gradient-statistics panel
- `why_blocked_now`: `telemetry interpretation contract unresolved`

### M39. `T48` 权重漂移轨迹批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 对长训练中的权重漂移轨迹做统一记录和解释
- `compute_budget`: `<= 14 GPUh`
- `acceptance`: 形成 weight-drift trajectory report
- `why_blocked_now`: `weight-drift schema not frozen`

### M40. `T49` 失败模式专用重跑批次

- `status`: `blocked`
- `priority_band`: `high for robustness`
- `goal`: 专门围绕已知失败模式做重跑和回放验证
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 failure-mode replay packet
- `why_blocked_now`: `failure taxonomy not approved`

### M41. `T50` 长时无人值守 smoke 候选批次

- `status`: `blocked`
- `priority_band`: `high for unattended readiness`
- `goal`: 构造第一条真正面向无人值守执行的 smoke 候选路径
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 unattended smoke candidate note with stop rules
- `why_blocked_now`: `cannot become active unattended smoke before release-review`

### M42. `T51` Wave 间依赖一致性审查批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 验证各 wave 之间的依赖、输入输出和 gate 没有互相冲突
- `compute_budget`: `<= 10 GPUh planning-side + future bounded reruns`
- `acceptance`: 形成 inter-wave dependency audit
- `why_blocked_now`: `wave-dependency audit not yet approved`

### M43. `T52` 候选 shortlist 生成批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 从更长管理层中压缩出真正可能进入 release-review 的候选 shortlist
- `compute_budget`: `<= 8 GPUh planning-side + future bounded evals`
- `acceptance`: 形成 candidate shortlist packet
- `why_blocked_now`: `shortlist cannot become runnable queue`

### M44. `T53` release-review 审查面模拟批次

- `status`: `blocked`
- `priority_band`: `high for gate readiness`
- `goal`: 提前模拟 release-review 会要求的证据面与失败点
- `compute_budget`: `<= 10 GPUh planning-side + future bounded rechecks`
- `acceptance`: 形成 release-review simulation packet
- `why_blocked_now`: `review simulation is not release approval`

### M45. `T54` final claim hierarchy 收口批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把 headline claim、secondary claim、boundary claim 压成不可漂移层级
- `compute_budget`: `<= 6 GPUh planning-side + future consistency reruns`
- `acceptance`: 形成 final claim hierarchy packet
- `why_blocked_now`: `claim hierarchy not approved for release`

### M46. `T55` artifact completeness gate 批次

- `status`: `blocked`
- `priority_band`: `high for unattended handoff`
- `goal`: 定义无人值守长时任务必须产出的完整 artifact 集合
- `compute_budget`: `<= 8 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 artifact completeness gate note
- `why_blocked_now`: `artifact gate unresolved`

### M47. `T56` budget escalation ladder 批次

- `status`: `blocked`
- `priority_band`: `high for long-horizon governance`
- `goal`: 把预算升级路径从单点申请改成分层 ladder
- `compute_budget`: `<= 6 GPUh planning-side + future budgeted checkpoints`
- `acceptance`: 形成 budget escalation ladder
- `why_blocked_now`: `budget ladder not approved`

### M48. `T57` final no-go criteria 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed governance`
- `goal`: 明确最终什么情况下必须判定这条线继续 no-go
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final no-go criteria charter
- `why_blocked_now`: `no-go charter must not be mistaken for release charter`

### M49. `T58` future go criteria 批次

- `status`: `blocked`
- `priority_band`: `highest for eventual release-review readability`
- `goal`: 明确未来在什么最小条件下才能从 no-go 进入 go-review
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 future go criteria charter
- `why_blocked_now`: `go criteria must not be mistaken for actual approval`

### M50. `T59` 最终执行顺序与依赖图批次

- `status`: `blocked`
- `priority_band`: `high for task-system readability`
- `goal`: 把长链任务的先后顺序、依赖和并行边界压成单一依赖图
- `compute_budget`: `<= 8 GPUh planning-side + future consistency checks`
- `acceptance`: 形成 execution-order and dependency atlas
- `why_blocked_now`: `dependency graph is planning only, not execution permit`

### M51. `T60` 最终长期 blocked program 总册

- `status`: `blocked`
- `priority_band`: `high for handoff`
- `goal`: 把整个 blocked program 压成单一总册，便于无人值守前阅读
- `compute_budget`: `<= 8 GPUh planning-side + future audit rechecks`
- `acceptance`: 形成 final blocked program masterbook
- `why_blocked_now`: `masterbook is not a release queue`

### M52. `T61` 最佳配置三次长跑一致性批次

- `status`: `blocked`
- `priority_band`: `high for long-run confidence`
- `goal`: 对最佳配置做三次长跑，判断长期一致性
- `compute_budget`: `<= 30 GPUh`
- `acceptance`: 形成 triple-longrun consistency report
- `why_blocked_now`: `long-run consistency packet not approved`

### M53. `T62` loss 组合系数 frontier 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 判断 loss 组合权重是否比 `lambda` 更能解释收益变化
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 loss-mixture frontier
- `why_blocked_now`: `loss-mixture contract unresolved`

### M54. `T63` 可训练 block 范围扫描批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 判断可训练 block 范围变化是否显著影响隐私-质量折中
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 trainable-block frontier
- `why_blocked_now`: `trainable-block scope not frozen`

### M55. `T64` 更大 shadow 数量 5/7 扩容批次

- `status`: `blocked`
- `priority_band`: `high for robustness`
- `goal`: 将 shadow 数量从当前面进一步扩到 5/7，检查方差与稳定性
- `compute_budget`: `<= 28 GPUh`
- `acceptance`: 形成 5/7-shadow expansion packet
- `why_blocked_now`: `shadow scale-up budget not approved`

### M56. `T65` 长时蒸馏/压缩副作用批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 审查更长训练中可能出现的蒸馏式或压缩式副作用
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 compression-side-effect report
- `why_blocked_now`: `side-effect interpretation contract unresolved`

### M57. `T66` 困难样本 stress 集重跑批次

- `status`: `blocked`
- `priority_band`: `high for adversarial robustness`
- `goal`: 在困难样本 stress 集上重跑最佳候选，检查最坏情况退化
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 hard-sample stress packet
- `why_blocked_now`: `stress-set scope and semantics not approved`

### M58. `T67` 时间切片重评批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 按训练时间切片重评模型，寻找收益/退化发生的时间窗口
- `compute_budget`: `<= 16 GPUh`
- `acceptance`: 形成 temporal-slice reevaluation report
- `why_blocked_now`: `time-slice evaluation schema unresolved`

### M59. `T68` 双设备确定性复跑批次

- `status`: `blocked`
- `priority_band`: `high for unattended reproducibility`
- `goal`: 在两台设备上做确定性复跑，确认长时任务不会因设备漂移而失真
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 dual-device determinism packet
- `why_blocked_now`: `cross-device deterministic replay not approved`

### M60. `T69` 最佳 3 rung ensemble 长跑批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 对最佳 3 个 rung 做更长时 ensemble 长跑，观察集成稳定性
- `compute_budget`: `<= 30 GPUh`
- `acceptance`: 形成 top-3 ensemble longrun report
- `why_blocked_now`: `ensemble longrun budget not approved`

### M61. `T70` 长时算力效率 frontier 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 在长时运行条件下找出成本/吞吐/收益的效率边界
- `compute_budget`: `<= 20 GPUh`
- `acceptance`: 形成 longrun compute-efficiency frontier
- `why_blocked_now`: `efficiency frontier is not yet a release criterion`

### M62. `T71` release-candidate burn-in 候选批次

- `status`: `blocked`
- `priority_band`: `highest once all prior evidence exists`
- `goal`: 构造真正面向 release-review 的 burn-in 候选包
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 release-candidate burn-in packet
- `why_blocked_now`: `burn-in candidate cannot be executed before approval`

### M63. `T72` 最终全波次证据总包批次

- `status`: `blocked`
- `priority_band`: `highest for handoff and audit`
- `goal`: 把所有 wave 的证据、失败模式和 gate 合成最终证据总包
- `compute_budget`: `<= 10 GPUh planning-side + future bounded checks`
- `acceptance`: 形成 all-wave evidence dossier
- `why_blocked_now`: `evidence dossier is not a queue-opening artifact`

### M64. `T73` Horizon H1 shortlist 批次

- `status`: `blocked`
- `priority_band`: `highest for near-horizon planning`
- `goal`: 从超长管理视图中压出真正近 horizon 的 H1 shortlist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Horizon H1 shortlist
- `why_blocked_now`: `H1 shortlist remains blocked until gate rewrite`

### M65. `T74` Horizon H2 shortlist 批次

- `status`: `blocked`
- `priority_band`: `high for medium-horizon planning`
- `goal`: 从超长管理视图中压出真正中期 horizon 的 H2 shortlist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Horizon H2 shortlist
- `why_blocked_now`: `H2 shortlist remains blocked until gate rewrite`

### M66. `T75` Horizon H3 governance pack 批次

- `status`: `blocked`
- `priority_band`: `high for far-horizon governance`
- `goal`: 给更远 horizon 准备一份 governance pack，避免后续重新拆分
- `compute_budget`: `<= 8 GPUh planning-side + future bounded checks`
- `acceptance`: 形成 Horizon H3 governance pack
- `why_blocked_now`: `governance pack is not a queue-opening artifact`

### M67. `T76` Priority tier A packet 批次

- `status`: `blocked`
- `priority_band`: `highest for priority readability`
- `goal`: 把 tier A 项统一压成一个最高优先级 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Priority Tier A packet
- `why_blocked_now`: `tier packet is planning only`

### M68. `T77` Priority tier B packet 批次

- `status`: `blocked`
- `priority_band`: `high`
- `goal`: 把 tier B 项统一压成一个中高优先级 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Priority Tier B packet
- `why_blocked_now`: `tier packet is planning only`

### M69. `T78` Priority tier C packet 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把 tier C 项统一压成一个中优先级 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Priority Tier C packet
- `why_blocked_now`: `tier packet is planning only`

### M70. `T79` Priority tier D packet 批次

- `status`: `blocked`
- `priority_band`: `medium`
- `goal`: 把 tier D 项统一压成一个低优先级但长程必要 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Priority Tier D packet
- `why_blocked_now`: `tier packet is planning only`

### M71. `T80` 全量 predecessor DAG 批次

- `status`: `blocked`
- `priority_band`: `high for orchestration clarity`
- `goal`: 把所有前驱依赖关系压成单一 DAG，供后续无人值守 orchestration 阅读
- `compute_budget`: `<= 8 GPUh planning-side + future consistency checks`
- `acceptance`: 形成 predecessor DAG
- `why_blocked_now`: `DAG is a planning map, not execution approval`

### M72. `T81` Review slot matrix 批次

- `status`: `blocked`
- `priority_band`: `high for review governance`
- `goal`: 预先定义未来 review slot 的分层占位和审批关系
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 review slot matrix
- `why_blocked_now`: `review slot matrix does not authorize execution`

### M73. `T82` Budget band matrix 批次

- `status`: `blocked`
- `priority_band`: `high for budget governance`
- `goal`: 把未来候选任务的预算档位压成固定带宽矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded checks`
- `acceptance`: 形成 budget band matrix
- `why_blocked_now`: `budget matrix is not an approved budget`

### M74. `T83` Artifact class taxonomy 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把所有预期 artifact 分类固定为单一 taxonomy
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 artifact class taxonomy
- `why_blocked_now`: `taxonomy is descriptive only, not release evidence`

### M75. `T84` Claim-to-artifact binding 批次

- `status`: `blocked`
- `priority_band`: `high for evidence discipline`
- `goal`: 把每类 headline/secondary/boundary claim 绑定到必须产出的 artifact
- `compute_budget`: `<= 6 GPUh planning-side + future bounded checks`
- `acceptance`: 形成 claim-to-artifact binding matrix
- `why_blocked_now`: `binding matrix is not admission approval`

### M76. `T85` Future smoke slot simulation 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 只在模拟层定义 future smoke slot 的占位和输入输出
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 future smoke slot simulation
- `why_blocked_now`: `smoke slot simulation is not a real smoke release`

### M77. `T86` Future preview slot simulation 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 只在模拟层定义 future preview slot 的占位和 gate
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 future preview slot simulation
- `why_blocked_now`: `preview slot simulation is not a real preview release`

### M78. `T87` Future decisive-rung slot simulation 批次

- `status`: `blocked`
- `priority_band`: `high for future decisive planning`
- `goal`: 只在模拟层定义 decisive-rung 的候选占位和约束
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 decisive-rung slot simulation
- `why_blocked_now`: `decisive-rung simulation is not decisive approval`

### M79. `T88` Future unattended-run slot simulation 批次

- `status`: `blocked`
- `priority_band`: `highest for unattended planning readability`
- `goal`: 只在模拟层定义 future unattended-run slot 的占位和守护规则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 unattended-run slot simulation
- `why_blocked_now`: `unattended-run simulation is not unattended execution approval`

### M80. `T89` Multi-wave packet index 批次

- `status`: `blocked`
- `priority_band`: `high for master navigation`
- `goal`: 给全部 wave/packet/horizon/tier 形成单一索引入口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-wave packet index
- `why_blocked_now`: `index is navigational only, not queue opening`

### M81. `T90` Future release decision tree 批次

- `status`: `blocked`
- `priority_band`: `highest for future governance readability`
- `goal`: 把未来可能的 go/no-go/review 分支压成单一决策树
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 future release decision tree
- `why_blocked_now`: `decision tree is a governance map, not a release decision`

### M82. `T91` Future long-run heartbeat simulation 批次

- `status`: `blocked`
- `priority_band`: `high for unattended orchestration`
- `goal`: 只在模拟层定义长时任务的 heartbeat 监控节奏和异常触发
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 long-run heartbeat simulation
- `why_blocked_now`: `heartbeat simulation is not a real unattended run`

### M83. `T92` Future resumable-run simulation 批次

- `status`: `blocked`
- `priority_band`: `high for recovery planning`
- `goal`: 只在模拟层定义可恢复长时任务的 resume 状态机和断点语义
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 resumable-run simulation
- `why_blocked_now`: `resume simulation is not resume approval`

### M84. `T93` Future artifact escrow simulation 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 只在模拟层定义长时任务 artifact escrow 和中间产物保全策略
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 artifact escrow simulation
- `why_blocked_now`: `artifact escrow simulation is descriptive only`

### M85. `T94` Final horizon execution map 批次

- `status`: `blocked`
- `priority_band`: `high for final navigation`
- `goal`: 把 H1/H2/H3 以及更远 horizon 的执行阅读路径压成单一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final horizon execution map
- `why_blocked_now`: `execution map is not execution permission`

### M86. `T95` Final planning packet 批次

- `status`: `blocked`
- `priority_band`: `high for final handoff`
- `goal`: 把所有 planning 层内容压成一份最终 planning packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final planning packet
- `why_blocked_now`: `planning packet is not a released packet`

### M87. `T96` Final blocked-master-schedule 冻结批次

- `status`: `blocked`
- `priority_band`: `highest for freeze governance`
- `goal`: 把整个 blocked master schedule 固定成最终冻结态
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final blocked-master-schedule freeze
- `why_blocked_now`: `freeze artifact does not authorize any run`

### M88. `T97` 500-step 高频 checkpoint 长时重跑

- `status`: `blocked`
- `priority_band`: `high for checkpoint-density evidence`
- `goal`: 用更高 checkpoint 密度观察长时训练轨迹
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 high-frequency checkpoint longrun report
- `why_blocked_now`: `high-frequency checkpoint run not approved`

### M89. `T98` 300-step repeat-longrun 批次

- `status`: `blocked`
- `priority_band`: `high for repeatability`
- `goal`: 用重复长跑确认 300-step 级别的长期一致性
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 repeat-longrun packet
- `why_blocked_now`: `repeat-longrun budget not approved`

### M90. `T99` 3x device envelope burn-in 批次

- `status`: `blocked`
- `priority_band`: `high for device robustness`
- `goal`: 在 3 个设备包络上做 burn-in，确认设备稳定性
- `compute_budget`: `<= 30 GPUh`
- `acceptance`: 形成 3x-device burn-in report
- `why_blocked_now`: `multi-device burn-in not approved`

### M91. `T100` 3x environment reproducibility burn-in 批次

- `status`: `blocked`
- `priority_band`: `high for environment robustness`
- `goal`: 在 3 个环境上做 reproducibility burn-in，确认环境稳定性
- `compute_budget`: `<= 30 GPUh`
- `acceptance`: 形成 3x-environment reproducibility report
- `why_blocked_now`: `multi-environment burn-in not approved`

### M92. `T101` 超长 24h burn-in 候选批次

- `status`: `blocked`
- `priority_band`: `highest for unattended long-run stress`
- `goal`: 为未来真正 24h 级长时任务准备候选 burn-in 包
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 24h burn-in candidate packet
- `why_blocked_now`: `24h unattended run cannot be opened before approval`

### M93. `T102` 最佳 2 config head-to-head 长跑批次

- `status`: `blocked`
- `priority_band`: `high for decisive comparison`
- `goal`: 让两个最优候选配置做 head-to-head 长跑比较
- `compute_budget`: `<= 24 GPUh`
- `acceptance`: 形成 head-to-head longrun comparator
- `why_blocked_now`: `decisive comparison not approved`

### M94. `T103` 大样本质量 panel 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把质量面板扩展到更大样本量，降低采样噪声
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 large-sample quality panel
- `why_blocked_now`: `large-sample quality evaluation not approved`

### M95. `T104` 大样本攻击 panel 批次

- `status`: `blocked`
- `priority_band`: `medium-high`
- `goal`: 把攻击评估扩展到更大样本量，降低波动
- `compute_budget`: `<= 18 GPUh`
- `acceptance`: 形成 large-sample attack panel
- `why_blocked_now`: `large-sample attack evaluation not approved`

### M96. `T105` 5-seed x 2-config stability grid 批次

- `status`: `blocked`
- `priority_band`: `highest for statistical stability`
- `goal`: 用 5 个 seed 和 2 个候选配置构造真正的稳定性网格
- `compute_budget`: `<= 28 GPUh`
- `acceptance`: 形成 5-seed x 2-config stability grid
- `why_blocked_now`: `stability grid remains blocked before release-review`

### M97. `T106` 5-split x 2-config transfer grid 批次

- `status`: `blocked`
- `priority_band`: `high for transfer robustness`
- `goal`: 用 5 个 split 与 2 个候选配置构造 transfer grid，观察跨 split 迁移稳定性
- `compute_budget`: `<= 30 GPUh`
- `acceptance`: 形成 5-split x 2-config transfer grid
- `why_blocked_now`: `transfer-grid remains blocked before release-review`

### M98. `T107` 7-shadow x 2-device 扩容批次

- `status`: `blocked`
- `priority_band`: `high for scale robustness`
- `goal`: 结合更大 shadow 数量与双设备条件检查规模外推稳定性
- `compute_budget`: `<= 32 GPUh`
- `acceptance`: 形成 7-shadow x 2-device expansion packet
- `why_blocked_now`: `scale-up and device envelope not approved`

### M99. `T108` 质量-攻击-算力三维大表重跑

- `status`: `blocked`
- `priority_band`: `highest for synthesis`
- `goal`: 把质量、攻击、算力三维指标重新汇成一张大表
- `compute_budget`: `<= 22 GPUh`
- `acceptance`: 形成 3D master table rerun packet
- `why_blocked_now`: `master table rerun is not yet release-approved`

### M100. `T109` 48h future-slot simulation 批次

- `status`: `blocked`
- `priority_band`: `high for unattended horizon simulation`
- `goal`: 只在模拟层定义 48h 长任务 slot 的守护、预算和停止语义
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 48h future-slot simulation
- `why_blocked_now`: `future-slot simulation is not execution approval`

### M101. `T110` 72h future-slot simulation 批次

- `status`: `blocked`
- `priority_band`: `high for unattended horizon simulation`
- `goal`: 只在模拟层定义 72h 长任务 slot 的守护、预算和停止语义
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 72h future-slot simulation
- `why_blocked_now`: `future-slot simulation is not execution approval`

### M102. `T111` 96h future-slot simulation 批次

- `status`: `blocked`
- `priority_band`: `high for unattended horizon simulation`
- `goal`: 只在模拟层定义 96h 长任务 slot 的守护、预算和停止语义
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 96h future-slot simulation
- `why_blocked_now`: `future-slot simulation is not execution approval`

### M103. `T112` final long-horizon slot cap 批次

- `status`: `blocked`
- `priority_band`: `highest for guardrail clarity`
- `goal`: 固定未来长时任务在 horizon 维度上的最高 slot cap
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final long-horizon slot cap
- `why_blocked_now`: `slot cap is a guardrail, not an opening`

### M104. `T113` final future shortlist lock 批次

- `status`: `blocked`
- `priority_band`: `highest for shortlist freeze`
- `goal`: 把未来 shortlist 最终冻结，避免后续持续漂移
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final future shortlist lock
- `why_blocked_now`: `shortlist lock remains blocked and non-executable`

### M105. `T114` final blocked candidate ladder 批次

- `status`: `blocked`
- `priority_band`: `high for candidate ordering`
- `goal`: 把 blocked candidate 的层级 ladder 压成最终版本
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final blocked candidate ladder
- `why_blocked_now`: `candidate ladder is ordering only`

### M106. `T115` final release-readiness anti-check 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed governance`
- `goal`: 定义未来 release-readiness 的反向审查面，确保不能误放行
- `compute_budget`: `<= 6 GPUh planning-side + future bounded checks`
- `acceptance`: 形成 release-readiness anti-check packet
- `why_blocked_now`: `anti-check packet is not release approval`

### M107. `T116` final handoff resume pack 批次

- `status`: `blocked`
- `priority_band`: `high for handoff`
- `goal`: 给未来接手者压出一个最终 handoff/resume 总包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final handoff resume pack
- `why_blocked_now`: `handoff pack is not an execution ticket`

### M108. `T117` final no-release proof 批次

- `status`: `blocked`
- `priority_band`: `highest for governance proof`
- `goal`: 把为什么当前仍应 no-release 的证明链固定下来
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final no-release proof
- `why_blocked_now`: `proof records why release is denied; it does not reopen gate`

### M109. `T118` final all-wave sync packet 批次

- `status`: `blocked`
- `priority_band`: `high for documentation consistency`
- `goal`: 把各 wave 的口径、artifact、gate 同步成单一 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final all-wave sync packet
- `why_blocked_now`: `sync packet is documentary only`

### M110. `T119` final master-schedule audit 批次

- `status`: `blocked`
- `priority_band`: `high for auditability`
- `goal`: 对整个 master schedule 做最后一次一致性审计
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final master-schedule audit
- `why_blocked_now`: `audit is not schedule release`

### M111. `T120` final frozen long-horizon blocked handbook 批次

- `status`: `blocked`
- `priority_band`: `highest for final handbook handoff`
- `goal`: 把整个长时 blocked 体系压成最终 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen long-horizon blocked handbook
- `why_blocked_now`: `handbook is explanatory only`

### M112. `T121` Campaign A packet 批次

- `status`: `blocked`
- `priority_band`: `high for campaign-level navigation`
- `goal`: 把 Campaign A 对应的所有任务压成单一 campaign packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Campaign A packet
- `why_blocked_now`: `campaign packet is navigational only, not execution permission`

### M113. `T122` Campaign B packet 批次

- `status`: `blocked`
- `priority_band`: `high for campaign-level navigation`
- `goal`: 把 Campaign B 对应任务压成单一 campaign packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Campaign B packet
- `why_blocked_now`: `campaign packet is navigational only`

### M114. `T123` Campaign C packet 批次

- `status`: `blocked`
- `priority_band`: `high for campaign-level navigation`
- `goal`: 把 Campaign C 对应任务压成单一 campaign packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Campaign C packet
- `why_blocked_now`: `campaign packet is navigational only`

### M115. `T124` Campaign D frozen handoff packet 批次

- `status`: `blocked`
- `priority_band`: `high for frozen handoff`
- `goal`: 把 Campaign D 压成单一 frozen handoff packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 Campaign D frozen handoff packet
- `why_blocked_now`: `handoff packet is not execution release`

### M116. `T125` H1-H2 bridge packet 批次

- `status`: `blocked`
- `priority_band`: `high for horizon bridging`
- `goal`: 压缩 H1 到 H2 的过渡条件、交接点和依赖
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 H1-H2 bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M117. `T126` H2-H3 bridge packet 批次

- `status`: `blocked`
- `priority_band`: `high for horizon bridging`
- `goal`: 压缩 H2 到 H3 的过渡条件、交接点和依赖
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 H2-H3 bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M118. `T127` H3-H4 bridge packet 批次

- `status`: `blocked`
- `priority_band`: `medium-high for horizon bridging`
- `goal`: 压缩 H3 到 H4 的过渡条件、交接点和依赖
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 H3-H4 bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M119. `T128` H4-H5 bridge packet 批次

- `status`: `blocked`
- `priority_band`: `medium-high for horizon bridging`
- `goal`: 压缩 H4 到 H5 的过渡条件、交接点和依赖
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 H4-H5 bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M120. `T129` priority band P0-P3 packet 批次

- `status`: `blocked`
- `priority_band`: `highest for priority normalization`
- `goal`: 把未来所有任务压到统一的 P0-P3 优先带
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 priority band P0-P3 packet
- `why_blocked_now`: `priority normalization is not execution permission`

### M121. `T130` release slot escalation ladder 批次

- `status`: `blocked`
- `priority_band`: `highest for slot governance`
- `goal`: 定义从 `R0` 到更高 slot 的升级阶梯和失败回落条件
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 release slot escalation ladder
- `why_blocked_now`: `escalation ladder is not slot approval`

### M122. `T131` campaign-to-slot binding 批次

- `status`: `blocked`
- `priority_band`: `high for governance binding`
- `goal`: 把 campaign 层和 slot 层绑定成单一关系图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 campaign-to-slot binding matrix
- `why_blocked_now`: `binding matrix is governance only`

### M123. `T132` long-horizon governance audit packet 批次

- `status`: `blocked`
- `priority_band`: `high for auditability`
- `goal`: 对整个 long-horizon 治理体系做单轮审计收口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 long-horizon governance audit packet
- `why_blocked_now`: `audit packet is not queue opening`

### M124. `T133` frozen Q1 blocked plan 批次

- `status`: `blocked`
- `priority_band`: `medium-high for calendarized planning`
- `goal`: 把 Q1 blocked plan 冻结成季度级别阅读层
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen Q1 blocked plan
- `why_blocked_now`: `quarter plan is explanatory only`

### M125. `T134` frozen Q2 blocked plan 批次

- `status`: `blocked`
- `priority_band`: `medium-high for calendarized planning`
- `goal`: 把 Q2 blocked plan 冻结成季度级别阅读层
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen Q2 blocked plan
- `why_blocked_now`: `quarter plan is explanatory only`

### M126. `T135` frozen Q3 blocked plan 批次

- `status`: `blocked`
- `priority_band`: `medium-high for calendarized planning`
- `goal`: 把 Q3 blocked plan 冻结成季度级别阅读层
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen Q3 blocked plan
- `why_blocked_now`: `quarter plan is explanatory only`

### M127. `T136` frozen Q4 blocked plan 批次

- `status`: `blocked`
- `priority_band`: `medium-high for calendarized planning`
- `goal`: 把 Q4 blocked plan 冻结成季度级别阅读层
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen Q4 blocked plan
- `why_blocked_now`: `quarter plan is explanatory only`

### M128. `T137` yearly blocked roadmap packet 批次

- `status`: `blocked`
- `priority_band`: `highest for annual planning readability`
- `goal`: 把全年 blocked 规划压成单一 yearly roadmap packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 yearly blocked roadmap packet
- `why_blocked_now`: `yearly roadmap is not execution approval`

### M129. `T138` final campaign scoreboard 批次

- `status`: `blocked`
- `priority_band`: `high for campaign visibility`
- `goal`: 把各 campaign 的进度/状态/冻结度压成单一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final campaign scoreboard
- `why_blocked_now`: `scoreboard is reporting only, not execution permission`

### M130. `T139` final horizon scoreboard 批次

- `status`: `blocked`
- `priority_band`: `high for horizon visibility`
- `goal`: 把各 horizon 的状态、bridge、freeze 信息压成单一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final horizon scoreboard
- `why_blocked_now`: `scoreboard is reporting only, not execution permission`

### M131. `T140` final slot scoreboard 批次

- `status`: `blocked`
- `priority_band`: `high for slot visibility`
- `goal`: 把 slot、slot-cap、slot-simulation 状态压成单一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final slot scoreboard
- `why_blocked_now`: `scoreboard is reporting only, not slot release`

### M132. `T141` frozen predecessor graph handbook 批次

- `status`: `blocked`
- `priority_band`: `high for dependency handoff`
- `goal`: 把 predecessor graph 压成稳定 handbook，便于后续只读接手
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen predecessor graph handbook
- `why_blocked_now`: `handbook is explanatory only`

### M133. `T142` frozen future-review packet 批次

- `status`: `blocked`
- `priority_band`: `high for review readiness`
- `goal`: 把 future-review 相关材料压成冻结 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen future-review packet
- `why_blocked_now`: `future-review packet is not review approval`

### M134. `T143` frozen blocked execution atlas 批次

- `status`: `blocked`
- `priority_band`: `high for navigation`
- `goal`: 把 blocked execution 阅读路径压成 atlas，供后续浏览
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen blocked execution atlas
- `why_blocked_now`: `atlas is navigational only`

### M135. `T144` final frozen campaign-horizon-slot handbook 批次

- `status`: `blocked`
- `priority_band`: `highest for integrated handoff`
- `goal`: 把 campaign/horizon/slot 三层统一成最终 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen campaign-horizon-slot handbook
- `why_blocked_now`: `handbook is not an execution ticket`

### M136. `T145` annual campaign map 批次

- `status`: `blocked`
- `priority_band`: `medium-high for annual planning`
- `goal`: 把全年 campaign 层组织成单一 map
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual campaign map
- `why_blocked_now`: `annual map is explanatory only`

### M137. `T146` annual horizon matrix 批次

- `status`: `blocked`
- `priority_band`: `medium-high for annual planning`
- `goal`: 把全年 horizon 分布组织成单一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual horizon matrix
- `why_blocked_now`: `annual matrix is explanatory only`

### M138. `T147` annual slot governance packet 批次

- `status`: `blocked`
- `priority_band`: `medium-high for annual governance`
- `goal`: 把全年 slot 治理规则压成统一 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual slot governance packet
- `why_blocked_now`: `governance packet is not slot approval`

### M139. `T148` annual anti-drift packet 批次

- `status`: `blocked`
- `priority_band`: `high for governance hygiene`
- `goal`: 把全年 anti-drift 规则压成统一 packet，防止口径漂移
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual anti-drift packet
- `why_blocked_now`: `anti-drift packet is governance only`

### M140. `T149` annual artifact retention packet 批次

- `status`: `blocked`
- `priority_band`: `high for artifact retention`
- `goal`: 把全年 artifact 保留、清理、归档规则压成统一 packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual artifact retention packet
- `why_blocked_now`: `retention packet is not execution permission`

### M141. `T150` annual master shortlist 批次

- `status`: `blocked`
- `priority_band`: `high for annual prioritization`
- `goal`: 把全年最重要的 blocked 候选压成 annual master shortlist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual master shortlist
- `why_blocked_now`: `master shortlist remains blocked`

### M142. `T151` annual execution atlas 批次

- `status`: `blocked`
- `priority_band`: `high for annual navigation`
- `goal`: 把全年 blocked execution 阅读路径压成年度 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual execution atlas
- `why_blocked_now`: `annual atlas is navigational only`

### M143. `T152` annual no-go ceiling 批次

- `status`: `blocked`
- `priority_band`: `highest for annual fail-closed governance`
- `goal`: 明确全年尺度上哪些条件仍然必须保持 no-go ceiling
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual no-go ceiling charter
- `why_blocked_now`: `no-go ceiling prevents release; it does not authorize one`

### M144. `T153` annual go-bar simulation 批次

- `status`: `blocked`
- `priority_band`: `high for annual what-if analysis`
- `goal`: 只在模拟层定义全年尺度上什么条件才可能触发 future go-bar
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 annual go-bar simulation
- `why_blocked_now`: `go-bar simulation is hypothetical only`

### M145. `T154` annual campaign scoreboard 批次

- `status`: `blocked`
- `priority_band`: `high for annual reporting`
- `goal`: 把全年 campaign 状态压成单一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual campaign scoreboard
- `why_blocked_now`: `scoreboard is reporting only`

### M146. `T155` annual handoff packet 批次

- `status`: `blocked`
- `priority_band`: `high for annual handoff`
- `goal`: 给全年 blocked 规划产出统一 handoff packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual handoff packet
- `why_blocked_now`: `handoff packet is not execution permission`

### M147. `T156` annual frozen blocked book 批次

- `status`: `blocked`
- `priority_band`: `highest for annual freeze readability`
- `goal`: 把全年 blocked 体系压成单一 frozen book
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 annual frozen blocked book
- `why_blocked_now`: `frozen book is explanatory only`

### M148. `T157` multi-year campaign map 批次

- `status`: `blocked`
- `priority_band`: `high for multi-year planning`
- `goal`: 把多年度 campaign 层组织成单一 map
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year campaign map
- `why_blocked_now`: `multi-year map is navigational only`

### M149. `T158` multi-year horizon matrix 批次

- `status`: `blocked`
- `priority_band`: `high for multi-year planning`
- `goal`: 把多年度 horizon 分布压成单一 matrix
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year horizon matrix
- `why_blocked_now`: `horizon matrix is explanatory only`

### M150. `T159` multi-year slot governance 批次

- `status`: `blocked`
- `priority_band`: `high for multi-year governance`
- `goal`: 把多年度 slot 规则压成统一 governance packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year slot governance
- `why_blocked_now`: `governance packet is not slot approval`

### M151. `T160` multi-year anti-release proof 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed governance`
- `goal`: 固定多年度尺度上为什么仍不得 release 的证明链
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year anti-release proof
- `why_blocked_now`: `proof records denial; it does not reopen gate`

### M152. `T161` multi-year roadmap packet 批次

- `status`: `blocked`
- `priority_band`: `high for roadmap readability`
- `goal`: 把多年度 blocked 规划压成单一 roadmap packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year roadmap packet
- `why_blocked_now`: `roadmap is explanatory only`

### M153. `T162` multi-year shortlist lock 批次

- `status`: `blocked`
- `priority_band`: `highest for shortlist freeze`
- `goal`: 把多年度 shortlist 最终冻结，防止持续漂移
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year shortlist lock
- `why_blocked_now`: `shortlist lock remains blocked`

### M154. `T163` multi-year handoff atlas 批次

- `status`: `blocked`
- `priority_band`: `high for atlas handoff`
- `goal`: 把多年度接手路径压成单一 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year handoff atlas
- `why_blocked_now`: `atlas is navigational only`

### M155. `T164` multi-year scoreboards 批次

- `status`: `blocked`
- `priority_band`: `high for executive visibility`
- `goal`: 把多年度 campaign/horizon/slot 状态压成多块 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M156. `T165` multi-year predecessor handbook 批次

- `status`: `blocked`
- `priority_band`: `high for dependency handoff`
- `goal`: 把多年度 predecessor 关系压成 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 multi-year predecessor handbook
- `why_blocked_now`: `handbook is explanatory only`

### M157. `T166` frozen campaign-year-slot ladder 批次

- `status`: `blocked`
- `priority_band`: `high for hierarchy freeze`
- `goal`: 把 campaign-year-slot 三层的梯子关系冻结成固定 ladder
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen campaign-year-slot ladder
- `why_blocked_now`: `ladder is governance only`

### M158. `T167` frozen multi-year no-release charter 批次

- `status`: `blocked`
- `priority_band`: `highest for multi-year fail-closed governance`
- `goal`: 固定多年度尺度上的 no-release 宪章
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen multi-year no-release charter
- `why_blocked_now`: `charter denies release; it does not permit one`

### M159. `T168` final frozen multi-year blocked canon 批次

- `status`: `blocked`
- `priority_band`: `highest for multi-year canon`
- `goal`: 把多年度 blocked 体系压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen multi-year blocked canon
- `why_blocked_now`: `canon is archival/governance only`

### M160. `T169` portfolio campaign map 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio-level navigation`
- `goal`: 把 portfolio 级别 campaign 层压成统一 map
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio campaign map
- `why_blocked_now`: `portfolio map is navigational only`

### M161. `T170` portfolio horizon matrix 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio-level planning`
- `goal`: 把 portfolio 级 horizon 分布压成统一 matrix
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio horizon matrix
- `why_blocked_now`: `matrix is descriptive only`

### M162. `T171` portfolio slot governance 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio-level governance`
- `goal`: 把 portfolio 级 slot 规则压成统一治理包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio slot governance packet
- `why_blocked_now`: `governance packet is not slot approval`

### M163. `T172` portfolio anti-release charter 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio fail-closed governance`
- `goal`: 固定 portfolio 尺度上的 anti-release 宪章
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio anti-release charter
- `why_blocked_now`: `charter blocks release; it does not permit one`

### M164. `T173` portfolio roadmap packet 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio roadmap readability`
- `goal`: 把 portfolio 级 blocked 规划压成单一 roadmap packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio roadmap packet
- `why_blocked_now`: `roadmap is explanatory only`

### M165. `T174` portfolio shortlist lock 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio shortlist freeze`
- `goal`: 把 portfolio 级 shortlist 固定到只读冻结态
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio shortlist lock
- `why_blocked_now`: `shortlist lock remains blocked`

### M166. `T175` portfolio handoff atlas 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio handoff`
- `goal`: 把 portfolio 级别的接手阅读路径压成单一 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio handoff atlas
- `why_blocked_now`: `atlas is navigational only`

### M167. `T176` portfolio scoreboards 批次

- `status`: `blocked`
- `priority_band`: `high for executive visibility`
- `goal`: 把 portfolio 级 campaign/horizon/slot 状态压成 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M168. `T177` portfolio predecessor canon 批次

- `status`: `blocked`
- `priority_band`: `high for dependency canonization`
- `goal`: 把 portfolio 级 predecessor 关系冻结成 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio predecessor canon
- `why_blocked_now`: `canon is governance only`

### M169. `T178` portfolio annual-review simulation 批次

- `status`: `blocked`
- `priority_band`: `high for annual what-if governance`
- `goal`: 只在模拟层定义 portfolio 级 annual review 的触发条件与路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 portfolio annual-review simulation
- `why_blocked_now`: `simulation is hypothetical only`

### M170. `T179` portfolio no-release proof 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio fail-closed proof`
- `goal`: 固定 portfolio 尺度上为什么仍然不得 release 的证明链
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio no-release proof
- `why_blocked_now`: `proof records denial; it does not reopen gate`

### M171. `T180` portfolio frozen blocked book 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio handbook`
- `goal`: 把 portfolio 级 blocked 体系压成最终 frozen book
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio frozen blocked book
- `why_blocked_now`: `book is explanatory only`

### M172. `T181` canon index packet 批次

- `status`: `blocked`
- `priority_band`: `high for canon navigation`
- `goal`: 把所有 canon 层对象压成统一 index packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon index packet
- `why_blocked_now`: `index packet is navigational only`

### M173. `T182` canon glossary packet 批次

- `status`: `blocked`
- `priority_band`: `high for terminology freeze`
- `goal`: 把 canon 层术语压成单一 glossary packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon glossary packet
- `why_blocked_now`: `glossary is descriptive only`

### M174. `T183` canon owner matrix 批次

- `status`: `blocked`
- `priority_band`: `high for ownership clarity`
- `goal`: 把 canon 层 owner 关系压成统一 matrix
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon owner matrix
- `why_blocked_now`: `owner matrix is governance only`

### M175. `T184` canon approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for approval clarity`
- `goal`: 把 canon 层 approval 关系压成统一 matrix
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon approval matrix
- `why_blocked_now`: `approval matrix is governance only`

### M176. `T185` canon retention matrix 批次

- `status`: `blocked`
- `priority_band`: `high for retention clarity`
- `goal`: 把 canon 层 retention 规则压成统一 matrix
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon retention matrix
- `why_blocked_now`: `retention matrix is governance only`

### M177. `T186` canon scoreboard packet 批次

- `status`: `blocked`
- `priority_band`: `high for canon visibility`
- `goal`: 把 canon 层状态压成统一 scoreboard packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon scoreboard packet
- `why_blocked_now`: `scoreboard is reporting only`

### M178. `T187` canon route map 批次

- `status`: `blocked`
- `priority_band`: `high for canon navigation`
- `goal`: 把 canon 层阅读路径压成单一路线图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon route map
- `why_blocked_now`: `route map is navigational only`

### M179. `T188` canon anti-drift proof 批次

- `status`: `blocked`
- `priority_band`: `highest for canon hygiene`
- `goal`: 固定 canon 层为什么不会再口径漂移的证明链
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon anti-drift proof
- `why_blocked_now`: `proof is governance only`

### M180. `T189` canon resume bundle 批次

- `status`: `blocked`
- `priority_band`: `high for resumability`
- `goal`: 为 canon 层生成一个最终可续跑的 resume bundle
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon resume bundle
- `why_blocked_now`: `resume bundle is not execution approval`

### M181. `T190` canon freeze checklist 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze discipline`
- `goal`: 把 canon 冻结前必须满足的检查项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze checklist
- `why_blocked_now`: `checklist is governance only`

### M182. `T191` canon freeze rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze rehearsal`
- `goal`: 在只读模式下预演 canon 冻结后的阅读和接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze rehearsal packet
- `why_blocked_now`: `rehearsal is simulated only`

### M183. `T192` final frozen planning canon 批次

- `status`: `blocked`
- `priority_band`: `highest for canon closure`
- `goal`: 把 planning 层压成最终 frozen canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen planning canon
- `why_blocked_now`: `canon closure is archival, not executable`

### M184. `T193` portfolio audit map 批次

- `status`: `blocked`
- `priority_band`: `high for portfolio auditability`
- `goal`: 把 portfolio 层对象与审计关系压成单一 map
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio audit map
- `why_blocked_now`: `audit map is descriptive only`

### M185. `T194` portfolio evidence ledger 批次

- `status`: `blocked`
- `priority_band`: `high for evidence discipline`
- `goal`: 固定 portfolio 层的证据台账
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio evidence ledger
- `why_blocked_now`: `ledger is documentary only`

### M186. `T195` portfolio claim ledger 批次

- `status`: `blocked`
- `priority_band`: `high for claim discipline`
- `goal`: 固定 portfolio 层的 claim 台账
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio claim ledger
- `why_blocked_now`: `ledger is documentary only`

### M187. `T196` portfolio control matrix 批次

- `status`: `blocked`
- `priority_band`: `high for control clarity`
- `goal`: 把 portfolio 层 control 关系压成矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio control matrix
- `why_blocked_now`: `control matrix is governance only`

### M188. `T197` portfolio boundary pack 批次

- `status`: `blocked`
- `priority_band`: `high for boundary clarity`
- `goal`: 把 portfolio 层边界条件压成统一 boundary pack
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio boundary pack
- `why_blocked_now`: `boundary pack is explanatory only`

### M189. `T198` portfolio route-control atlas 批次

- `status`: `blocked`
- `priority_band`: `high for route-control navigation`
- `goal`: 把 route 与 control 两层合成 portfolio atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio route-control atlas
- `why_blocked_now`: `atlas is navigational only`

### M190. `T199` portfolio retention charter 批次

- `status`: `blocked`
- `priority_band`: `high for retention governance`
- `goal`: 固定 portfolio 层长期保留与清理规则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio retention charter
- `why_blocked_now`: `charter is governance only`

### M191. `T200` portfolio review charter 批次

- `status`: `blocked`
- `priority_band`: `high for review governance`
- `goal`: 固定 portfolio 层 review 进入与退出规则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio review charter
- `why_blocked_now`: `review charter is not a review approval`

### M192. `T201` portfolio freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio freeze governance`
- `goal`: 把 portfolio 层最终冻结条件压成单一 charter
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio freeze charter
- `why_blocked_now`: `freeze charter closes change, not opens execution`

### M193. `T202` portfolio audit checklist 批次

- `status`: `blocked`
- `priority_band`: `high for audit execution discipline`
- `goal`: 把 portfolio 层审计前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio audit checklist
- `why_blocked_now`: `checklist is governance only`

### M194. `T203` portfolio handoff doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for handoff doctrine`
- `goal`: 固定 portfolio 层接手与续跑原则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio handoff doctrine
- `why_blocked_now`: `doctrine is explanatory only`

### M195. `T204` final portfolio audit canon 批次

- `status`: `blocked`
- `priority_band`: `highest for portfolio audit closure`
- `goal`: 把 portfolio 审计体系压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final portfolio audit canon
- `why_blocked_now`: `canon is archival/governance only`

### M196. `T205` canon freeze enforcement map 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze enforcement clarity`
- `goal`: 把 canon freeze 的执行边界与执法点压成单一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze enforcement map
- `why_blocked_now`: `enforcement map is not an execution permit`

### M197. `T206` canon freeze SLA 批次

- `status`: `blocked`
- `priority_band`: `high for operational discipline`
- `goal`: 定义 canon freeze 后各类响应和同步 SLA
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze SLA
- `why_blocked_now`: `SLA is governance only`

### M198. `T207` canon freeze exception matrix 批次

- `status`: `blocked`
- `priority_band`: `high for exception handling`
- `goal`: 固定 freeze 状态下哪些异常可以进入例外流程
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze exception matrix
- `why_blocked_now`: `exception matrix is not an exception approval`

### M199. `T208` canon freeze escalation matrix 批次

- `status`: `blocked`
- `priority_band`: `high for escalation clarity`
- `goal`: 固定 freeze 状态下问题升级路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze escalation matrix
- `why_blocked_now`: `escalation matrix is governance only`

### M200. `T209` canon freeze monitoring spec 批次

- `status`: `blocked`
- `priority_band`: `high for monitoring design`
- `goal`: 给 freeze 状态设计监控规范和告警点
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze monitoring spec
- `why_blocked_now`: `monitoring spec is descriptive only`

### M201. `T210` canon freeze telemetry schema 批次

- `status`: `blocked`
- `priority_band`: `high for telemetry consistency`
- `goal`: 给 freeze 监控定义统一 telemetry schema
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze telemetry schema
- `why_blocked_now`: `telemetry schema is not active monitoring rollout`

### M202. `T211` canon freeze compliance pack 批次

- `status`: `blocked`
- `priority_band`: `high for compliance traceability`
- `goal`: 把 freeze 相关的 compliance 证据压成单一 pack
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze compliance pack
- `why_blocked_now`: `compliance pack is documentary only`

### M203. `T212` canon freeze dry-run 批次

- `status`: `blocked`
- `priority_band`: `high for rehearsal`
- `goal`: 在不改变任何真实队列的前提下模拟一次 freeze dry-run
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze dry-run note
- `why_blocked_now`: `dry-run is simulated only`

### M204. `T213` canon freeze verdict packet 批次

- `status`: `blocked`
- `priority_band`: `highest for final verdict clarity`
- `goal`: 形成 freeze 状态下的最终 verdict packet
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze verdict packet
- `why_blocked_now`: `verdict packet records denial, not approval`

### M205. `T214` canon freeze registry 批次

- `status`: `blocked`
- `priority_band`: `high for registry integrity`
- `goal`: 形成 freeze 对象和状态的统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze registry
- `why_blocked_now`: `registry is governance only`

### M206. `T215` canon freeze handbook 批次

- `status`: `blocked`
- `priority_band`: `high for handbook handoff`
- `goal`: 把 freeze 状态下的阅读和操作边界写成 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon freeze handbook
- `why_blocked_now`: `handbook is explanatory only`

### M207. `T216` final frozen planning canon v2 批次

- `status`: `blocked`
- `priority_band`: `highest for canon v2 closure`
- `goal`: 把 planning canon 升级为 v2 并固定最终结构
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen planning canon v2
- `why_blocked_now`: `v2 canon is archival/governance only`

### M208. `T217` portfolio canon audit map 批次

- `status`: `blocked`
- `priority_band`: `high for cross-layer auditability`
- `goal`: 将 portfolio 层与 canon 层的审计关系压成统一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon audit map
- `why_blocked_now`: `audit map is descriptive only`

### M209. `T218` portfolio canon registry 批次

- `status`: `blocked`
- `priority_band`: `high for canon registry integrity`
- `goal`: 把 portfolio-canon 层对象压成统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon registry
- `why_blocked_now`: `registry is governance only`

### M210. `T219` portfolio canon glossary 批次

- `status`: `blocked`
- `priority_band`: `high for terminology freeze`
- `goal`: 把 portfolio-canon 层术语压成统一 glossary
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon glossary
- `why_blocked_now`: `glossary is descriptive only`

### M211. `T220` portfolio canon owner-doctrine matrix 批次

- `status`: `blocked`
- `priority_band`: `high for ownership clarity`
- `goal`: 把 owner 与 doctrine 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 owner-doctrine matrix
- `why_blocked_now`: `matrix is governance only`

### M212. `T221` portfolio canon route-doctrine atlas 批次

- `status`: `blocked`
- `priority_band`: `high for route/doctrine navigation`
- `goal`: 把 route 与 doctrine 两层压成统一 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 route-doctrine atlas
- `why_blocked_now`: `atlas is navigational only`

### M213. `T222` portfolio canon scoreboards v2 批次

- `status`: `blocked`
- `priority_band`: `high for reporting clarity`
- `goal`: 把 canon 层 scoreboards 升级为 v2 统一视图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon scoreboards v2
- `why_blocked_now`: `scoreboards are reporting only`

### M214. `T223` portfolio canon no-release doctrine 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed doctrine`
- `goal`: 固定 portfolio-canon 尺度上的 no-release doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon no-release doctrine
- `why_blocked_now`: `doctrine denies release; it does not permit one`

### M215. `T224` portfolio canon quarterly-yearly bridge 批次

- `status`: `blocked`
- `priority_band`: `medium-high for calendar bridging`
- `goal`: 把季度与年度层的过渡关系压成桥接包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 quarterly-yearly bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M216. `T225` portfolio canon horizon-campaign bridge 批次

- `status`: `blocked`
- `priority_band`: `medium-high for hierarchy bridging`
- `goal`: 把 horizon 与 campaign 两层压成桥接包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 horizon-campaign bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M217. `T226` portfolio canon slot-campaign bridge 批次

- `status`: `blocked`
- `priority_band`: `medium-high for hierarchy bridging`
- `goal`: 把 slot 与 campaign 两层压成桥接包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 slot-campaign bridge packet
- `why_blocked_now`: `bridge packet is explanatory only`

### M218. `T227` portfolio canon exception doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for exception governance`
- `goal`: 固定 portfolio-canon 例外处理 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 portfolio canon exception doctrine
- `why_blocked_now`: `doctrine is governance only`

### M219. `T228` final portfolio canon audit charter 批次

- `status`: `blocked`
- `priority_band`: `highest for audit closure`
- `goal`: 把 portfolio-canon 审计体系压成最终 charter
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final portfolio canon audit charter
- `why_blocked_now`: `charter is governance only`

### M220. `T229` frozen operations doctrine draft 批次

- `status`: `blocked`
- `priority_band`: `high for operations-layer drafting`
- `goal`: 为 operations doctrine 产出第一版 draft
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen operations doctrine draft
- `why_blocked_now`: `draft is preparatory only`

### M221. `T230` frozen operations doctrine review 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine review`
- `goal`: 对 operations doctrine draft 做系统化 review
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine review
- `why_blocked_now`: `review is not approval`

### M222. `T231` frozen operations doctrine audit 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine audit`
- `goal`: 对 operations doctrine 做审计收口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine audit
- `why_blocked_now`: `audit is documentary only`

### M223. `T232` frozen operations doctrine scoreboard 批次

- `status`: `blocked`
- `priority_band`: `medium-high for doctrine visibility`
- `goal`: 把 operations doctrine 状态压成 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine scoreboard
- `why_blocked_now`: `scoreboard is reporting only`

### M224. `T233` frozen operations doctrine atlas 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine navigation`
- `goal`: 把 operations doctrine 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine atlas
- `why_blocked_now`: `atlas is navigational only`

### M225. `T234` frozen operations doctrine handoff 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine handoff`
- `goal`: 把 operations doctrine 的接手规则压成单一 handoff 包
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine handoff packet
- `why_blocked_now`: `handoff packet is explanatory only`

### M226. `T235` frozen operations doctrine freeze checklist 批次

- `status`: `blocked`
- `priority_band`: `highest for operations freeze discipline`
- `goal`: 把 operations doctrine 冻结前必须满足的检查项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine freeze checklist
- `why_blocked_now`: `checklist is governance only`

### M227. `T236` frozen operations doctrine freeze rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for operations freeze rehearsal`
- `goal`: 在只读模式下预演 operations doctrine 冻结后的阅读和接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine freeze rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M228. `T237` frozen operations doctrine verdict 批次

- `status`: `blocked`
- `priority_band`: `highest for verdict clarity`
- `goal`: 形成 operations doctrine 冻结状态下的最终 verdict
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine verdict
- `why_blocked_now`: `verdict records governance state, not execution approval`

### M229. `T238` frozen operations doctrine registry 批次

- `status`: `blocked`
- `priority_band`: `high for registry integrity`
- `goal`: 把 operations doctrine 对象和状态压成统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 operations doctrine registry
- `why_blocked_now`: `registry is governance only`

### M230. `T239` final frozen operations doctrine handbook 批次

- `status`: `blocked`
- `priority_band`: `highest for doctrine handbook`
- `goal`: 把 operations doctrine 压成最终 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen operations doctrine handbook
- `why_blocked_now`: `handbook is explanatory only`

### M231. `T240` final frozen operations doctrine canon 批次

- `status`: `blocked`
- `priority_band`: `highest for doctrine canon closure`
- `goal`: 把 operations doctrine 收口成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen operations doctrine canon
- `why_blocked_now`: `canon is archival/governance only`

### M232. `T241` canon integrity map 批次

- `status`: `blocked`
- `priority_band`: `high for integrity navigation`
- `goal`: 把 canon integrity 相关对象和路径压成单一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity map
- `why_blocked_now`: `map is descriptive only`

### M233. `T242` canon integrity ledger 批次

- `status`: `blocked`
- `priority_band`: `high for integrity traceability`
- `goal`: 把 canon integrity 相关记录压成统一 ledger
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity ledger
- `why_blocked_now`: `ledger is documentary only`

### M234. `T243` canon integrity glossary 批次

- `status`: `blocked`
- `priority_band`: `high for integrity terminology`
- `goal`: 把 canon integrity 术语压成统一 glossary
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity glossary
- `why_blocked_now`: `glossary is descriptive only`

### M235. `T244` canon integrity owner matrix 批次

- `status`: `blocked`
- `priority_band`: `high for integrity ownership`
- `goal`: 把 canon integrity owner 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity owner matrix
- `why_blocked_now`: `owner matrix is governance only`

### M236. `T245` canon integrity approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for integrity approval clarity`
- `goal`: 把 canon integrity approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity approval matrix
- `why_blocked_now`: `approval matrix is governance only`

### M237. `T246` canon integrity route map 批次

- `status`: `blocked`
- `priority_band`: `high for integrity route clarity`
- `goal`: 把 canon integrity 的 route 关系压成单一路线图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity route map
- `why_blocked_now`: `route map is navigational only`

### M238. `T247` canon integrity retention matrix 批次

- `status`: `blocked`
- `priority_band`: `high for integrity retention clarity`
- `goal`: 把 canon integrity retention 规则压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity retention matrix
- `why_blocked_now`: `retention matrix is governance only`

### M239. `T248` canon integrity scoreboard 批次

- `status`: `blocked`
- `priority_band`: `medium-high for integrity visibility`
- `goal`: 把 canon integrity 状态压成统一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity scoreboard
- `why_blocked_now`: `scoreboard is reporting only`

### M240. `T249` canon integrity doctrine 批次

- `status`: `blocked`
- `priority_band`: `highest for integrity doctrine closure`
- `goal`: 把 canon integrity 原则压成统一 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity doctrine
- `why_blocked_now`: `doctrine is governance only`

### M241. `T250` canon integrity audit checklist 批次

- `status`: `blocked`
- `priority_band`: `high for integrity audit discipline`
- `goal`: 把 canon integrity 审计前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity audit checklist
- `why_blocked_now`: `checklist is governance only`

### M242. `T251` canon integrity rehearsal 批次

- `status`: `blocked`
- `priority_band`: `high for integrity rehearsal`
- `goal`: 在只读模式下预演 canon integrity 路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 canon integrity rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M243. `T252` final canon integrity charter 批次

- `status`: `blocked`
- `priority_band`: `highest for integrity closure`
- `goal`: 把 canon integrity 体系压成最终 charter
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final canon integrity charter
- `why_blocked_now`: `charter is governance only`

### M244. `T253` blocked universe map 批次

- `status`: `blocked`
- `priority_band`: `high for universe navigation`
- `goal`: 把 blocked universe 层对象压成统一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe map
- `why_blocked_now`: `map is navigational only`

### M245. `T254` blocked universe ledger 批次

- `status`: `blocked`
- `priority_band`: `high for universe traceability`
- `goal`: 把 blocked universe 层记录压成统一 ledger
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe ledger
- `why_blocked_now`: `ledger is documentary only`

### M246. `T255` blocked universe doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for universe doctrine clarity`
- `goal`: 把 blocked universe 层原则压成统一 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe doctrine
- `why_blocked_now`: `doctrine is governance only`

### M247. `T256` blocked universe owner-approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for universe ownership clarity`
- `goal`: 把 blocked universe 的 owner/approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe owner-approval matrix
- `why_blocked_now`: `matrix is governance only`

### M248. `T257` blocked universe scoreboards 批次

- `status`: `blocked`
- `priority_band`: `medium-high for universe visibility`
- `goal`: 把 blocked universe 状态压成 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M249. `T258` blocked universe route atlas 批次

- `status`: `blocked`
- `priority_band`: `high for universe navigation`
- `goal`: 把 blocked universe 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe route atlas
- `why_blocked_now`: `atlas is navigational only`

### M250. `T259` blocked universe audit pack 批次

- `status`: `blocked`
- `priority_band`: `high for universe auditability`
- `goal`: 把 blocked universe 相关审计材料压成单一 pack
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe audit pack
- `why_blocked_now`: `audit pack is documentary only`

### M251. `T260` blocked universe handoff doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for universe handoff`
- `goal`: 固定 blocked universe 层接手原则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe handoff doctrine
- `why_blocked_now`: `doctrine is explanatory only`

### M252. `T261` blocked universe freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for universe freeze governance`
- `goal`: 把 blocked universe 层冻结条件压成 charter
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M253. `T262` blocked universe freeze registry 批次

- `status`: `blocked`
- `priority_band`: `high for freeze registry integrity`
- `goal`: 把 blocked universe freeze 对象压成 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe freeze registry
- `why_blocked_now`: `registry is governance only`

### M254. `T263` blocked universe freeze handbook 批次

- `status`: `blocked`
- `priority_band`: `high for handbook handoff`
- `goal`: 把 blocked universe freeze 状态压成 handbook
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 blocked universe freeze handbook
- `why_blocked_now`: `handbook is explanatory only`

### M255. `T264` final frozen blocked universe canon 批次

- `status`: `blocked`
- `priority_band`: `highest for universe canon closure`
- `goal`: 把 blocked universe 体系压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen blocked universe canon
- `why_blocked_now`: `canon is archival/governance only`

### M256. `T265` universe audit doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for audit doctrine closure`
- `goal`: 把 universe 层审计原则压成统一 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit doctrine
- `why_blocked_now`: `doctrine is governance only`

### M257. `T266` universe audit registry 批次

- `status`: `blocked`
- `priority_band`: `high for audit registry integrity`
- `goal`: 把 universe 审计对象压成统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit registry
- `why_blocked_now`: `registry is governance only`

### M258. `T267` universe audit glossary 批次

- `status`: `blocked`
- `priority_band`: `high for universe terminology`
- `goal`: 把 universe 审计术语压成统一 glossary
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit glossary
- `why_blocked_now`: `glossary is descriptive only`

### M259. `T268` universe audit owner matrix 批次

- `status`: `blocked`
- `priority_band`: `high for owner clarity`
- `goal`: 把 universe 审计 owner 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit owner matrix
- `why_blocked_now`: `owner matrix is governance only`

### M260. `T269` universe audit approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for approval clarity`
- `goal`: 把 universe 审计 approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit approval matrix
- `why_blocked_now`: `approval matrix is governance only`

### M261. `T270` universe route-doctrine atlas 批次

- `status`: `blocked`
- `priority_band`: `high for route/doctrine navigation`
- `goal`: 把 universe 层 route 与 doctrine 两层压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe route-doctrine atlas
- `why_blocked_now`: `atlas is navigational only`

### M262. `T271` universe doctrine scoreboard 批次

- `status`: `blocked`
- `priority_band`: `medium-high for doctrine visibility`
- `goal`: 把 universe doctrine 状态压成统一 scoreboard
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe doctrine scoreboard
- `why_blocked_now`: `scoreboard is reporting only`

### M263. `T272` universe no-release charter 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed governance`
- `goal`: 固定 universe 层的 no-release 宪章
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe no-release charter
- `why_blocked_now`: `charter denies release; it does not permit one`

### M264. `T273` universe audit checklist 批次

- `status`: `blocked`
- `priority_band`: `high for audit discipline`
- `goal`: 把 universe 审计前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe audit checklist
- `why_blocked_now`: `checklist is governance only`

### M265. `T274` universe handoff doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for handoff doctrine`
- `goal`: 固定 universe 层接手原则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe handoff doctrine
- `why_blocked_now`: `doctrine is explanatory only`

### M266. `T275` universe freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze governance`
- `goal`: 固定 universe 层冻结条件和边界
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M267. `T276` final universe audit canon 批次

- `status`: `blocked`
- `priority_band`: `highest for audit canon closure`
- `goal`: 把 universe 审计体系压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final universe audit canon
- `why_blocked_now`: `canon is archival/governance only`

### M268. `T277` frozen constitution draft 批次

- `status`: `blocked`
- `priority_band`: `high for constitution drafting`
- `goal`: 为最终冻结态生成一版 constitution draft
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution draft
- `why_blocked_now`: `draft is preparatory only`

### M269. `T278` frozen constitution review 批次

- `status`: `blocked`
- `priority_band`: `high for constitution review`
- `goal`: 对 constitution draft 做系统 review
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution review
- `why_blocked_now`: `review is not approval`

### M270. `T279` frozen constitution audit 批次

- `status`: `blocked`
- `priority_band`: `high for constitution audit`
- `goal`: 对 constitution 做审计收口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution audit
- `why_blocked_now`: `audit is documentary only`

### M271. `T280` frozen constitution scoreboards 批次

- `status`: `blocked`
- `priority_band`: `medium-high for constitution visibility`
- `goal`: 把 constitution 层状态压成 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M272. `T281` frozen constitution atlas 批次

- `status`: `blocked`
- `priority_band`: `high for constitution navigation`
- `goal`: 把 constitution 层阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution atlas
- `why_blocked_now`: `atlas is navigational only`

### M273. `T282` frozen constitution owner matrix 批次

- `status`: `blocked`
- `priority_band`: `high for constitution ownership clarity`
- `goal`: 把 constitution 层 owner 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution owner matrix
- `why_blocked_now`: `owner matrix is governance only`

### M274. `T283` frozen constitution approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for constitution approval clarity`
- `goal`: 把 constitution 层 approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution approval matrix
- `why_blocked_now`: `approval matrix is governance only`

### M275. `T284` frozen constitution handoff bundle 批次

- `status`: `blocked`
- `priority_band`: `high for constitution handoff`
- `goal`: 把 constitution 层接手材料压成统一 bundle
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution handoff bundle
- `why_blocked_now`: `handoff bundle is explanatory only`

### M276. `T285` frozen constitution checklist 批次

- `status`: `blocked`
- `priority_band`: `highest for constitution freeze discipline`
- `goal`: 把 constitution 冻结前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution checklist
- `why_blocked_now`: `checklist is governance only`

### M277. `T286` frozen constitution rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for constitution rehearsal`
- `goal`: 在只读模式下预演 constitution 冻结后的阅读与接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M278. `T287` frozen constitution verdict 批次

- `status`: `blocked`
- `priority_band`: `highest for verdict clarity`
- `goal`: 形成 constitution 冻结状态下的最终 verdict
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen constitution verdict
- `why_blocked_now`: `verdict records governance state, not execution approval`

### M279. `T288` final frozen planning constitution 批次

- `status`: `blocked`
- `priority_band`: `highest for constitution closure`
- `goal`: 把 planning 层压成最终 frozen constitution
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen planning constitution
- `why_blocked_now`: `constitution is archival/governance only`

### M280. `T289` universe governance map 批次

- `status`: `blocked`
- `priority_band`: `high for governance navigation`
- `goal`: 把 universe governance 层对象压成统一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance map
- `why_blocked_now`: `map is descriptive only`

### M281. `T290` universe governance ledger 批次

- `status`: `blocked`
- `priority_band`: `high for governance traceability`
- `goal`: 把 universe governance 记录压成统一 ledger
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance ledger
- `why_blocked_now`: `ledger is documentary only`

### M282. `T291` universe governance glossary 批次

- `status`: `blocked`
- `priority_band`: `high for governance terminology`
- `goal`: 把 universe governance 术语压成统一 glossary
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance glossary
- `why_blocked_now`: `glossary is descriptive only`

### M283. `T292` universe governance matrix 批次

- `status`: `blocked`
- `priority_band`: `high for governance matrix clarity`
- `goal`: 把 universe governance 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance matrix
- `why_blocked_now`: `matrix is governance only`

### M284. `T293` universe governance doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine closure`
- `goal`: 把 universe governance 原则压成统一 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance doctrine
- `why_blocked_now`: `doctrine is governance only`

### M285. `T294` universe governance audit checklist 批次

- `status`: `blocked`
- `priority_band`: `high for audit discipline`
- `goal`: 把 universe governance 审计前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance audit checklist
- `why_blocked_now`: `checklist is governance only`

### M286. `T295` universe governance route atlas 批次

- `status`: `blocked`
- `priority_band`: `high for route navigation`
- `goal`: 把 universe governance 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance route atlas
- `why_blocked_now`: `atlas is navigational only`

### M287. `T296` universe governance handoff bundle 批次

- `status`: `blocked`
- `priority_band`: `high for governance handoff`
- `goal`: 把 universe governance 接手材料压成 bundle
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance handoff bundle
- `why_blocked_now`: `bundle is explanatory only`

### M288. `T297` universe governance freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for governance freeze`
- `goal`: 固定 universe governance 冻结条件和边界
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M289. `T298` universe governance verdict 批次

- `status`: `blocked`
- `priority_band`: `highest for governance verdict clarity`
- `goal`: 形成 universe governance 层的最终 verdict
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance verdict
- `why_blocked_now`: `verdict records governance state, not execution approval`

### M290. `T299` universe governance registry 批次

- `status`: `blocked`
- `priority_band`: `high for governance registry integrity`
- `goal`: 把 universe governance 对象压成统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 universe governance registry
- `why_blocked_now`: `registry is governance only`

### M291. `T300` final universe governance canon 批次

- `status`: `blocked`
- `priority_band`: `highest for governance canon closure`
- `goal`: 把 universe governance 层压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final universe governance canon
- `why_blocked_now`: `canon is archival/governance only`

### M292. `T301` planning republic draft 批次

- `status`: `blocked`
- `priority_band`: `high for republic drafting`
- `goal`: 为 planning republic 产出第一版 draft
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic draft
- `why_blocked_now`: `draft is preparatory only`

### M293. `T302` planning republic review 批次

- `status`: `blocked`
- `priority_band`: `high for republic review`
- `goal`: 对 planning republic draft 做系统 review
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic review
- `why_blocked_now`: `review is not approval`

### M294. `T303` planning republic audit 批次

- `status`: `blocked`
- `priority_band`: `high for republic audit`
- `goal`: 对 planning republic 做审计收口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic audit
- `why_blocked_now`: `audit is documentary only`

### M295. `T304` planning republic atlas 批次

- `status`: `blocked`
- `priority_band`: `high for republic navigation`
- `goal`: 把 planning republic 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic atlas
- `why_blocked_now`: `atlas is navigational only`

### M296. `T305` planning republic owner-approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for ownership clarity`
- `goal`: 把 planning republic owner/approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic owner-approval matrix
- `why_blocked_now`: `matrix is governance only`

### M297. `T306` planning republic scoreboards 批次

- `status`: `blocked`
- `priority_band`: `medium-high for republic visibility`
- `goal`: 把 planning republic 状态压成 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M298. `T307` planning republic handoff doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for republic handoff`
- `goal`: 固定 planning republic 接手原则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic handoff doctrine
- `why_blocked_now`: `doctrine is explanatory only`

### M299. `T308` planning republic freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for republic freeze governance`
- `goal`: 固定 planning republic 冻结条件和边界
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M300. `T309` planning republic checklist 批次

- `status`: `blocked`
- `priority_band`: `highest for republic freeze discipline`
- `goal`: 把 planning republic 冻结前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic checklist
- `why_blocked_now`: `checklist is governance only`

### M301. `T310` planning republic rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for republic rehearsal`
- `goal`: 在只读模式下预演 planning republic 冻结后的阅读与接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M302. `T311` planning republic verdict 批次

- `status`: `blocked`
- `priority_band`: `highest for verdict clarity`
- `goal`: 形成 planning republic 冻结状态下的最终 verdict
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 planning republic verdict
- `why_blocked_now`: `verdict records governance state, not execution approval`

### M303. `T312` final frozen planning republic 批次

- `status`: `blocked`
- `priority_band`: `highest for republic closure`
- `goal`: 把 planning republic 压成最终 frozen republic
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen planning republic
- `why_blocked_now`: `republic is archival/governance only`

### M304. `T313` republic audit map 批次

- `status`: `blocked`
- `priority_band`: `high for republic auditability`
- `goal`: 把 republic 层审计对象与关系压成统一地图
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic audit map
- `why_blocked_now`: `audit map is descriptive only`

### M305. `T314` republic audit registry 批次

- `status`: `blocked`
- `priority_band`: `high for republic registry integrity`
- `goal`: 把 republic 审计对象压成统一 registry
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic audit registry
- `why_blocked_now`: `registry is governance only`

### M306. `T315` republic audit glossary 批次

- `status`: `blocked`
- `priority_band`: `high for republic terminology`
- `goal`: 把 republic 审计术语压成统一 glossary
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic audit glossary
- `why_blocked_now`: `glossary is descriptive only`

### M307. `T316` republic governance matrix 批次

- `status`: `blocked`
- `priority_band`: `high for governance clarity`
- `goal`: 把 republic 层 governance 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic governance matrix
- `why_blocked_now`: `matrix is governance only`

### M308. `T317` republic doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for doctrine closure`
- `goal`: 把 republic 层原则压成统一 doctrine
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic doctrine
- `why_blocked_now`: `doctrine is governance only`

### M309. `T318` republic route atlas 批次

- `status`: `blocked`
- `priority_band`: `high for republic navigation`
- `goal`: 把 republic 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic route atlas
- `why_blocked_now`: `atlas is navigational only`

### M310. `T319` republic scoreboards 批次

- `status`: `blocked`
- `priority_band`: `medium-high for republic visibility`
- `goal`: 把 republic 状态压成统一 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M311. `T320` republic no-release charter 批次

- `status`: `blocked`
- `priority_band`: `highest for fail-closed governance`
- `goal`: 固定 republic 层的 no-release 宪章
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic no-release charter
- `why_blocked_now`: `charter denies release; it does not permit one`

### M312. `T321` republic freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze governance`
- `goal`: 固定 republic 层冻结条件和边界
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M313. `T322` republic audit checklist 批次

- `status`: `blocked`
- `priority_band`: `high for audit discipline`
- `goal`: 把 republic 审计前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic audit checklist
- `why_blocked_now`: `checklist is governance only`

### M314. `T323` republic freeze rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for freeze rehearsal`
- `goal`: 在只读模式下预演 republic 冻结后的阅读与接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 republic freeze rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M315. `T324` final republic audit canon 批次

- `status`: `blocked`
- `priority_band`: `highest for audit canon closure`
- `goal`: 把 republic 审计体系压成最终 canon
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final republic audit canon
- `why_blocked_now`: `canon is archival/governance only`

### M316. `T325` frozen commonwealth draft 批次

- `status`: `blocked`
- `priority_band`: `high for commonwealth drafting`
- `goal`: 为 commonwealth 层生成一版 draft
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth draft
- `why_blocked_now`: `draft is preparatory only`

### M317. `T326` frozen commonwealth review 批次

- `status`: `blocked`
- `priority_band`: `high for commonwealth review`
- `goal`: 对 commonwealth draft 做系统 review
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth review
- `why_blocked_now`: `review is not approval`

### M318. `T327` frozen commonwealth audit 批次

- `status`: `blocked`
- `priority_band`: `high for commonwealth audit`
- `goal`: 对 commonwealth 层做审计收口
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth audit
- `why_blocked_now`: `audit is documentary only`

### M319. `T328` frozen commonwealth atlas 批次

- `status`: `blocked`
- `priority_band`: `high for commonwealth navigation`
- `goal`: 把 commonwealth 阅读路径压成 atlas
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth atlas
- `why_blocked_now`: `atlas is navigational only`

### M320. `T329` frozen commonwealth owner-approval matrix 批次

- `status`: `blocked`
- `priority_band`: `high for ownership clarity`
- `goal`: 把 commonwealth owner/approval 关系压成统一矩阵
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth owner-approval matrix
- `why_blocked_now`: `matrix is governance only`

### M321. `T330` frozen commonwealth scoreboards 批次

- `status`: `blocked`
- `priority_band`: `medium-high for commonwealth visibility`
- `goal`: 把 commonwealth 状态压成最终 scoreboards
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth scoreboards
- `why_blocked_now`: `scoreboards are reporting only`

### M322. `T331` frozen commonwealth handoff doctrine 批次

- `status`: `blocked`
- `priority_band`: `high for commonwealth handoff`
- `goal`: 固定 commonwealth 层接手与续跑原则
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth handoff doctrine
- `why_blocked_now`: `doctrine is explanatory only`

### M323. `T332` frozen commonwealth freeze charter 批次

- `status`: `blocked`
- `priority_band`: `highest for commonwealth freeze governance`
- `goal`: 固定 commonwealth 层冻结条件和边界
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth freeze charter
- `why_blocked_now`: `charter closes change, not opens execution`

### M324. `T333` frozen commonwealth checklist 批次

- `status`: `blocked`
- `priority_band`: `highest for commonwealth freeze discipline`
- `goal`: 把 commonwealth 冻结前置项压成 checklist
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth checklist
- `why_blocked_now`: `checklist is governance only`

### M325. `T334` frozen commonwealth rehearsal 批次

- `status`: `blocked`
- `priority_band`: `highest for commonwealth rehearsal`
- `goal`: 在只读模式下预演 commonwealth 冻结后的阅读与接手路径
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth rehearsal
- `why_blocked_now`: `rehearsal is simulated only`

### M326. `T335` frozen commonwealth verdict 批次

- `status`: `blocked`
- `priority_band`: `highest for verdict clarity`
- `goal`: 形成 commonwealth 冻结状态下的最终 verdict
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 frozen commonwealth verdict
- `why_blocked_now`: `verdict records governance state, not execution approval`

### M327. `T336` final frozen planning commonwealth 批次

- `status`: `blocked`
- `priority_band`: `highest for commonwealth closure`
- `goal`: 把 planning 层压成最终 frozen commonwealth
- `compute_budget`: `<= 6 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 final frozen planning commonwealth
- `why_blocked_now`: `commonwealth is archival/governance only`

### M328. longrun repeat-burnin ladder 批次

- `status`: `blocked`
- `priority_band`: `highest for future longrun closure`
- `predecessor_pack`: `T61 + T97 + T98 + T101`
- `goal`: 把 repeat longrun、high-frequency checkpoint 与 24h burn-in 候选压成一个持续时长阶梯
- `compute_budget`: `<= 36 GPUh`
- `acceptance`: 形成 repeat-burnin ladder packet
- `why_blocked_now`: `all constituent heavy runs remain blocked`

### M329. cross-device and cross-environment burn-in grid 批次

- `status`: `blocked`
- `priority_band`: `highest for unattended portability`
- `predecessor_pack`: `T44 + T45 + T68 + T99 + T100`
- `goal`: 把设备包络、环境可复现和 burn-in 级复跑压成统一 grid
- `compute_budget`: `<= 40 GPUh`
- `acceptance`: 形成 cross-device/environment burn-in grid
- `why_blocked_now`: `portability burn-in remains blocked before release-review`

### M330. unattended heartbeat and recovery ladder 批次

- `status`: `blocked`
- `priority_band`: `highest for unattended runtime design`
- `predecessor_pack`: `T46 + T50 + T91 + T92 + T93`
- `goal`: 把 heartbeat、resumable-run、artifact escrow 和 smoke 候选压成无人值守恢复梯子
- `compute_budget`: `<= 24 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 unattended heartbeat and recovery ladder
- `why_blocked_now`: `simulation and monitoring layers do not authorize execution`

### M331. duration-slot escalation ladder 批次

- `status`: `blocked`
- `priority_band`: `high for long-duration planning`
- `predecessor_pack`: `T109 + T110 + T111 + T112`
- `goal`: 把 48h/72h/96h future-slot 和 slot-cap 压成单一 duration ladder
- `compute_budget`: `<= 24 GPUh planning-side + future bounded simulations`
- `acceptance`: 形成 duration-slot escalation ladder
- `why_blocked_now`: `duration ladder is hypothetical only`

### M332. decisive head-to-head stability ladder 批次

- `status`: `blocked`
- `priority_band`: `highest for decisive comparator planning`
- `predecessor_pack`: `T102 + T105 + T106 + T108`
- `goal`: 把 head-to-head 长跑、stability grid、transfer grid 和 3D master table 压成 decisive comparator ladder
- `compute_budget`: `<= 40 GPUh`
- `acceptance`: 形成 decisive head-to-head stability ladder
- `why_blocked_now`: `constituent decisive runs remain blocked`

### M333. commonwealth watchdog packet 批次

- `status`: `blocked`
- `priority_band`: `high for higher-level unattended governance`
- `predecessor_pack`: `T209 + T210 + T212 + T331-T336`
- `goal`: 把 monitoring、telemetry、dry-run 和 commonwealth tail 收敛成 watchdog packet
- `compute_budget`: `<= 12 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 commonwealth watchdog packet
- `why_blocked_now`: `watchdog design is governance only`

### M334. commonwealth escalation tree 批次

- `status`: `blocked`
- `priority_band`: `high for escalation governance`
- `predecessor_pack`: `T208 + T213 + T321-T336`
- `goal`: 把 escalation、verdict 和 commonwealth freeze 层压成 escalation tree
- `compute_budget`: `<= 12 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 commonwealth escalation tree
- `why_blocked_now`: `escalation tree is governance only`

### M335. commonwealth checkpoint escrow policy 批次

- `status`: `blocked`
- `priority_band`: `high for artifact retention`
- `predecessor_pack`: `T93 + T149 + T215 + T333-T336`
- `goal`: 把 checkpoint escrow、artifact retention 与 commonwealth handoff 压成统一 policy
- `compute_budget`: `<= 12 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 commonwealth checkpoint escrow policy
- `why_blocked_now`: `retention policy is documentary only`

### M336. commonwealth resume-state ladder 批次

- `status`: `blocked`
- `priority_band`: `highest for resumability closure`
- `predecessor_pack`: `T92 + T116 + T189 + T331-T336`
- `goal`: 把 resumable-run、handoff resume 和 commonwealth tail 压成最终 resume-state ladder
- `compute_budget`: `<= 12 GPUh planning-side + future bounded audits`
- `acceptance`: 形成 commonwealth resume-state ladder
- `why_blocked_now`: `resume-state ladder is governance only`

### Managed Todolist Hard Rule

- `M01-M336` 只是一层管理视图，不改变对应 `Txx` 的 blocked 身份
- 这 336 项的排序表示“若未来进入 release-review 时的阅读顺序”，不表示当前执行许可
- 当前 round 允许维护这份 todolist，但不允许把它切换成 active GPU runbook

## Program Waves

### Wave 0. Comparator And Gate Hardening

- `W0-A`: canonical comparator 锁定
- `W0-B`: attack/quality/runtime 统一 schema
- `W0-C`: release-review packet 模板锁定

### Wave 1. Hyperparameter Frontier

- `T01-T06`
- 目标：锁定 `lambda / rank / epochs / optimizer` 的候选前沿

### Wave 2. Stability And Portability

- `T07-T14`
- 目标：证明当前收益不是偶然值，并确认 8GB/多 target portability

### Wave 3. Unified Evaluation And Threat Hardening

- `T15-T28`
- 目标：形成 quality / adaptive / multi-attack / cross-split / cross-target 的统一比较面

### Wave 4. Consolidation And Release-Candidate Dossier

- `T29-T40`
- 目标：只在前 3 个 wave 形成足够证据后，生成最终 release-review 候选总包

### Wave 5. Long-Horizon Stress And Reproducibility

- `T41-T50`
- 目标：验证真正长时运行、恢复、跨环境、跨资源包络下的稳定性

### Wave 6. Final Program Governance Pack

- `T51-T60`
- 目标：形成候选执行册、审查包、发布顺序和最终 no-go/go 决策面

### Wave 7. Decisive Stress Closure

- `T61-T72`
- 目标：围绕候选最佳 rung 做最终长时压力、跨设备、跨协议和 burn-in 级 blocked 候选重跑

### Wave 8. Master-Schedule Consolidation

- `T73-T84`
- 目标：把 72 项 blocked backlog 进一步压成可执行 horizon / shortlist / packet 系统

### Wave 9. Future Release Simulation

- `T85-T96`
- 目标：仅做 future release simulation，不触发真实 release

### Wave 10. Large-Scale Burn-In Candidates

- `T97-T108`
- 目标：只围绕未来候选最佳 rung 做更大规模、更长时长的 blocked burn-in 候选压力包

### Wave 11. Final Long-Horizon Stress And Closure

- `T109-T120`
- 目标：形成真正面向未来 release-review 的长期压力闭环，但当前仍保持 blocked

### Wave 12. Campaign-Level Consolidation

- `T121-T132`
- 目标：把 120 项 blocked backlog 压成 campaign 级执行包与长期治理包

### Wave 13. Frozen Multi-Turn Master Schedule

- `T133-T144`
- 目标：冻结可交接的长期 blocked master schedule，避免后续再重复扩表

### Wave 14. Annual Campaign Closure

- `T145-T156`
- 目标：把年度 blocked 规划压成 campaign-year 级执行闭环

### Wave 15. Multi-Year Frozen Governance

- `T157-T168`
- 目标：冻结跨年度 blocked governance / handoff / no-release 体系

### Wave 16. Portfolio-Level Governance

- `T169-T180`
- 目标：把多年期 blocked 规划压成 portfolio 级 governance / retention / audit 体系

### Wave 17. Final Frozen Planning Canon

- `T181-T192`
- 目标：冻结最终 planning canon，后续轮次不再需要继续扩表

### Wave 18. Portfolio Audit Closure

- `T193-T204`
- 目标：把 portfolio-level 规划压成可审计、可复核、可交接的长期 blocked 审计包

### Wave 19. Canon Freeze Enforcement

- `T205-T216`
- 目标：把最终 planning canon 转成不可漂移的冻结执行宪章

### Wave 20. Portfolio Canon Audit

- `T217-T228`
- 目标：把 canon-freeze 体系压成可长期审计的 portfolio canon audit 包

### Wave 21. Final Frozen Operations Doctrine

- `T229-T240`
- 目标：冻结最终 operations doctrine，后续轮次不再扩展 blocked planning 结构

### Wave 22. Canon Integrity Closure

- `T241-T252`
- 目标：把 canon/registry/ledger/handbook 压成最终一致性与完整性闭环

### Wave 23. Final Frozen Blocked Universe

- `T253-T264`
- 目标：冻结最终 blocked universe，后续轮次不再继续扩展 planning 结构

### Wave 24. Universe Audit And Doctrine Closure

- `T265-T276`
- 目标：把 blocked universe 进一步收口成可审计、可追责、可冻结的 doctrine 包

### Wave 25. Final Frozen Planning Constitution

- `T277-T288`
- 目标：冻结最终 planning constitution，后续轮次不再继续扩表

### Wave 26. Universe Governance Closure

- `T289-T300`
- 目标：把 blocked universe 与 planning constitution 收敛成统一 governance 闭环

### Wave 27. Final Frozen Planning Republic

- `T301-T312`
- 目标：冻结最终 planning republic，彻底终止继续扩展 blocked planning 结构

### Wave 28. Republic Audit And Constitution Closure

- `T313-T324`
- 目标：把 planning republic 进一步收口成可审计、可追责、可冻结的 republic 宪章包

### Wave 29. Final Frozen Planning Commonwealth

- `T325-T336`
- 目标：冻结最终 planning commonwealth，后续轮次不再继续扩展 blocked planning 结构

## Execution Tiers

- `Tier A`: comparator/gate hardening heavy tasks
- `Tier B`: frontier and ablation heavy tasks
- `Tier C`: stability/portability heavy tasks
- `Tier D`: consolidation/burn-in heavy tasks

## Horizons

- `H1`: 未来 1-2 周的 blocked preparation pack
- `H2`: 未来 2-6 周的 blocked decisive pack
- `H3`: 未来 6 周以上的 blocked governance/release-simulation pack
- `H4`: 未来 3 个月以上的 blocked burn-in / long-horizon stress pack
- `H5`: 未来 6 个月以上的 blocked campaign / frozen master-schedule pack
- `H6`: 未来 12 个月以上的 blocked multi-year governance pack
- `H7`: 未来多年期的 blocked portfolio / frozen planning canon pack
- `H8`: 未来超长期的 blocked audit / canon-freeze enforcement pack
- `H9`: 未来超长期的 blocked portfolio-canon / operations-doctrine pack
- `H10`: 未来长期冻结态的 blocked universe / integrity pack
- `H11`: 未来终态冻结的 blocked constitution / doctrine pack
- `H12`: 未来终局冻结的 blocked governance / republic pack
- `H13`: 未来终局冻结的 blocked commonwealth / constitutional pack

## Campaigns

- `Campaign A`: comparator-and-frontier hardening
- `Campaign B`: stability-portability-burn-in
- `Campaign C`: governance-and-slot simulation
- `Campaign D`: frozen blocked master-schedule handoff
- `Campaign E`: annual blocked execution governance
- `Campaign F`: frozen multi-year no-release governance
- `Campaign G`: portfolio-level blocked governance
- `Campaign H`: final frozen planning canon
- `Campaign I`: portfolio audit and enforcement
- `Campaign J`: frozen canon governance
- `Campaign K`: portfolio canon audit
- `Campaign L`: final frozen operations doctrine
- `Campaign M`: canon integrity governance
- `Campaign N`: frozen blocked universe
- `Campaign O`: universe audit and doctrine governance
- `Campaign P`: final frozen planning constitution
- `Campaign Q`: universe governance closure
- `Campaign R`: final frozen planning republic
- `Campaign S`: republic audit and constitution governance
- `Campaign T`: final frozen planning commonwealth

## Release Slots

- `R0`: planning-only slot
- `R1`: future smoke candidate slot
- `R2`: future preview candidate slot
- `R3`: future decisive-rung candidate slot

当前所有 T01-T336 都固定在：

- `release_slot = R0`
- `status = blocked`
- `gpu_release = none`

## Stop Classes

- `SC-1`: no-directional-gain
- `SC-2`: protocol-drift
- `SC-3`: artifact-incomplete
- `SC-4`: runtime-budget-overflow
- `SC-5`: quality-collapse

## Long-Horizon GPU Heavy Task Backlog

### T01. `lambda=0.05` 短程完整重跑

- `status`: `blocked`
- `hypothesis`: `lambda=0.05` 在短程训练下可能优于 `lambda=0.1`
- `asset requirement`: 当前 DDPM target checkpoint、member/nonmember 目录、LoRA rank4 配置、现有训练脚本
- `compute budget`: `1 decisive run`, `<= 4 GPUh`, `single GPU`, 若 30 分钟内无法稳定写 checkpoint 则停
- `stop conditions`: 无法完成 final；AUC 未优于 `0.3438` 且无质量增量；OOM
- `expected artifact`: `summary.json`, `evaluation.json`, `final/lora_weights.pt`, `config.json`
- `current_gate`: `blocked by release-review / not-requestable`

### T02. `lambda=0.05` 100 epochs 长训练完整重跑

- `status`: `blocked`
- `hypothesis`: 更低 `lambda` 的长训练可能保留隐私收益同时减轻过拟合
- `asset requirement`: 与 T01 相同，外加长训练日志与中间 checkpoint 保留策略
- `compute budget`: `1 long run`, `<= 12 GPUh`, `single GPU`, 每 100 step 持久化
- `stop conditions`: 过拟合提前出现；评估劣于 `ep=10`；checkpoint 连续失败
- `expected artifact`: 长训练曲线、最终评估、最优 step 选择记录
- `current_gate`: `blocked by release-review / not-requestable`

### T03. 更低 `lambda` 前沿扫描：`0.01 / 0.02 / 0.05`

- `status`: `blocked`
- `hypothesis`: 当前最优点可能还在 `lambda=0.1` 以下
- `asset requirement`: 单一 canonical training/eval protocol、3 个新配置目录
- `compute budget`: `3 runs`, each `<= 4 GPUh`, total `<= 12 GPUh`
- `stop conditions`: 前两项即出现训练不稳定或质量崩溃；指标无任何优于 `0.3438` 的方向
- `expected artifact`: 扫描表、单配置 eval JSON、最优点摘要
- `current_gate`: `blocked by release-review / comparator not locked`

### T04. `rank` 扫描：`1 / 2 / 4 / 8`

- `status`: `blocked`
- `hypothesis`: 更低或更高 rank 可能改善隐私-质量折中
- `asset requirement`: 可变 rank 的 LoRA 注入配置、统一 eval 脚本
- `compute budget`: `4 runs`, each `<= 4 GPUh`, total `<= 16 GPUh`
- `stop conditions`: 高 rank 显存不可控；低 rank 明显无效；结果与 `rank=4` 无增量
- `expected artifact`: rank frontier 表、显存与时长记录、最佳 rank 说明
- `current_gate`: `blocked by release-review / 8GB portability not locked`

### T05. `epochs` 前沿扫描：`5 / 10 / 15 / 20 / 40`

- `status`: `blocked`
- `hypothesis`: 当前最优可能来自更窄训练窗口而非更长训练
- `asset requirement`: 固定 `lambda/rank`，统一 logging 和 early-stop schema
- `compute budget`: `5 runs`, total `<= 20 GPUh`
- `stop conditions`: 指标在 `10` 之后持续恶化；质量明显塌陷；训练曲线不稳定
- `expected artifact`: epoch frontier 图、最佳早停点建议
- `current_gate`: `blocked by release-review / no canonical early-stop rule`

### T06. 学习率与优化器前沿扫描

- `status`: `blocked`
- `hypothesis`: 当前结果可能受优化器超参数影响而非方法本身上限
- `asset requirement`: 至少 3 组 optimizer/lr 组合、固定 seed
- `compute budget`: `3-4 runs`, total `<= 16 GPUh`
- `stop conditions`: 训练直接发散；结果只是在复制旧最优；日志缺失
- `expected artifact`: optimizer ablation 表、稳定性说明
- `current_gate`: `blocked by release-review / protocol delta not yet closed`

### T07. 多随机种子稳定性重跑

- `status`: `blocked`
- `hypothesis`: 当前最优配置不是 seed 偶然值
- `asset requirement`: 至少 5 个 seed、固定最佳配置、统一 eval
- `compute budget`: `5 runs`, each `<= 4 GPUh`, total `<= 20 GPUh`
- `stop conditions`: 方差过大；多数 seed 不复现方向性下降；任一 run 无 final artifact
- `expected artifact`: mean/std 汇总、seed-stability packet
- `current_gate`: `blocked by release-review / stability packet missing`

### T08. 更大评估集重复测量

- `status`: `blocked`
- `hypothesis`: 当前 `63/63` 评估规模可能不足以支撑 decision-grade comparability
- `asset requirement`: 更大 member/nonmember eval split、统一 GSA evaluator
- `compute budget`: `2-3 eval-heavy runs`, total `<= 10 GPUh`
- `stop conditions`: eval 集不可比；指标波动解释不清；输出 schema 不统一
- `expected artifact`: enlarged-eval summary、方差对比
- `current_gate`: `blocked by release-review / eval protocol not locked`

### T09. 代理攻击模型统一：logistic vs MLP

- `status`: `blocked`
- `hypothesis`: SMP-LoRA 论文语境更接近 MLP proxy，当前 logistic 可能低估或扭曲 comparability
- `asset requirement`: 统一 attack API、MLP proxy 实现、同数据切分
- `compute budget`: `2 training + 2 eval runs`, total `<= 14 GPUh`
- `stop conditions`: 代理攻击模型无法与当前协议对齐；结果不可比；没有统一 artifact schema
- `expected artifact`: proxy-attack comparator packet
- `current_gate`: `blocked by release-review / attack protocol mismatch unresolved`

### T10. 自适应 GSA 攻击评估

- `status`: `blocked`
- `hypothesis`: 当前非自适应收益在 adaptive threat 下仍可维持
- `asset requirement`: 自适应攻击配置、至少 3 个候选 checkpoint、统一评估脚本
- `compute budget`: `3 eval-heavy runs`, total `<= 18 GPUh`
- `stop conditions`: adaptive config 不可复现；结果只产生解释漂移；无统一比较基线
- `expected artifact`: adaptive attack report、best/worst-case packet
- `current_gate`: `blocked by release-review / adaptive protocol not locked`

### T11. 图像质量面板：`FID / IS / PRD`

- `status`: `blocked`
- `hypothesis`: 当前隐私收益可能伴随不可接受的质量退化，需要质量 sidecar 证明
- `asset requirement`: 生成样本管线、质量评估脚本、参考数据集缓存
- `compute budget`: `5 checkpoint evals`, total `<= 14 GPUh`
- `stop conditions`: 生成样本数量不足；质量脚本不可重复；结果无法对齐 checkpoint
- `expected artifact`: quality frontier panel、privacy-quality tradeoff summary
- `current_gate`: `blocked by release-review / quality contract missing`

### T12. `W-1 strong-v3` 与 `SMP-LoRA` 同协议统一比较

- `status`: `blocked`
- `hypothesis`: 在统一 threat/eval/quality 协议下，SMP-LoRA 与 `W-1` 可形成 decision-grade comparator
- `asset requirement`: 统一 attack panel、统一 quality panel、统一 output schema
- `compute budget`: `1 comparison batch`, `<= 16 GPUh`
- `stop conditions`: 协议无法统一；比较只会制造误导；没有 locked comparator definition
- `expected artifact`: `SMP-LoRA vs W-1` unified comparator packet
- `current_gate`: `blocked by release-review / canonical comparator missing`

### T13. 第二 target checkpoint 可迁移性验证

- `status`: `blocked`
- `hypothesis`: 当前收益不是单 target checkpoint 偶然现象
- `asset requirement`: 第二 target checkpoint、匹配的数据划分、统一训练与评估脚本
- `compute budget`: `2 full runs`, total `<= 16 GPUh`
- `stop conditions`: 第二 checkpoint 不兼容；结果完全翻转； artifact 不可对齐
- `expected artifact`: portability note、cross-target summary
- `current_gate`: `blocked by release-review / target portability not scoped`

### T14. 8GB 可迁移训练链与 batch/accum frontier

- `status`: `blocked`
- `hypothesis`: SMP-LoRA 可被约束到单卡 8GB 可运行的 portable rung
- `asset requirement`: batch size / grad accumulation schema、显存 telemetry、稳定 checkpoint 保存
- `compute budget`: `3-4 runs`, total `<= 14 GPUh`
- `stop conditions`: 8GB 无法稳定写出 final；训练吞吐不可接受；结果无方向性增量
- `expected artifact`: 8GB portability packet、runtime envelope table
- `current_gate`: `blocked by release-review / portable rung not proven`

### T15. 防御强度-质量-算力三维 frontier 汇总重跑

- `status`: `blocked`
- `hypothesis`: 通过少量关键 rung 可以形成可讲的三维 frontier，而不是零散单点
- `asset requirement`: 已锁定的最佳 3-5 个 checkpoint、统一 attack/quality/runtime schema
- `compute budget`: `1 batch packet`, `<= 20 GPUh`
- `stop conditions`: 前置 comparator/quality/runtime 契约未锁定；结果仍不够 decision-grade
- `expected artifact`: final frontier packet、release-review candidate summary
- `current_gate`: `blocked by release-review / depends on T01-T14 partial closure`

### T16. 多 target checkpoint 交叉验证批次

- `status`: `blocked`
- `hypothesis`: 当前隐私收益可能只对单一 target checkpoint 成立，需要跨 checkpoint 证明
- `asset requirement`: 至少 3 个 target checkpoint、统一成员划分、统一训练脚本
- `compute budget`: `3 full runs`, total `<= 24 GPUh`
- `stop conditions`: 任一 checkpoint 协议不一致；结果方向完全分裂；artifact 无法统一
- `expected artifact`: cross-target packet、mean/std 汇总
- `current_gate`: `blocked by release-review / cross-target scope not locked`

### T17. 成员/非成员比例敏感性扫描

- `status`: `blocked`
- `hypothesis`: 当前结果可能依赖现有 member/nonmember 平衡比例
- `asset requirement`: 多组 eval split、统一训练配置、统一 attack panel
- `compute budget`: `4 runs`, total `<= 18 GPUh`
- `stop conditions`: split 不可比；收益只来自数据比例伪差异；输出 schema 漂移
- `expected artifact`: ratio-sensitivity table、风险说明
- `current_gate`: `blocked by release-review / eval split contract not frozen`

### T18. 代理攻击容量扫描：`small / medium / large MLP`

- `status`: `blocked`
- `hypothesis`: SMP-LoRA 收益在不同 proxy 容量下是否稳定
- `asset requirement`: 三档 MLP proxy、统一数据切分、统一训练日志
- `compute budget`: `3 training + 3 eval`, total `<= 20 GPUh`
- `stop conditions`: proxy 不收敛；结果与 logistic/MLP 对齐失败；无统一可比口径
- `expected artifact`: proxy-capacity frontier packet
- `current_gate`: `blocked by release-review / attack model contract unresolved`

### T19. 中间 checkpoint 选优重评批次

- `status`: `blocked`
- `hypothesis`: 最优隐私点可能出现在中间 step，而不是 final checkpoint
- `asset requirement`: 高频 checkpoint 保存策略、统一中间评估脚本
- `compute budget`: `1 long run + 6 eval windows`, total `<= 14 GPUh`
- `stop conditions`: checkpoint 不完整；中间评估无方向性收益；存储超预算
- `expected artifact`: checkpoint-selection curve、best-step decision note
- `current_gate`: `blocked by release-review / storage and evaluation contract not locked`

### T20. 两阶段训练 curriculum 对照

- `status`: `blocked`
- `hypothesis`: 先短程稳态再长程微调，可能优于单段式长训练
- `asset requirement`: 两阶段训练调度器、阶段切换 checkpoint、统一 eval
- `compute budget`: `2 curriculum runs`, total `<= 18 GPUh`
- `stop conditions`: curriculum 只增加成本不增益；阶段切换不稳定；final 不可复现
- `expected artifact`: curriculum-vs-single-stage comparator
- `current_gate`: `blocked by release-review / training schedule contract not frozen`

### T21. 梯度累积与 microbatch frontier

- `status`: `blocked`
- `hypothesis`: 更细粒度的 microbatch/accum 组合可改善 8GB 可迁移性且不损伤隐私收益
- `asset requirement`: microbatch grid、显存遥测、稳定恢复机制
- `compute budget`: `4 runs`, total `<= 16 GPUh`
- `stop conditions`: 吞吐不可接受；数值不稳定；收益弱于现有 portable rung
- `expected artifact`: microbatch frontier table、8GB 运行包
- `current_gate`: `blocked by release-review / portability gate unresolved`

### T22. 权重衰减与正则化扫描

- `status`: `blocked`
- `hypothesis`: 额外正则项可能减少长训练过拟合
- `asset requirement`: 至少 4 组 regularization 配置、统一日志
- `compute budget`: `4 runs`, total `<= 16 GPUh`
- `stop conditions`: 训练退化；质量显著崩溃；结果无法解释为方法收益
- `expected artifact`: regularization ablation packet
- `current_gate`: `blocked by release-review / optimization contract unresolved`

### T23. 数据增强鲁棒性重跑

- `status`: `blocked`
- `hypothesis`: SMP-LoRA 收益在轻量增强下仍稳健，而非依赖固定数据管线
- `asset requirement`: augmentation pipeline、可重复 seed、统一 baseline 对照
- `compute budget`: `3 runs`, total `<= 15 GPUh`
- `stop conditions`: augmentation 改变协议语义；结果仅来自数据扰动；artifact 不可对齐
- `expected artifact`: augmentation-robustness report
- `current_gate`: `blocked by release-review / data pipeline contract not frozen`

### T24. 多攻击面联合评估批次

- `status`: `blocked`
- `hypothesis`: 当前 GSA 收益在 loss-based / gradient-based / proxy-based 联合攻击面下仍成立
- `asset requirement`: 多攻击器统一接口、统一 checkpoint 集、统一 schema
- `compute budget`: `1 multi-attack batch`, `<= 20 GPUh`
- `stop conditions`: 攻击器协议不一致；比较只制造新的解释歧义；结果无法统一摘要
- `expected artifact`: multi-attack evaluation packet
- `current_gate`: `blocked by release-review / attack panel unification missing`

### T25. 多 shadow 扩容重跑

- `status`: `blocked`
- `hypothesis`: 更大 shadow 数量能减少当前 comparability 的方差
- `asset requirement`: 更多 shadow 资产或等价近似、统一 GSA 训练/eval 管线
- `compute budget`: `1 expansion batch`, `<= 24 GPUh`
- `stop conditions`: shadow 资产不匹配；扩容只放大成本不提升证据质量；artifact 不完整
- `expected artifact`: shadow-expansion packet
- `current_gate`: `blocked by release-review / shadow asset scope unresolved`

### T26. 跨 split 可迁移性重跑

- `status`: `blocked`
- `hypothesis`: 当前结论不只绑定单一 split，而能跨 2-3 组 split 维持方向性
- `asset requirement`: 多 split 定义、可重复训练/eval 配置、统一 seed policy
- `compute budget`: `3 full runs`, total `<= 24 GPUh`
- `stop conditions`: split 协议不一致；收益翻转；跨 split 不能形成稳定摘要
- `expected artifact`: cross-split packet、variance analysis
- `current_gate`: `blocked by release-review / split contract unresolved`

### T27. 双目标 release-candidate consolidation 重跑

- `status`: `blocked`
- `hypothesis`: 经过前序筛选后，最佳 2 个 rung 值得做最终 release-candidate 级重跑
- `asset requirement`: 已锁定 best-2 configs、统一 attack/quality/runtime schema、完整 checkpoint 路径
- `compute budget`: `2 decisive runs`, total `<= 20 GPUh`
- `stop conditions`: best-2 未锁定；重跑无法复现；成本超预算
- `expected artifact`: release-candidate rerun packet
- `current_gate`: `blocked by release-review / depends on T01-T26 partial closure`

### T28. 最终大包汇总批次

- `status`: `blocked`
- `hypothesis`: 若前序关键 backlog 收口，可形成一份 release-review-ready 的长程 GPU 总包
- `asset requirement`: 前序关键 packet、统一摘要模板、locked headline claims
- `compute budget`: `1 final batch`, `<= 24 GPUh`
- `stop conditions`: headline 口径仍未锁定；packet 仍不够 decision-grade；风险无法收敛
- `expected artifact`: final long-horizon package、release-review dossier
- `current_gate`: `blocked by release-review / depends on T01-T27 partial closure`

### T29. LoRA layer placement 扫描

- `status`: `blocked`
- `hypothesis`: 不同 attention/value/query/key 注入位置会改变隐私收益和质量成本
- `asset requirement`: 可配置 layer placement、统一训练和评估脚本
- `compute budget`: `4 runs`, total `<= 18 GPUh`
- `stop conditions`: placement 不可迁移；收益不稳定；显存开销过高
- `expected artifact`: layer-placement ablation packet
- `current_gate`: `blocked by release-review / layer-placement contract unresolved`

### T30. LoRA 参数量 frontier

- `status`: `blocked`
- `hypothesis`: 当前参数量可能不是最佳隐私-算力点
- `asset requirement`: 多组参数量配置、统一 rank/placement 解释
- `compute budget`: `4 runs`, total `<= 18 GPUh`
- `stop conditions`: 参数量变化只带来成本上升；指标无增量；配置不可比
- `expected artifact`: parameter-budget frontier table
- `current_gate`: `blocked by release-review / parameterization contract unresolved`

### T31. Mixed precision / full precision 对照

- `status`: `blocked`
- `hypothesis`: 混合精度可能影响稳定性或隐私测量
- `asset requirement`: fp16/bf16/fp32 训练配置、统一日志和评估
- `compute budget`: `3 runs`, total `<= 15 GPUh`
- `stop conditions`: 数值不稳定；结果不一致但无法解释；资源消耗超预算
- `expected artifact`: precision comparator packet
- `current_gate`: `blocked by release-review / numeric stability contract not frozen`

### T32. 训练集规模扩展批次

- `status`: `blocked`
- `hypothesis`: SMP-LoRA 收益随训练集规模扩展后仍然可维持
- `asset requirement`: 更大训练样本集、可比 split、统一评估
- `compute budget`: `3 runs`, total `<= 24 GPUh`
- `stop conditions`: 数据规模改变协议语义；收益完全消失；artifact 不完整
- `expected artifact`: dataset-scale sensitivity packet
- `current_gate`: `blocked by release-review / dataset scale scope unresolved`

### T33. 训练步数与评估步数联合网格

- `status`: `blocked`
- `hypothesis`: 最佳隐私点由训练步数和评估步数组合共同决定
- `asset requirement`: train/eval grid 配置、统一 scheduler 和日志
- `compute budget`: `6 runs`, total `<= 22 GPUh`
- `stop conditions`: 网格爆炸超预算；趋势无法解释；重复旧结论
- `expected artifact`: train-eval grid heatmap
- `current_gate`: `blocked by release-review / step-grid contract unresolved`

### T34. 多次重启恢复鲁棒性测试

- `status`: `blocked`
- `hypothesis`: 真正无人值守长任务必须能经受中断/恢复而不破坏最终结论
- `asset requirement`: checkpoint resume 能力、恢复日志、统一 eval
- `compute budget`: `2 interrupted long runs`, total `<= 16 GPUh`
- `stop conditions`: 无法可靠恢复；恢复后指标漂移过大；checkpoint 损坏
- `expected artifact`: interruption-resume robustness packet
- `current_gate`: `blocked by release-review / resume path not implemented`

### T35. 多 run ensemble 评估

- `status`: `blocked`
- `hypothesis`: 多 run ensemble 能降低评估方差并形成更稳 comparability
- `asset requirement`: 至少 3 条已完成 run、统一 ensemble 规则
- `compute budget`: `3 training + 1 eval batch`, total `<= 20 GPUh`
- `stop conditions`: ensemble 只掩盖单 run 波动；无统一摘要规则；成本过高
- `expected artifact`: ensemble evaluation packet
- `current_gate`: `blocked by release-review / ensemble contract unresolved`

### T36. LoRA merge/unmerge 可复现性批次

- `status`: `blocked`
- `hypothesis`: merge/unmerge 过程可能改变最终评估，需锁定 reproducibility
- `asset requirement`: merge/unmerge 工具链、统一导出格式
- `compute budget`: `2 runs + export batch`, total `<= 12 GPUh`
- `stop conditions`: merge 后结果漂移；导出不可复现；权重不一致
- `expected artifact`: merge reproducibility packet
- `current_gate`: `blocked by release-review / export contract unresolved`

### T37. 跨攻击器阈值曲线重跑

- `status`: `blocked`
- `hypothesis`: 当前 AUC 改善在不同 FPR 阈值面上仍稳定，而非单点偶然
- `asset requirement`: 多阈值评估脚本、统一 ROC/TPR 摘要
- `compute budget`: `1 eval-heavy batch`, `<= 10 GPUh`
- `stop conditions`: 阈值面解释不清；曲线与 headline 冲突；schema 不统一
- `expected artifact`: threshold-surface packet
- `current_gate`: `blocked by release-review / threshold contract unresolved`

### T38. 多质量指标联合 panel

- `status`: `blocked`
- `hypothesis`: 单一 FID 不足以代表质量，需要联合 `FID / IS / precision / recall`
- `asset requirement`: 多质量评估器、参考缓存、统一输出 schema
- `compute budget`: `1 quality batch`, `<= 12 GPUh`
- `stop conditions`: 指标冲突无法解释；缓存或脚本不稳定；只重复已有质量结论
- `expected artifact`: quality multi-panel packet
- `current_gate`: `blocked by release-review / quality panel not frozen`

### T39. 最佳 5 个 rung 的全量复核批次

- `status`: `blocked`
- `hypothesis`: 只有经过全量复核，best-5 rung 才能进入 release-review shortlist
- `asset requirement`: best-5 rung 已锁定、统一复核脚本、全量 artifact 清单
- `compute budget`: `5 eval/training mixed runs`, total `<= 24 GPUh`
- `stop conditions`: best-5 尚未锁定；复核结果分裂；artifact 缺失
- `expected artifact`: shortlist verification packet
- `current_gate`: `blocked by release-review / depends on T01-T38 partial closure`

### T40. 最终长期无人值守候选运行册

- `status`: `blocked`
- `hypothesis`: 若前序关键 backlog 收口，可形成一份长期无人值守候选运行册，但仍须单独审批
- `asset requirement`: waves 0-4 的已锁定 packet、统一执行模板、风险清单
- `compute budget`: `planning batch only`, future release candidate envelope `<= 24 GPUh per decisive rung`
- `stop conditions`: 任何 headline / comparator / gate 未锁定；任何条目仍缺核心 artifact
- `expected artifact`: unattended-run candidate manual + release-review dossier
- `current_gate`: `blocked by release-review / depends on T01-T39 partial closure`

### T41. 超长训练 200 epochs 过拟合边界批次

- `status`: `blocked`
- `hypothesis`: 极长训练可以帮助确认当前过拟合边界的真实拐点
- `asset requirement`: 最佳候选 config、稳定 checkpoint 保存、高频评估点
- `compute budget`: `1 ultra-long run`, `<= 24 GPUh`
- `stop conditions`: 指标连续恶化；质量崩塌；checkpoint 丢失
- `expected artifact`: overfit-boundary curve、stop-decision note
- `current_gate`: `blocked by release-review / long-run justification unresolved`

### T42. 多阶段 early-stop 策略批次

- `status`: `blocked`
- `hypothesis`: 预定义 early-stop 规则可降低无人值守长跑浪费
- `asset requirement`: 多阶段 stop 规则、实时 metrics logging、统一 eval hooks
- `compute budget`: `3 long runs`, total `<= 20 GPUh`
- `stop conditions`: 规则无法稳定触发；收益不优于固定 epochs；日志不完整
- `expected artifact`: early-stop policy packet
- `current_gate`: `blocked by release-review / stop-policy contract unresolved`

### T43. 不同 sample budget 下的质量-隐私联合批次

- `status`: `blocked`
- `hypothesis`: 当前结论可能对生成样本预算高度敏感
- `asset requirement`: 多 sample budget 评估器、统一 quality and attack schema
- `compute budget`: `4 eval-heavy batches`, total `<= 16 GPUh`
- `stop conditions`: sample budget 改变协议语义；结果不可比较；输出不统一
- `expected artifact`: sample-budget sensitivity packet
- `current_gate`: `blocked by release-review / sample budget contract unresolved`

### T44. 跨 GPU 设备包络验证

- `status`: `blocked`
- `hypothesis`: 不同 GPU 设备类型可能影响稳定性与吞吐，不应只绑定单设备
- `asset requirement`: 至少 2 类 GPU 设备、统一 runtime telemetry、统一 config
- `compute budget`: `2-3 runs`, total `<= 20 GPUh`
- `stop conditions`: 设备不可得；跨设备结果分裂； telemetry 缺失
- `expected artifact`: device-envelope packet
- `current_gate`: `blocked by release-review / hardware portability unresolved`

### T45. 跨环境可复现性批次

- `status`: `blocked`
- `hypothesis`: 在无本地 scheduler 的环境下，长程任务仍可复现
- `asset requirement`: repo-relative config、portable launch path、环境锁定文件
- `compute budget`: `2 reproducibility runs`, total `<= 18 GPUh`
- `stop conditions`: 环境差异导致结果失真；依赖不可锁定；路径不可迁移
- `expected artifact`: environment reproducibility packet
- `current_gate`: `blocked by release-review / portability contract unresolved`

### T46. 训练日志与监控信号对齐批次

- `status`: `blocked`
- `hypothesis`: 无人值守长跑需要更强的训练日志和监控信号对齐
- `asset requirement`: 统一 log schema、metrics sink、异常状态码
- `compute budget`: `2 monitored long runs`, total `<= 12 GPUh`
- `stop conditions`: 监控信号不稳定；日志丢失；无法支持后验审查
- `expected artifact`: monitoring-and-log packet
- `current_gate`: `blocked by release-review / observability contract unresolved`

### T47. 梯度统计面板批次

- `status`: `blocked`
- `hypothesis`: 梯度统计可帮助解释 SMP-LoRA 的隐私收益与失效模式
- `asset requirement`: 梯度统计导出、统一可视化、样本绑定
- `compute budget`: `2 runs + export batch`, total `<= 14 GPUh`
- `stop conditions`: 统计不可稳定导出；解释层与 headline 冲突；成本过高
- `expected artifact`: gradient-statistics packet
- `current_gate`: `blocked by release-review / mechanism packet unresolved`

### T48. 权重漂移轨迹批次

- `status`: `blocked`
- `hypothesis`: LoRA 权重漂移轨迹可能提供更稳的早停或异常检测信号
- `asset requirement`: 权重差异导出、统一 checkpoint 频率、轨迹摘要工具
- `compute budget`: `2 long runs`, total `<= 14 GPUh`
- `stop conditions`: 轨迹信号噪声过大；无法映射到结果；存储超预算
- `expected artifact`: weight-drift trajectory packet
- `current_gate`: `blocked by release-review / mechanism packet unresolved`

### T49. 失败模式专用重跑批次

- `status`: `blocked`
- `hypothesis`: 单独围绕已知失败模式重跑，能更快界定 no-go 边界
- `asset requirement`: 失败模式 catalog、对应配置、统一 debug schema
- `compute budget`: `3 targeted runs`, total `<= 15 GPUh`
- `stop conditions`: 失败模式不可稳定重现；只放大噪声；结论无收敛
- `expected artifact`: failure-mode packet
- `current_gate`: `blocked by release-review / failure catalog not frozen`

### T50. 长时无人值守 smoke 候选批次

- `status`: `blocked`
- `hypothesis`: 在极小 release envelope 下可以先定义“无人值守 smoke”的未来候选面
- `asset requirement`: resume path、monitoring、artifact completeness checks
- `compute budget`: `1 candidate smoke`, future envelope `<= 8 GPUh`
- `stop conditions`: 任何基础条件缺失；不能自动恢复；artifact 不完整
- `expected artifact`: unattended-smoke candidate packet
- `current_gate`: `blocked by release-review / unattended smoke not approved`

### T51. Wave 间依赖一致性审查批次

- `status`: `blocked`
- `hypothesis`: 各 wave 之间的依赖若不先收敛，会导致长程 program 失真
- `asset requirement`: wave dependency map、per-task gate matrix
- `compute budget`: `planning/eval batch`, future `<= 4 GPUh`
- `stop conditions`: 依赖无法压缩；任务间矛盾未解；无统一 matrix
- `expected artifact`: dependency consistency packet
- `current_gate`: `blocked by release-review / governance matrix unresolved`

### T52. 候选 shortlist 生成批次

- `status`: `blocked`
- `hypothesis`: 60 项 backlog 中只有少数条目值得进入 future shortlist
- `asset requirement`: scorecard、priority rubric、risk rubric
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: rubric 不稳定；shortlist 无法解释；输入证据不足
- `expected artifact`: shortlist packet
- `current_gate`: `blocked by release-review / shortlist rubric unresolved`

### T53. release-review 审查面模拟批次

- `status`: `blocked`
- `hypothesis`: 在真正申请 GPU 之前，先模拟 release-review 可提前暴露 drift
- `asset requirement`: 审查模板、headline claims、artifact checklist
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: headline 仍未锁定；artifact checklist 缺口过大；无法形成单页结论
- `expected artifact`: mock release-review packet
- `current_gate`: `blocked by release-review / review template unresolved`

### T54. final claim hierarchy 收口批次

- `status`: `blocked`
- `hypothesis`: 必须先明确 strongest claim / allowed claim / disallowed claim 层级，才能长期运行
- `asset requirement`: claim matrix、当前证据映射、风险审查
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: claim hierarchy 仍漂移；多文档口径冲突；无法映射到 artifacts
- `expected artifact`: claim hierarchy packet
- `current_gate`: `blocked by release-review / claim hierarchy unresolved`

### T55. artifact completeness gate 批次

- `status`: `blocked`
- `hypothesis`: 长任务只有在 artifact completeness gate 通过时才值得放行
- `asset requirement`: artifact schema、presence checks、checksum/metadata policy
- `compute budget`: `planning/eval batch`, future `<= 2 GPUh`
- `stop conditions`: completeness gate 太弱；关键字段缺失；不适配现有 runs
- `expected artifact`: artifact-completeness packet
- `current_gate`: `blocked by release-review / artifact gate unresolved`

### T56. budget escalation ladder 批次

- `status`: `blocked`
- `hypothesis`: 需要先定义从 `smoke -> preview -> decisive rung` 的 budget 升级梯子
- `asset requirement`: budget ladder、stop rule、risk cap
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: ladder 不能约束成本；与现有条目不兼容；无法执行
- `expected artifact`: budget escalation packet
- `current_gate`: `blocked by release-review / budget ladder unresolved`

### T57. final no-go criteria 批次

- `status`: `blocked`
- `hypothesis`: 在进入任何长时无人值守 run 之前，必须先锁定 no-go 边界
- `asset requirement`: no-go matrix、failure thresholds、quality thresholds、comparability thresholds
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: no-go 条件不够硬；阈值无法落地；风险无法收敛
- `expected artifact`: no-go criteria packet
- `current_gate`: `blocked by release-review / no-go matrix unresolved`

### T58. future go criteria 批次

- `status`: `blocked`
- `hypothesis`: 只有 go criteria 明确，blocked backlog 才能在未来安全升级
- `asset requirement`: go matrix、minimum evidence bar、approval owner fields
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: go 条件过宽；无法被 artifacts 支撑；owner 未定义
- `expected artifact`: go criteria packet
- `current_gate`: `blocked by release-review / go matrix unresolved`

### T59. 最终执行顺序与依赖图批次

- `status`: `blocked`
- `hypothesis`: 60 项 backlog 需要最终执行顺序图，否则 program 过长而不可管理
- `asset requirement`: DAG 草图、wave gating、priority tiers
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: DAG 无法压缩；依赖冲突；排序无法自洽
- `expected artifact`: execution DAG packet
- `current_gate`: `blocked by release-review / DAG unresolved`

### T60. 最终长期 blocked program 总册

- `status`: `blocked`
- `hypothesis`: 只有把 60 项 backlog、waves、gate、claims、no-go/go 条件收成总册，后续才可无损接手
- `asset requirement`: T01-T59 的治理产物、统一目录、统一摘要模板
- `compute budget`: `planning batch only`, future decisive envelope `<= 24 GPUh per candidate rung`
- `stop conditions`: 任一核心 packet 未完成；总册仍导致 release drift；交接不可用
- `expected artifact`: final blocked-program handbook
- `current_gate`: `blocked by release-review / depends on T01-T59 partial closure`

### T61. 最佳配置三次长跑一致性批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T07 + T27 + T39`
- `hypothesis`: 当前最佳配置经过三次长跑后仍能保持方向一致
- `asset requirement`: best-1 config locked、统一长跑日志、完整 artifact gate
- `compute budget`: `3 long runs`, total `<= 24 GPUh`
- `stop conditions`: 三次结果分裂；artifact 缺失；超预算
- `expected artifact`: consistency packet、mean/std summary
- `current_gate`: `blocked by release-review / best-1 config not locked`

### T62. loss 组合系数 frontier 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-4`
- `predecessor_pack`: `T03 + T06`
- `hypothesis`: SMP-LoRA 目标函数中的组合系数仍有更优前沿
- `asset requirement`: 可配置 loss coefficient、统一日志和评估脚本
- `compute budget`: `4 runs`, total `<= 18 GPUh`
- `stop conditions`: 训练发散；结果不可解释；预算超标
- `expected artifact`: coefficient frontier packet
- `current_gate`: `blocked by release-review / objective contract unresolved`

### T63. 可训练 block 范围扫描批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-4`
- `predecessor_pack`: `T29 + T30`
- `hypothesis`: 只训练部分 block 可能得到更好隐私-算力折中
- `asset requirement`: trainable-block mask、统一 LoRA 注入配置
- `compute budget`: `4 runs`, total `<= 18 GPUh`
- `stop conditions`: block mask 不可比；收益不稳定；显存失控
- `expected artifact`: trainable-block frontier packet
- `current_gate`: `blocked by release-review / block-mask contract unresolved`

### T64. 更大 shadow 数量 5/7 扩容批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-4`
- `predecessor_pack`: `T25`
- `hypothesis`: 在更多 shadow 下 comparability 方差可进一步下降
- `asset requirement`: 5/7 shadow 资产或等价近似、统一 GSA 管线
- `compute budget`: `2 expansion batches`, total `<= 24 GPUh`
- `stop conditions`: shadow 资产不一致；成本过高；结论无增量
- `expected artifact`: large-shadow packet
- `current_gate`: `blocked by release-review / shadow asset scope unresolved`

### T65. 长时蒸馏/压缩副作用批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-5`
- `predecessor_pack`: `T15 + T38`
- `hypothesis`: 长时训练后再做压缩/蒸馏可能改变隐私和质量边界
- `asset requirement`: compression/distillation path、统一评估面板
- `compute budget`: `2 long runs + 1 eval batch`, total `<= 20 GPUh`
- `stop conditions`: 协议漂移；质量塌缩；结论不可比较
- `expected artifact`: compression side-effect packet
- `current_gate`: `blocked by release-review / compression path unresolved`

### T66. 困难样本 stress 集重跑批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-5`
- `predecessor_pack`: `T10 + T24 + T49`
- `hypothesis`: 当前收益在困难样本 stress 集上仍可维持
- `asset requirement`: hard-case sample set、统一评估脚本、样本绑定记录
- `compute budget`: `2 stress batches`, total `<= 18 GPUh`
- `stop conditions`: stress 集协议不稳；结果翻转；artifact 不完整
- `expected artifact`: hard-case stress packet
- `current_gate`: `blocked by release-review / hard-case protocol unresolved`

### T67. 时间切片重评批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T19 + T33`
- `hypothesis`: 最优隐私点在不同时间切片上有稳定窗口
- `asset requirement`: fixed time slices、统一中间 checkpoint evaluator
- `compute budget`: `1 long run + sliced eval`, total `<= 14 GPUh`
- `stop conditions`: 时间窗口不稳定；中间 artifact 缺失；结果不可解释
- `expected artifact`: time-slice packet
- `current_gate`: `blocked by release-review / time-slice contract unresolved`

### T68. 双设备确定性复跑批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T44 + T45`
- `hypothesis`: 在两类设备与两套环境上可得到方向一致的 blocked-candidate 结论
- `asset requirement`: 双设备、双环境、统一 deterministic policy
- `compute budget`: `2 long runs`, total `<= 20 GPUh`
- `stop conditions`: 确定性无法保障；结果分裂；artifact 不可对齐
- `expected artifact`: dual-device reproducibility packet
- `current_gate`: `blocked by release-review / deterministic policy unresolved`

### T69. 最佳 3 rung ensemble 长跑批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T35 + T39`
- `hypothesis`: best-3 rung ensemble 在长跑后可形成更稳 shortlist
- `asset requirement`: best-3 rung locked、ensemble policy、完整 eval schema
- `compute budget`: `3 long runs + 1 eval batch`, total `<= 24 GPUh`
- `stop conditions`: shortlist 未锁定；ensemble 规则不稳；超预算
- `expected artifact`: ensemble-longrun packet
- `current_gate`: `blocked by release-review / shortlist not locked`

### T70. 长时算力效率 frontier 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-4`
- `predecessor_pack`: `T14 + T21 + T56`
- `hypothesis`: 不同长时算力 envelope 存在更好的 privacy-per-GPUh 前沿
- `asset requirement`: runtime telemetry、budget ladder、统一 cost schema
- `compute budget`: `4 runs`, total `<= 20 GPUh`
- `stop conditions`: cost schema 漂移；收益不增；效率无优势
- `expected artifact`: efficiency frontier packet
- `current_gate`: `blocked by release-review / efficiency contract unresolved`

### T71. release-candidate burn-in 候选批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4/SC-5`
- `predecessor_pack`: `T61 + T68 + T69 + T70`
- `hypothesis`: 若若干 decisive rung 候选通过前序 gate，可做 burn-in 级 blocked-candidate 压力批次
- `asset requirement`: release-candidate shortlist、完整 monitoring、artifact completeness gate
- `compute budget`: `1 burn-in batch`, `<= 24 GPUh`
- `stop conditions`: 任何 burn-in 信号失稳；artifact 缺失；超预算或质量崩塌
- `expected artifact`: burn-in candidate packet
- `current_gate`: `blocked by release-review / depends on T61-T70 partial closure`

### T72. 最终全波次证据总包批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T01-T71`
- `hypothesis`: 只有整合所有 wave 的关键证据，才能形成最终 blocked-candidate 总包
- `asset requirement`: waves 0-7 关键 packet、统一 summary 模板、claim hierarchy matrix
- `compute budget`: `final integration batch`, future decisive envelope `<= 24 GPUh per rung`
- `stop conditions`: 任一关键 packet 缺失；claim hierarchy 漂移；release drift 无法消除
- `expected artifact`: all-wave evidence dossier
- `current_gate`: `blocked by release-review / depends on T01-T71 partial closure`

### T73. Horizon H1 shortlist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T01-T24`
- `hypothesis`: 需要先把 H1 候选从大表里压出来，后续接手才不需要重新拆分
- `asset requirement`: horizon rubric、priority rubric、已有 packet 索引
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: shortlist 无法自洽；与 wave 结构冲突；证据不足
- `expected artifact`: H1 shortlist packet
- `current_gate`: `blocked by release-review / horizon rubric unresolved`

### T74. Horizon H2 shortlist 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T25-T60`
- `hypothesis`: H2 必须只保留真正可能进入 decisive pack 的候选
- `asset requirement`: H2 rubric、packet completeness table、cost bands
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: shortlist 过宽；与 H1/H3 边界冲突；证据映射缺失
- `expected artifact`: H2 shortlist packet
- `current_gate`: `blocked by release-review / horizon rubric unresolved`

### T75. Horizon H3 governance pack 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T61-T72`
- `hypothesis`: H3 主要是治理与 future release simulation，必须单独收口
- `asset requirement`: governance template、claim matrix、owner fields
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: H3 与 H2 混层；governance fields 不完整；packet 无法交接
- `expected artifact`: H3 governance packet
- `current_gate`: `blocked by release-review / governance pack unresolved`

### T76. Priority tier A packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T73`
- `hypothesis`: Tier A 必须形成固定包，后续 planning 才不会漂移
- `asset requirement`: Tier A membership table、gate table、artifact list
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: tier 定义不稳定；条目跨 tier 漂移；资产映射不全
- `expected artifact`: Tier A packet
- `current_gate`: `blocked by release-review / tier packet unresolved`

### T77. Priority tier B packet 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T74`
- `hypothesis`: Tier B 才是真正的 frontier 主战场，需要单独成包
- `asset requirement`: Tier B frontier matrix、cost matrix、risk matrix
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: frontier 定义不稳；条目重复；matrix 缺失
- `expected artifact`: Tier B packet
- `current_gate`: `blocked by release-review / tier packet unresolved`

### T78. Priority tier C packet 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T74 + T75`
- `hypothesis`: Tier C 是稳定性/可迁移性核验层，必须明确依赖图
- `asset requirement`: portability matrix、stability matrix、device/env map
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: portability/stability 混层；device/env map 缺失；依赖图不稳
- `expected artifact`: Tier C packet
- `current_gate`: `blocked by release-review / tier packet unresolved`

### T79. Priority tier D packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T75`
- `hypothesis`: Tier D 是最后 consolidation/burn-in/governance 层，必须独立成包
- `asset requirement`: burn-in checklist、go/no-go matrix、handoff template
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: burn-in/governance 混乱；handoff 不可用；matrix 不闭环
- `expected artifact`: Tier D packet
- `current_gate`: `blocked by release-review / tier packet unresolved`

### T80. 全量 predecessor DAG 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T01-T79`
- `hypothesis`: 没有 predecessor DAG，60+ 项计划仍然不可执行
- `asset requirement`: predecessor map、task metadata、wave-tier-horizon map
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: DAG 不能压缩；存在环；依赖冲突未解
- `expected artifact`: predecessor DAG packet
- `current_gate`: `blocked by release-review / DAG unresolved`

### T81. Review slot matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T80`
- `hypothesis`: 需要先定义 future review slot，防止 backlog 排序被误读成 release
- `asset requirement`: review slot schema、owner fields、approval fields
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: slot 定义不清；owner 缺失；与 release slot 混淆
- `expected artifact`: review-slot matrix
- `current_gate`: `blocked by release-review / slot matrix unresolved`

### T82. Budget band matrix 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-4`
- `predecessor_pack`: `T56 + T80`
- `hypothesis`: 需要把所有任务压到几个固定的 budget band，后续才可管
- `asset requirement`: budget bands、per-task mapping、overrun rules
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: budget band 无法覆盖全部条目；映射失真；overrun rule 太弱
- `expected artifact`: budget-band matrix
- `current_gate`: `blocked by release-review / budget matrix unresolved`

### T83. Artifact class taxonomy 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T55`
- `hypothesis`: 需要先定义 artifact taxonomy，后续 packet 才能自动化
- `asset requirement`: artifact type table、mandatory fields、optional fields
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: taxonomy 仍含糊；mandatory fields 不稳；映射不到现有 artifacts
- `expected artifact`: artifact taxonomy packet
- `current_gate`: `blocked by release-review / taxonomy unresolved`

### T84. Claim-to-artifact binding 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T54 + T83`
- `hypothesis`: strongest claim / allowed claim 必须有一一对应的 artifact binding
- `asset requirement`: claim hierarchy、artifact taxonomy、binding schema
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: claim 无法绑定；binding 产生歧义；审查层不可用
- `expected artifact`: claim-binding packet
- `current_gate`: `blocked by release-review / claim binding unresolved`

### T85. Future smoke slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R1`
- `stop_class`: `SC-2/SC-3/SC-4`
- `predecessor_pack`: `T50 + T81`
- `hypothesis`: 可以先模拟 future smoke slot，但不做真实 release
- `asset requirement`: smoke slot schema、minimum artifact bar、resume checks
- `compute budget`: `simulation batch`, future `<= 4 GPUh`
- `stop conditions`: smoke slot bar 过低；与 blocked queue 混淆；resume 检查不完整
- `expected artifact`: smoke-slot simulation packet
- `current_gate`: `blocked by release-review / smoke slot not approved`

### T86. Future preview slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R2`
- `stop_class`: `SC-2/SC-3/SC-4`
- `predecessor_pack`: `T82 + T85`
- `hypothesis`: preview slot 需要比 smoke 更强的 artifact bar 和 budget bar
- `asset requirement`: preview slot schema、budget bands、artifact gate
- `compute budget`: `simulation batch`, future `<= 8 GPUh`
- `stop conditions`: preview 与 smoke 无法区分；slot 条件不稳；artifact bar 不闭环
- `expected artifact`: preview-slot simulation packet
- `current_gate`: `blocked by release-review / preview slot not approved`

### T87. Future decisive-rung slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R3`
- `stop_class`: `SC-2/SC-3/SC-4/SC-5`
- `predecessor_pack`: `T82 + T86`
- `hypothesis`: decisive-rung slot 必须先模拟其最小可接受证据面
- `asset requirement`: decisive slot schema、quality bar、comparability bar、approval owner fields
- `compute budget`: `simulation batch`, future `<= 24 GPUh`
- `stop conditions`: decisive bar 过宽；owner 未定义；comparability bar 不闭环
- `expected artifact`: decisive-slot simulation packet
- `current_gate`: `blocked by release-review / decisive slot not approved`

### T88. Future unattended-run slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R3`
- `stop_class`: `SC-2/SC-3/SC-4/SC-5`
- `predecessor_pack`: `T34 + T46 + T50 + T87`
- `hypothesis`: 真正的 unattended-run slot 只能是 future simulation，不得现时放行
- `asset requirement`: monitoring bar、resume bar、artifact completeness、failure policy
- `compute budget`: `simulation batch`, future `<= 24 GPUh`
- `stop conditions`: 任一长期运行基础条件缺失；failure policy 不闭环；drift 无法消除
- `expected artifact`: unattended-slot simulation packet
- `current_gate`: `blocked by release-review / unattended slot not approved`

### T89. Multi-wave packet index 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T73-T88`
- `hypothesis`: 需要多 wave packet index，后续接手才不用重新找证据
- `asset requirement`: packet index schema、wave/tier/horizon mapping
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: packet index 不完整；映射冲突；不可检索
- `expected artifact`: packet-index handbook
- `current_gate`: `blocked by release-review / packet index unresolved`

### T90. Future release decision tree 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T57 + T58 + T81`
- `hypothesis`: 最终需要一棵 `go / hold / no-go` 决策树，避免再次重分析
- `asset requirement`: decision tree schema、thresholds、owner fields
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: decision tree 不可执行；阈值不稳；owner 缺失
- `expected artifact`: release decision tree
- `current_gate`: `blocked by release-review / decision tree unresolved`

### T91. Future long-run heartbeat simulation 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R1`
- `stop_class`: `SC-3/SC-4`
- `predecessor_pack`: `T46 + T88`
- `hypothesis`: 未来长跑需要 heartbeat 机制，但当前只能先模拟
- `asset requirement`: heartbeat schema、timeout rules、alert fields
- `compute budget`: `simulation batch`, future `<= 2 GPUh`
- `stop conditions`: heartbeat 条件过弱；alert 不可落地；与 release drift 冲突
- `expected artifact`: heartbeat simulation packet
- `current_gate`: `blocked by release-review / heartbeat not approved`

### T92. Future resumable-run simulation 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R1`
- `stop_class`: `SC-3/SC-4`
- `predecessor_pack`: `T34 + T45 + T91`
- `hypothesis`: 未来恢复型长跑必须先通过 resumable-run simulation
- `asset requirement`: resume checkpoints、resume validation schema、state handoff fields
- `compute budget`: `simulation batch`, future `<= 4 GPUh`
- `stop conditions`: resume 验证不闭环；handoff fields 不完整；与 current blocked policy 冲突
- `expected artifact`: resumable-run simulation packet
- `current_gate`: `blocked by release-review / resumable-run not approved`

### T93. Future artifact escrow simulation 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T55 + T83 + T92`
- `hypothesis`: 长程 program 需要 artifact escrow 机制，但当前只能先做模拟
- `asset requirement`: escrow schema、checksum policy、retention policy
- `compute budget`: `simulation batch`, future `<= 2 GPUh`
- `stop conditions`: escrow policy 过弱；checksum 不能覆盖；retention 无法执行
- `expected artifact`: artifact-escrow simulation packet
- `current_gate`: `blocked by release-review / escrow not approved`

### T94. Final horizon execution map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T73-T93`
- `hypothesis`: 最终需要一个 horizon execution map，后续接手才可无损续跑
- `asset requirement`: horizon map、tier map、slot map、decision tree
- `compute budget`: `planning batch`, future `<= 2 GPUh`
- `stop conditions`: map 仍混乱；跨维度映射冲突；不可交接
- `expected artifact`: final horizon execution map
- `current_gate`: `blocked by release-review / execution map unresolved`

### T95. Final planning packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T01-T94`
- `hypothesis`: 需要一份最终 planning packet，把 90+ 条 blocked backlog 压成可审查总包
- `asset requirement`: all-wave summaries、decision tree、execution map、claim bindings
- `compute budget`: `planning batch only`, future `<= 2 GPUh`
- `stop conditions`: 总包仍导致 drift；关键 packet 缺失；层级仍混乱
- `expected artifact`: final planning packet
- `current_gate`: `blocked by release-review / planning packet unresolved`

### T96. Final blocked-master-schedule 冻结批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T95`
- `hypothesis`: 只有冻结成最终 blocked master schedule，下一轮才无需重新扩表
- `asset requirement`: final planning packet、state sync fields、resume fields
- `compute budget`: `planning batch only`, future `<= 2 GPUh`
- `stop conditions`: 仍需重新拆分全局；resume fields 不够；drift 风险未收敛
- `expected artifact`: frozen blocked master schedule
- `current_gate`: `blocked by release-review / master schedule not frozen`

### T97. 500-step 高频 checkpoint 长时重跑

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T19 + T61`
- `hypothesis`: 更高 checkpoint 频率可发现被当前稀疏保存掩盖的最优窗口
- `asset_requirement`: 高频保存策略、稳定存储、统一中间评估
- `compute_budget`: `1 long run`, `<= 24 GPUh`
- `stop_conditions`: 存储爆炸；中间评估无增量；artifact 不完整
- `expected_artifact`: dense-checkpoint curve packet
- `current_gate`: `blocked by release-review / dense checkpoint policy unresolved`

### T98. 300-step repeat-longrun 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-4`
- `predecessor_pack`: `T61 + T97`
- `hypothesis`: 重复长跑是验证长周期稳定性的最低条件
- `asset_requirement`: locked best config、统一 seed policy、longrun telemetry
- `compute_budget`: `2 long runs`, total `<= 24 GPUh`
- `stop_conditions`: 两次结果分裂；无方向性收益；预算溢出
- `expected_artifact`: repeat-longrun packet
- `current_gate`: `blocked by release-review / longrun repeat criteria unresolved`

### T99. 3x device envelope burn-in 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T44 + T68`
- `hypothesis`: 真正的长期候选必须跨 3 类设备保持方向一致
- `asset_requirement`: 3 类 GPU 设备、统一 deterministic policy、统一 artifact schema
- `compute_budget`: `3 long runs`, total `<= 24 GPUh`
- `stop_conditions`: 跨设备结果分裂；设备缺失；artifact 不对齐
- `expected_artifact`: triple-device envelope packet
- `current_gate`: `blocked by release-review / multi-device scope unresolved`

### T100. 3x environment reproducibility burn-in 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T45 + T68`
- `hypothesis`: 跨 3 套环境复现是未来 release-review 的最低门槛
- `asset_requirement`: 3 套环境锁定文件、统一 launch path、portable config
- `compute_budget`: `3 long runs`, total `<= 24 GPUh`
- `stop_conditions`: 环境不可复现；结果分裂；portable path 失效
- `expected_artifact`: triple-env reproducibility packet
- `current_gate`: `blocked by release-review / environment reproducibility unresolved`

### T101. 超长 24h burn-in 候选批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4/SC-5`
- `predecessor_pack`: `T71 + T88`
- `hypothesis`: 只有超长 burn-in 才能暴露真正的长时失效模式
- `asset_requirement`: monitoring、resume、artifact completeness、quality checks
- `compute_budget`: `1 burn-in run`, `<= 24 GPUh`
- `stop_conditions`: 任何 heartbeat/resume/quality 条件失败；artifact 缺失；预算失控
- `expected_artifact`: 24h burn-in candidate packet
- `current_gate`: `blocked by release-review / burn-in slot not approved`

### T102. 最佳 2 config head-to-head 长跑批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-4/SC-5`
- `predecessor_pack`: `T27 + T39 + T69`
- `hypothesis`: 最佳 2 个 config 需要 head-to-head 长跑，才能形成未来 shortlist 决策
- `asset_requirement`: best-2 config locked、统一 quality and attack panel
- `compute_budget`: `2 long runs`, total `<= 24 GPUh`
- `stop_conditions`: best-2 未锁定；结论继续打平；质量崩塌
- `expected_artifact`: head-to-head longrun packet
- `current_gate`: `blocked by release-review / best-2 not locked`

### T103. 大样本质量 panel 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-3/SC-4/SC-5`
- `predecessor_pack`: `T11 + T38 + T43`
- `hypothesis`: 当前质量结论需要更大样本量才能稳定
- `asset_requirement`: 大样本生成管线、质量缓存、统一 panel schema
- `compute_budget`: `1 quality mega-batch`, `<= 18 GPUh`
- `stop_conditions`: 缓存失稳；质量指标冲突不可解释；样本不足
- `expected_artifact`: large-sample quality packet
- `current_gate`: `blocked by release-review / large-sample quality unresolved`

### T104. 大样本攻击 panel 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4`
- `predecessor_pack`: `T08 + T24 + T37`
- `hypothesis`: 当前攻击收益需要在更大样本量下确认
- `asset_requirement`: 大样本 eval split、多攻击器统一接口
- `compute_budget`: `1 attack mega-batch`, `<= 18 GPUh`
- `stop_conditions`: attack panel 漂移；样本量不足；artifact 不统一
- `expected_artifact`: large-sample attack packet
- `current_gate`: `blocked by release-review / large-sample attack unresolved`

### T105. 5-seed x 2-config stability grid 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-4`
- `predecessor_pack`: `T07 + T102`
- `hypothesis`: 真正的候选必须在多 seed x 多 config 网格上保持稳定
- `asset_requirement`: 5 seeds、2 locked configs、统一 metrics
- `compute_budget`: `10 runs`, total `<= 24 GPUh`
- `stop_conditions`: 方差过大；计算预算不可接受；best config 继续漂移
- `expected_artifact`: seed-config grid packet
- `current_gate`: `blocked by release-review / stability grid unresolved`

### T106. 5-split x 2-config transfer grid 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-4`
- `predecessor_pack`: `T17 + T26 + T105`
- `hypothesis`: 当前 blocked-candidate 需要跨 split transfer 稳定性
- `asset_requirement`: 5 split、2 configs、统一 transfer summary schema
- `compute_budget`: `10 runs`, total `<= 24 GPUh`
- `stop_conditions`: split 协议漂移；结果翻转；预算失控
- `expected_artifact`: split-transfer grid packet
- `current_gate`: `blocked by release-review / transfer grid unresolved`

### T107. 7-shadow x 2-device 扩容批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-2/SC-4`
- `predecessor_pack`: `T64 + T99`
- `hypothesis`: 更大 shadow 和双设备联合压力能更真实地暴露 future decisive rung 风险
- `asset_requirement`: 7 shadow、2 devices、统一 GSA schema
- `compute_budget`: `2 expansion batches`, total `<= 24 GPUh`
- `stop_conditions`: shadow 或 device 任何一侧不稳；结果不可解释；预算超标
- `expected_artifact`: large-shadow device packet
- `current_gate`: `blocked by release-review / large shadow-device scope unresolved`

### T108. 质量-攻击-算力三维大表重跑

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-1/SC-3/SC-4/SC-5`
- `predecessor_pack`: `T15 + T70 + T103 + T104`
- `hypothesis`: 最终三维大表必须建立在更大样本和更长跑的基础上
- `asset_requirement`: unified cost schema、large-sample panels、locked configs
- `compute_budget`: `1 mega-batch`, `<= 24 GPUh`
- `stop_conditions`: 三维表仍不稳定；指标冲突；artifact 不足
- `expected_artifact`: mega frontier table
- `current_gate`: `blocked by release-review / mega frontier unresolved`

### T109. 48h future-slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R3`
- `stop_class`: `SC-3/SC-4/SC-5`
- `predecessor_pack`: `T88 + T101`
- `hypothesis`: 真正的未来 unattended slot 必须先经过 48h simulation 评估
- `asset_requirement`: 48h telemetry schema、resume/heartbeat/escrow policy
- `compute_budget`: `simulation batch`, future envelope `<= 24 GPUh`
- `stop_conditions`: 48h slot 条件不闭环；基础设施不足；release drift 风险过高
- `expected_artifact`: 48h slot simulation packet
- `current_gate`: `blocked by release-review / long-slot simulation not approved`

### T110. 72h future-slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R3`
- `stop_class`: `SC-3/SC-4/SC-5`
- `predecessor_pack`: `T109`
- `hypothesis`: 72h slot 只是 future simulation 上限，不应被提前放行
- `asset_requirement`: 72h telemetry schema、failure policy escalation、artifact escrow
- `compute_budget`: `simulation batch`, future envelope `<= 24 GPUh`
- `stop_conditions`: 72h slot 不可管理；failure policy 不闭环；artifact 风险过大
- `expected_artifact`: 72h slot simulation packet
- `current_gate`: `blocked by release-review / 72h slot not approved`

### T111. 96h future-slot simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R3`
- `stop_class`: `SC-3/SC-4/SC-5`
- `predecessor_pack`: `T110`
- `hypothesis`: 96h simulation 只用于证明为什么当前不能直接无人值守
- `asset_requirement`: 96h risk model、escrow policy、heartbeat escalation tree
- `compute_budget`: `simulation batch`, future envelope `<= 24 GPUh`
- `stop_conditions`: 风险不可控；simulation 无法定义；release drift 风险放大
- `expected_artifact`: 96h slot simulation packet
- `current_gate`: `blocked by release-review / 96h slot not approved`

### T112. final long-horizon slot cap 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-4`
- `predecessor_pack`: `T109-T111`
- `hypothesis`: 必须先明确 future long-horizon slot 的成本和时长上限
- `asset_requirement`: slot cap matrix、risk cap matrix、owner fields
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: slot cap 过宽；owner 缺失；与 budget ladder 冲突
- `expected_artifact`: long-horizon slot cap packet
- `current_gate`: `blocked by release-review / slot cap unresolved`

### T113. final future shortlist lock 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T73-T112`
- `hypothesis`: 需要最终锁定 future shortlist，否则 90+ backlog 仍过宽
- `asset_requirement`: shortlist rubric、tier/horizon map、decision tree
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: shortlist 仍漂移；tier/horizon map 冲突；证据不足
- `expected_artifact`: final future shortlist packet
- `current_gate`: `blocked by release-review / shortlist not frozen`

### T114. final blocked candidate ladder 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T56 + T82 + T113`
- `hypothesis`: 所有 future candidate 必须映射到一个固定 blocked candidate ladder
- `asset_requirement`: candidate ladder、budget bands、claim bindings
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: ladder 仍含糊；budget 和 claim binding 冲突；不可交接
- `expected_artifact`: blocked candidate ladder
- `current_gate`: `blocked by release-review / ladder unresolved`

### T115. final release-readiness anti-check 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T57 + T58 + T90 + T114`
- `hypothesis`: 在任何 future go 之前，必须先有一套 anti-check 防止误放行
- `asset_requirement`: anti-checklist、negative signals、drift rules
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: anti-check 太弱；negative signals 不完整；drift rules 无法执行
- `expected_artifact`: release-readiness anti-check packet
- `current_gate`: `blocked by release-review / anti-check unresolved`

### T116. final handoff resume pack 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T89 + T94 + T96 + T115`
- `hypothesis`: 下一轮接手必须无需重分析全局
- `asset_requirement`: resume fields、handoff packet、packet index、execution map
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: handoff 仍需重分析；resume fields 不足；索引不稳定
- `expected_artifact`: handoff-resume pack
- `current_gate`: `blocked by release-review / handoff pack unresolved`

### T117. final no-release proof 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T115 + T116`
- `hypothesis`: 需要一份明确的 no-release proof，防止 backlog 被误读成可执行
- `asset_requirement`: no-release reasoning、evidence links、hard boundary set
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: no-release proof 不完整；边界不够硬；仍可被误读
- `expected_artifact`: no-release proof packet
- `current_gate`: `blocked by release-review / no-release proof unresolved`

### T118. final all-wave sync packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T01-T117`
- `hypothesis`: 只有 all-wave sync packet 才能把 backlog、waves、tiers、horizons 完整对齐
- `asset_requirement`: all-wave summaries、tier/horizon maps、slot matrices
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: 任一层映射漂移；sync packet 缺字段；仍不可检索
- `expected_artifact`: all-wave sync packet
- `current_gate`: `blocked by release-review / all-wave sync unresolved`

### T119. final master-schedule audit 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T118`
- `hypothesis`: 需要对最终 master schedule 做一次自审，确保没有 release drift
- `asset_requirement`: audit checklist、drift checklist、claim-check matrix
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: drift 未消除；claim-check 仍冲突；审计不通过
- `expected_artifact`: master-schedule audit packet
- `current_gate`: `blocked by release-review / audit unresolved`

### T120. final frozen long-horizon blocked handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T119`
- `hypothesis`: 需要把所有 blocked planning 最终冻结成 long-horizon handbook，后续无需再扩表
- `asset_requirement`: frozen handbook template、state sync fields、resume fields、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需重新扩表；resume 不完整；hard boundary 不足
- `expected_artifact`: frozen long-horizon blocked handbook
- `current_gate`: `blocked by release-review / handbook not frozen`

### T121. Campaign A packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T01-T38`
- `hypothesis`: comparator/frontier 任务必须独立收成 Campaign A 才能持续接力
- `asset_requirement`: frontier matrix、claim bindings、artifact taxonomy
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: frontier 面仍漂移；claim binding 不闭环；taxonomy 不稳
- `expected_artifact`: Campaign A packet
- `current_gate`: `blocked by release-review / campaign packet unresolved`

### T122. Campaign B packet 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3/SC-4`
- `predecessor_pack`: `T39-T72`
- `hypothesis`: 稳定性/可迁移性/长跑任务必须独立收成 Campaign B
- `asset_requirement`: stability matrix、device-env map、burn-in policy
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: stability/burn-in 条件不闭环；device-env map 缺失；层级冲突
- `expected_artifact`: Campaign B packet
- `current_gate`: `blocked by release-review / campaign packet unresolved`

### T123. Campaign C packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T73-T120`
- `hypothesis`: governance/slot simulation 任务必须独立收成 Campaign C
- `asset_requirement`: slot matrix、decision tree、anti-check、handoff pack
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: slot simulation 仍漂移；decision tree 不可执行；handoff 不可用
- `expected_artifact`: Campaign C packet
- `current_gate`: `blocked by release-review / campaign packet unresolved`

### T124. Campaign D frozen handoff packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T123`
- `hypothesis`: 最终 frozen handoff 需要单独的 Campaign D 包
- `asset_requirement`: cross-campaign summary、resume fields、frozen schedule template
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: cross-campaign 仍冲突；resume 不完整；模板不可交接
- `expected_artifact`: Campaign D packet
- `current_gate`: `blocked by release-review / frozen handoff unresolved`

### T125. H1-H2 bridge packet 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T73 + T74 + T82`
- `hypothesis`: H1 到 H2 的升级条件需要单独收口
- `asset_requirement`: horizon bridge schema、budget ladder、priority rubric
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: horizon bridge 仍模糊；rubric 冲突；升级条件不可执行
- `expected_artifact`: H1-H2 bridge packet
- `current_gate`: `blocked by release-review / horizon bridge unresolved`

### T126. H2-H3 bridge packet 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T74 + T75 + T90`
- `hypothesis`: H2 到 H3 的边界不清会导致 governance 与 decisive pack 混层
- `asset_requirement`: H2-H3 bridge schema、go/no-go matrix、slot matrix
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: bridge 条件不稳；matrix 冲突；slot 边界不清
- `expected_artifact`: H2-H3 bridge packet
- `current_gate`: `blocked by release-review / horizon bridge unresolved`

### T127. H3-H4 bridge packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T75 + T109-T112`
- `hypothesis`: H3 到 H4 的迁移必须明确 long-slot simulation 的边界
- `asset_requirement`: long-slot cap、simulation packets、risk cap matrix
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: H4 边界不稳；risk cap 过宽；simulation 条件冲突
- `expected_artifact`: H3-H4 bridge packet
- `current_gate`: `blocked by release-review / horizon bridge unresolved`

### T128. H4-H5 bridge packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T124`
- `hypothesis`: H5 只有在 campaign/frozen handoff 完成后才有意义
- `asset_requirement`: H5 scope、frozen schedule template、campaign summaries
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: H5 scope 过宽；campaign 未闭环；template 不可交接
- `expected_artifact`: H4-H5 bridge packet
- `current_gate`: `blocked by release-review / horizon bridge unresolved`

### T129. priority band P0-P3 packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T73-T79`
- `hypothesis`: 需要把 140+ 项候选映射到固定 priority bands，后续接手才不混乱
- `asset_requirement`: priority band schema、per-task mapping、band descriptions
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: bands 过宽；映射冲突；条目跨 band 漂移
- `expected_artifact`: priority-band packet
- `current_gate`: `blocked by release-review / band schema unresolved`

### T130. release slot escalation ladder 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-4`
- `predecessor_pack`: `T81 + T85-T88`
- `hypothesis`: smoke/preview/decisive/unattended 的升级梯子必须先冻结
- `asset_requirement`: slot ladder schema、approval owner fields、cap matrix
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: slot ladder 不闭环；owner 缺失；cap matrix 冲突
- `expected_artifact`: slot-escalation ladder
- `current_gate`: `blocked by release-review / slot ladder unresolved`

### T131. campaign-to-slot binding 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T130`
- `hypothesis`: 只有把 campaigns 绑定到 slots，后续 future review 才不会误判
- `asset_requirement`: campaign map、slot ladder、binding schema
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: binding 含糊；campaign 跨 slot 漂移；不可审查
- `expected_artifact`: campaign-slot binding packet
- `current_gate`: `blocked by release-review / binding unresolved`

### T132. long-horizon governance audit packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T131`
- `hypothesis`: 需要一份总治理审计包确认 backlog 仍未越界
- `asset_requirement`: governance checklist、drift checklist、hard boundary set
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: 仍有 release drift；边界不完整；审计不通过
- `expected_artifact`: long-horizon governance audit packet
- `current_gate`: `blocked by release-review / governance audit unresolved`

### T133. frozen Q1 blocked plan 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T132`
- `hypothesis`: 需要季度化 blocked plan，后续接手不应再从头拆分
- `asset_requirement`: quarter template、Q1 scope、task mapping
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: quarter scope 含糊；与 horizons 冲突；mapping 不稳
- `expected_artifact`: frozen Q1 blocked plan
- `current_gate`: `blocked by release-review / quarterly plan unresolved`

### T134. frozen Q2 blocked plan 批次

- `status`: `blocked`
- `tier`: `Tier B`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T133`
- `hypothesis`: Q2 应承接 H2/H3 decisive/governance blocked pack
- `asset_requirement`: Q2 scope、band mapping、campaign mapping
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: scope 冲突；mapping 漂移；不可交接
- `expected_artifact`: frozen Q2 blocked plan
- `current_gate`: `blocked by release-review / quarterly plan unresolved`

### T135. frozen Q3 blocked plan 批次

- `status`: `blocked`
- `tier`: `Tier C`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T134`
- `hypothesis`: Q3 应主要承接 H4 burn-in/long-slot blocked pack
- `asset_requirement`: Q3 scope、H4 mapping、slot cap bindings
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: H4 mapping 不稳；quarter 边界不清；不可交接
- `expected_artifact`: frozen Q3 blocked plan
- `current_gate`: `blocked by release-review / quarterly plan unresolved`

### T136. frozen Q4 blocked plan 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T135`
- `hypothesis`: Q4 主要承接 H5 frozen governance / master schedule handoff
- `asset_requirement`: Q4 scope、H5 mapping、handoff packet
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: H5 scope 不稳；handoff packet 缺失；quarter 失真
- `expected_artifact`: frozen Q4 blocked plan
- `current_gate`: `blocked by release-review / quarterly plan unresolved`

### T137. yearly blocked roadmap packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T133-T136`
- `hypothesis`: 最终需要年尺度 blocked roadmap，后续才不会继续无边界扩表
- `asset_requirement`: yearly roadmap template、quarter summaries、campaign map
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: 年计划仍需重拆分；quarter summaries 冲突；template 不稳
- `expected_artifact`: yearly blocked roadmap
- `current_gate`: `blocked by release-review / yearly roadmap unresolved`

### T138. final campaign scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T121-T137`
- `hypothesis`: 需要一个 scoreboard 把 campaign 完成度、阻塞、风险统一量化
- `asset_requirement`: scoreboard schema、campaign metrics、risk fields
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；scoreboard 误导；risk fields 缺失
- `expected_artifact`: campaign scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T139. final horizon scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T125-T137`
- `hypothesis`: horizons 也需要 scoreboard，否则长程计划仍不可导航
- `asset_requirement`: horizon metrics、scope fields、dependency fields
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: horizon metrics 漂移；scope 字段不稳；导航无效
- `expected_artifact`: horizon scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T140. final slot scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T130-T132`
- `hypothesis`: slot/slot-ladder 也需要 scoreboard，防止 future release simulation 混成真 release
- `asset_requirement`: slot metrics、cap fields、approval fields
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: slot metrics 不稳；approval fields 缺失；混层风险未消除
- `expected_artifact`: slot scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T141. frozen predecessor graph handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T80 + T94 + T137`
- `hypothesis`: predecessor graph 需要冻结成手册，后续接手才不会再重画
- `asset_requirement`: graph export、node metadata、edge rules
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: graph 仍漂移；edge rules 不清；手册不可检索
- `expected_artifact`: predecessor-graph handbook
- `current_gate`: `blocked by release-review / graph handbook unresolved`

### T142. frozen future-review packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T85-T140`
- `hypothesis`: 需要一份 future-review packet，但必须明确它不是当前 release
- `asset_requirement`: future review template、negative boundary set、slot scoreboards
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: packet 容易被误读成 release；negative boundary 不够硬；template 漂移
- `expected_artifact`: frozen future-review packet
- `current_gate`: `blocked by release-review / future-review packet unresolved`

### T143. frozen blocked execution atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T141 + T142`
- `hypothesis`: 需要一个 atlas 把所有 blocked planning 变成可浏览执行地图
- `asset_requirement`: atlas template、graph handbook、scoreboards、quarter plans
- `compute_budget`: `planning batch`, future `<= 2 GPUh`
- `stop_conditions`: atlas 仍需重分析；导航不完整；数据不同步
- `expected_artifact`: blocked execution atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T144. final frozen campaign-horizon-slot handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T143`
- `hypothesis`: 需要最终冻结一本 campaign-horizon-slot handbook，后续无需再扩展计划结构
- `asset_requirement`: handbook template、all-wave sync、resume fields、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩展结构；resume fields 缺失；hard boundaries 不足
- `expected_artifact`: frozen campaign-horizon-slot handbook
- `current_gate`: `blocked by release-review / handbook not frozen`

### T145. annual campaign map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T121-T144`
- `hypothesis`: 年度级 blocked 计划必须有独立 campaign map，后续才不需要重新拆解
- `asset_requirement`: annual map template、campaign summaries、quarter plans
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: annual map 不稳；campaign 仍冲突；quarter plans 不闭环
- `expected_artifact`: annual campaign map
- `current_gate`: `blocked by release-review / annual map unresolved`

### T146. annual horizon matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T125-T128 + T137`
- `hypothesis`: H1-H6 需要年度级 horizon matrix 才能长期交接
- `asset_requirement`: horizon matrix、bridge packets、yearly roadmap
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: horizon 边界不稳；bridge packet 冲突；matrix 不可检索
- `expected_artifact`: annual horizon matrix
- `current_gate`: `blocked by release-review / annual horizon unresolved`

### T147. annual slot governance packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T130 + T140 + T142`
- `hypothesis`: 年度级 slot governance 才能防止 future slot simulation 被误读成 release
- `asset_requirement`: slot ladder、slot scoreboard、future-review packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: slot governance 含糊；scoreboard 漂移；future-review packet 不稳
- `expected_artifact`: annual slot governance packet
- `current_gate`: `blocked by release-review / slot governance unresolved`

### T148. annual anti-drift packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T115 + T117 + T132`
- `hypothesis`: 年尺度 blocked 规划需要更强 anti-drift 包，防止表述越界
- `asset_requirement`: anti-check、no-release proof、governance audit
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: anti-drift 信号不够硬；proof 仍可误读；audit 不闭环
- `expected_artifact`: annual anti-drift packet
- `current_gate`: `blocked by release-review / anti-drift unresolved`

### T149. annual artifact retention packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T55 + T93 + T118`
- `hypothesis`: 年度 blocked 计划必须先定义 artifact retention/escrow 规则
- `asset_requirement`: retention schema、escrow packet、all-wave sync
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: retention 规则太弱；escrow 不稳定；sync 不完整
- `expected_artifact`: annual artifact retention packet
- `current_gate`: `blocked by release-review / retention unresolved`

### T150. annual master shortlist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T113 + T129 + T138 + T139`
- `hypothesis`: 年度级 master shortlist 能防止 160+ backlog 继续无边界膨胀
- `asset_requirement`: shortlist rubric、priority bands、campaign/horizon scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: shortlist 仍过宽；scoreboard 冲突；band 映射漂移
- `expected_artifact`: annual master shortlist
- `current_gate`: `blocked by release-review / shortlist unresolved`

### T151. annual execution atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T143 + T145 + T146`
- `hypothesis`: 年度 blocked 计划需要 atlas 层，后续接手才无需重分析
- `asset_requirement`: atlas template、annual map、horizon matrix、graph handbook
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；annual/horizon 仍冲突；graph 不稳
- `expected_artifact`: annual execution atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T152. annual no-go ceiling 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T57 + T90 + T112`
- `hypothesis`: 年度 blocked 计划需要 no-go ceiling，限制长期扩跑幻想
- `asset_requirement`: no-go matrix、decision tree、slot cap
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: no-go ceiling 过宽；thresholds 不闭环；与 decision tree 冲突
- `expected_artifact`: annual no-go ceiling
- `current_gate`: `blocked by release-review / no-go ceiling unresolved`

### T153. annual go-bar simulation 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T58 + T90 + T150`
- `hypothesis`: 年度级 go-bar 只能先做 simulation，不能提前放行
- `asset_requirement`: go matrix、master shortlist、decision tree
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: go-bar 过宽；shortlist 不稳；drift 风险未收敛
- `expected_artifact`: annual go-bar simulation
- `current_gate`: `blocked by release-review / go-bar unresolved`

### T154. annual campaign scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T138 + T145`
- `hypothesis`: 需要把 campaign scoreboard 升成年度级 scoreboard
- `asset_requirement`: scoreboard template、annual metrics、campaign map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 漂移；template 不稳；campaign map 不闭环
- `expected_artifact`: annual campaign scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T155. annual handoff packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T116 + T151 + T154`
- `hypothesis`: 年度 blocked 计划需要 handoff 包，防止跨季度/跨年丢失状态
- `asset_requirement`: handoff template、resume fields、annual atlas、annual scoreboard
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: handoff 仍需重分析；resume 不完整；atlas/scoreboard 不闭环
- `expected_artifact`: annual handoff packet
- `current_gate`: `blocked by release-review / handoff unresolved`

### T156. annual frozen blocked book 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T145-T155`
- `hypothesis`: 年度级 blocked 规划最终需要一本 frozen blocked book
- `asset_requirement`: frozen template、annual packet set、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；frozen 模板不稳；边界不够硬
- `expected_artifact`: annual frozen blocked book
- `current_gate`: `blocked by release-review / annual book unresolved`

### T157. multi-year campaign map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T156`
- `hypothesis`: 多年期 blocked 体系需要独立 campaign map，否则 H6 无意义
- `asset_requirement`: multi-year template、annual books、campaign summaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: multi-year template 漂移；annual books 冲突；campaign 不闭环
- `expected_artifact`: multi-year campaign map
- `current_gate`: `blocked by release-review / multi-year map unresolved`

### T158. multi-year horizon matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T146 + T157`
- `hypothesis`: H1-H6 需要跨年度 horizon matrix 才能真正冻结
- `asset_requirement`: multi-year horizon schema、annual horizon matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: horizon schema 不稳；annual matrices 冲突；H6 仍空泛
- `expected_artifact`: multi-year horizon matrix
- `current_gate`: `blocked by release-review / multi-year horizon unresolved`

### T159. multi-year slot governance 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T147 + T157`
- `hypothesis`: slot governance 需要跨年度收口，才能真正阻止 release drift
- `asset_requirement`: slot governance template、annual slot packets、multi-year map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: slot governance 漂移；跨年度映射不稳；drift 风险未收敛
- `expected_artifact`: multi-year slot governance
- `current_gate`: `blocked by release-review / multi-year slot unresolved`

### T160. multi-year anti-release proof 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T148 + T159`
- `hypothesis`: 多年期 blocked 计划必须有更强 anti-release proof
- `asset_requirement`: anti-release schema、annual anti-drift packet、slot governance
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: proof 仍不够硬；schema 不闭环；slot governance 未定
- `expected_artifact`: multi-year anti-release proof
- `current_gate`: `blocked by release-review / anti-release proof unresolved`

### T161. multi-year roadmap packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T137 + T156 + T157`
- `hypothesis`: 需要一个跨年度 roadmap packet，避免后续重复扩表
- `asset_requirement`: multi-year roadmap template、annual books、campaign map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: roadmap 仍需重分析；template 不稳；annual books 冲突
- `expected_artifact`: multi-year roadmap packet
- `current_gate`: `blocked by release-review / roadmap unresolved`

### T162. multi-year shortlist lock 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T150 + T153 + T161`
- `hypothesis`: 多年期 blocked 体系也需要锁定 shortlist，否则继续膨胀
- `asset_requirement`: shortlist rubric、annual shortlist、roadmap packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: shortlist 仍漂移；rubric 不稳；roadmap 不闭环
- `expected_artifact`: multi-year shortlist lock
- `current_gate`: `blocked by release-review / shortlist unresolved`

### T163. multi-year handoff atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T151 + T155 + T161`
- `hypothesis`: 需要一个跨年度 handoff atlas 才能让后续轮次无损接手
- `asset_requirement`: atlas template、multi-year map、annual handoff packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；handoff packet 冲突；multi-year map 漂移
- `expected_artifact`: multi-year handoff atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T164. multi-year scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T154 + T139 + T140`
- `hypothesis`: 需要年度以上的 campaign/horizon/slot scoreboard 体系
- `asset_requirement`: scoreboard templates、annual scoreboards、metrics schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不可汇总；templates 冲突；scoreboard 误导
- `expected_artifact`: multi-year scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T165. multi-year predecessor handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T141 + T163`
- `hypothesis`: predecessor graph 需要跨年度冻结成 handbook，避免未来重绘
- `asset_requirement`: graph handbook、multi-year atlas、edge rules
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: edge rules 不稳；atlas 冲突；handbook 不可检索
- `expected_artifact`: multi-year predecessor handbook
- `current_gate`: `blocked by release-review / handbook unresolved`

### T166. frozen campaign-year-slot ladder 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T131 + T137 + T161`
- `hypothesis`: 最终需要一条 campaign-year-slot ladder 把长期 blocked 结构完全固定
- `asset_requirement`: ladder template、campaign map、year roadmap、slot ladder
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: ladder 仍模糊；template 冲突；slot ladder 不闭环
- `expected_artifact`: campaign-year-slot ladder
- `current_gate`: `blocked by release-review / ladder unresolved`

### T167. frozen multi-year no-release charter 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T160 + T166`
- `hypothesis`: 需要一份跨年度 no-release charter，彻底阻止误放行
- `asset_requirement`: charter template、anti-release proof、ladder
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不够硬；anti-release proof 不闭环；ladder 不稳
- `expected_artifact`: frozen multi-year no-release charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T168. final frozen multi-year blocked canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T157-T167`
- `hypothesis`: 需要一本最终 frozen multi-year blocked canon，后续不再继续扩展计划结构
- `asset_requirement`: canon template、all annual/multi-year packets、hard boundaries、resume fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；canon 不稳；resume fields 不够
- `expected_artifact`: frozen multi-year blocked canon
- `current_gate`: `blocked by release-review / canon not frozen`

### T169. portfolio campaign map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T157-T168`
- `hypothesis`: 多年期 blocked 体系需要升级为 portfolio campaign map，后续才不会继续散开
- `asset_requirement`: portfolio template、multi-year canon、campaign summaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: portfolio template 不稳；canon 冲突；campaign 映射不闭环
- `expected_artifact`: portfolio campaign map
- `current_gate`: `blocked by release-review / portfolio map unresolved`

### T170. portfolio horizon matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T158 + T169`
- `hypothesis`: H1-H7 需要组合成 portfolio horizon matrix，后续接手才不用重新定义时间尺度
- `asset_requirement`: H1-H7 schema、multi-year matrices、portfolio map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: H7 仍空泛；matrix 不闭环；portfolio map 漂移
- `expected_artifact`: portfolio horizon matrix
- `current_gate`: `blocked by release-review / portfolio horizon unresolved`

### T171. portfolio slot governance 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T159 + T170`
- `hypothesis`: 需要 portfolio 级 slot governance 才能长期防止 release drift
- `asset_requirement`: portfolio slot schema、multi-year slot governance、cap matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: slot governance 不稳；cap matrix 冲突；portfolio schema 含糊
- `expected_artifact`: portfolio slot governance
- `current_gate`: `blocked by release-review / portfolio slot unresolved`

### T172. portfolio anti-release charter 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T160 + T167 + T171`
- `hypothesis`: portfolio 级 blocked 体系需要更强的 anti-release charter
- `asset_requirement`: charter template、anti-release proof、slot governance
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 仍不够硬；proof 不闭环；slot governance 漂移
- `expected_artifact`: portfolio anti-release charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T173. portfolio roadmap packet 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T161 + T169 + T170`
- `hypothesis`: portfolio 层需要单独 roadmap packet，后续不再继续扩主表
- `asset_requirement`: roadmap template、portfolio map、horizon matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: roadmap 不稳；portfolio map 冲突；matrix 不闭环
- `expected_artifact`: portfolio roadmap packet
- `current_gate`: `blocked by release-review / roadmap unresolved`

### T174. portfolio shortlist lock 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T162 + T173`
- `hypothesis`: 需要 portfolio 级 shortlist，防止 190+ backlog 继续膨胀
- `asset_requirement`: shortlist rubric、portfolio roadmap、band schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: shortlist 仍过宽；rubric 漂移；roadmap 未锁定
- `expected_artifact`: portfolio shortlist lock
- `current_gate`: `blocked by release-review / shortlist unresolved`

### T175. portfolio handoff atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T163 + T173 + T174`
- `hypothesis`: 需要 portfolio atlas 才能多年期无损交接
- `asset_requirement`: atlas template、portfolio map、roadmap packet、shortlist
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；roadmap/shortlist 冲突；交接不完整
- `expected_artifact`: portfolio handoff atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T176. portfolio scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T164 + T173`
- `hypothesis`: 需要 portfolio scoreboards 把 campaign/horizon/slot 指标统一起来
- `asset_requirement`: scoreboard schema、portfolio metrics、existing scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 冲突；schema 不稳；scoreboards 无法聚合
- `expected_artifact`: portfolio scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T177. portfolio predecessor canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T165 + T175`
- `hypothesis`: predecessor graph 需要升级成 portfolio canon，后续才无需重建依赖
- `asset_requirement`: predecessor handbook、portfolio atlas、edge canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: edge canon 仍漂移；atlas 不稳；handbook 无法对齐
- `expected_artifact`: portfolio predecessor canon
- `current_gate`: `blocked by release-review / predecessor canon unresolved`

### T178. portfolio annual-review simulation 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T173-T177`
- `hypothesis`: 需要年度 review simulation，但仍不能误写成 release review
- `asset_requirement`: annual-review template、negative boundary set、portfolio scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 容易被误读成 release；negative boundary 不够硬；scoreboards 不闭环
- `expected_artifact`: annual-review simulation packet
- `current_gate`: `blocked by release-review / simulation unresolved`

### T179. portfolio no-release proof 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T172 + T178`
- `hypothesis`: 需要 portfolio 级 no-release proof，彻底压住误放行风险
- `asset_requirement`: no-release proof template、anti-release charter、review simulation
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: proof 仍可误读；charter 不稳；simulation 不闭环
- `expected_artifact`: portfolio no-release proof
- `current_gate`: `blocked by release-review / no-release proof unresolved`

### T180. portfolio frozen blocked book 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T169-T179`
- `hypothesis`: 需要一本 portfolio 级 frozen blocked book，避免继续发散
- `asset_requirement`: frozen template、portfolio packets、resume fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需扩表；template 漂移；resume 字段不够
- `expected_artifact`: portfolio frozen blocked book
- `current_gate`: `blocked by release-review / book unresolved`

### T181. canon index packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T89 + T180`
- `hypothesis`: 需要一个最终 canon index，让未来检索不依赖重新阅读全局
- `asset_requirement`: index template、packet maps、portfolio book
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: index 不完整；映射漂移；检索无效
- `expected_artifact`: canon index
- `current_gate`: `blocked by release-review / index unresolved`

### T182. canon glossary packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T181`
- `hypothesis`: 需要 glossary 才能冻结 190+ 项 planning 语义
- `asset_requirement`: glossary template、term bindings、claim bindings
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: term 漂移；bindings 不稳；glossary 无法支撑 handoff
- `expected_artifact`: canon glossary
- `current_gate`: `blocked by release-review / glossary unresolved`

### T183. canon owner matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T181 + T182`
- `hypothesis`: 长期 blocked 体系需要 owner matrix，后续才可安全接手
- `asset_requirement`: owner fields、role matrix、handoff matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: owner 未定义；role matrix 冲突；handoff 无法执行
- `expected_artifact`: canon owner matrix
- `current_gate`: `blocked by release-review / owner matrix unresolved`

### T184. canon approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T183 + T90`
- `hypothesis`: approval matrix 需要从 future simulation 一侧冻结下来
- `asset_requirement`: approval fields、decision tree、owner matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: approval matrix 不闭环；owner 未定义；decision tree 冲突
- `expected_artifact`: canon approval matrix
- `current_gate`: `blocked by release-review / approval matrix unresolved`

### T185. canon retention matrix 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T149 + T181`
- `hypothesis`: retention 规则要从 annual 级升级到 canon 级
- `asset_requirement`: retention fields、escrow policy、index packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: retention 仍不稳；escrow policy 冲突；index 不完整
- `expected_artifact`: canon retention matrix
- `current_gate`: `blocked by release-review / retention unresolved`

### T186. canon scoreboard packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T176 + T181`
- `hypothesis`: 需要一个总 scoreboard 来冻结 190+ 项 blocked 计划的整体导航
- `asset_requirement`: global metrics、portfolio scoreboards、index packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；scoreboards 冲突；导航无效
- `expected_artifact`: canon scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T187. canon route map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T94 + T141 + T181`
- `hypothesis`: route map 需要从 predecessor/horizon/quarter 层统一冻结
- `asset_requirement`: route map template、execution maps、graph handbook
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: route map 仍不稳；execution maps 冲突；graph handbook 不闭环
- `expected_artifact`: canon route map
- `current_gate`: `blocked by release-review / route map unresolved`

### T188. canon anti-drift proof 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T148 + T179 + T186`
- `hypothesis`: 需要最终 canon 级 anti-drift proof，防止任何后续 wording 越界
- `asset_requirement`: anti-drift packet、no-release proof、global scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: proof 仍薄弱；scoreboards 无法支撑；drift 风险未消除
- `expected_artifact`: canon anti-drift proof
- `current_gate`: `blocked by release-review / anti-drift proof unresolved`

### T189. canon resume bundle 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T116 + T155 + T163 + T187`
- `hypothesis`: 需要最终 resume bundle，确保后续完全不重分析全局
- `asset_requirement`: resume fields、handoff packets、route map、atlas
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: resume 仍不完整；atlas/route map 冲突；bundle 不可交接
- `expected_artifact`: canon resume bundle
- `current_gate`: `blocked by release-review / resume bundle unresolved`

### T190. canon freeze checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T181-T189`
- `hypothesis`: 最终冻结前需要一套 checklist，防止漏项
- `asset_requirement`: freeze checklist、required packets、owner matrix、approval matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不够全；required packets 缺失；owners/approvals 冲突
- `expected_artifact`: canon freeze checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T191. canon freeze rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T190`
- `hypothesis`: 在真正冻结前需要一次 rehearsal，确认所有字段可交接
- `asset_requirement`: rehearsal template、checklist、resume bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；resume bundle 不稳；checklist 不闭环
- `expected_artifact`: canon freeze rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T192. final frozen planning canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T191`
- `hypothesis`: 需要最终 frozen planning canon，后续轮次不再继续扩表
- `asset_requirement`: final canon template、resume bundle、hard boundaries、freeze checklist
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；final template 不稳；freeze checklist 未通过
- `expected_artifact`: final frozen planning canon
- `current_gate`: `blocked by release-review / canon not frozen`

### T193. portfolio audit map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T169-T192`
- `hypothesis`: 需要一份 portfolio audit map，把 200+ blocked planning 变成可审计对象
- `asset_requirement`: audit map template、portfolio canon、campaign maps
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: audit map 不稳；canon 漂移；campaign maps 不闭环
- `expected_artifact`: portfolio audit map
- `current_gate`: `blocked by release-review / audit map unresolved`

### T194. portfolio evidence ledger 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T83 + T89 + T118 + T181`
- `hypothesis`: 长期 blocked 体系需要 evidence ledger，后续才能防止证据丢失
- `asset_requirement`: ledger schema、packet index、artifact taxonomy、all-wave sync
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: ledger 不完整；schema 漂移；packet index 冲突
- `expected_artifact`: portfolio evidence ledger
- `current_gate`: `blocked by release-review / ledger unresolved`

### T195. portfolio claim ledger 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T54 + T84 + T182`
- `hypothesis`: claims 需要 ledger 化，后续才不会被不同轮次重复改写
- `asset_requirement`: claim schema、glossary、claim bindings
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: claim schema 不稳；glossary 不闭环；bindings 冲突
- `expected_artifact`: portfolio claim ledger
- `current_gate`: `blocked by release-review / claim ledger unresolved`

### T196. portfolio control matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T183 + T184 + T190`
- `hypothesis`: owner/approval/freeze checklist 需要组合成 control matrix
- `asset_requirement`: owner matrix、approval matrix、freeze checklist
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: matrix 不闭环；owners/approvals 冲突；checklist 不稳
- `expected_artifact`: portfolio control matrix
- `current_gate`: `blocked by release-review / control matrix unresolved`

### T197. portfolio boundary pack 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T117 + T160 + T172 + T188`
- `hypothesis`: 所有 hard boundary 需要独立收口成 boundary pack
- `asset_requirement`: hard-boundary set、anti-release proofs、anti-drift proofs
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: boundary 不够硬；proof 相互冲突；pack 可被误读
- `expected_artifact`: portfolio boundary pack
- `current_gate`: `blocked by release-review / boundary pack unresolved`

### T198. portfolio route-control atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T187 + T193 + T196`
- `hypothesis`: route map 和 control matrix 需要合成 atlas 才能长期导航
- `asset_requirement`: route map、audit map、control matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 无法导航；route/control 冲突；audit 不稳
- `expected_artifact`: portfolio route-control atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T199. portfolio retention charter 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T149 + T185 + T194`
- `hypothesis`: retention/escrow 需要 charter 级收口，后续才能长期保存 blocked 证据
- `asset_requirement`: retention matrix、ledger、escrow policy
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 过弱；ledger 不完整；policy 冲突
- `expected_artifact`: retention charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T200. portfolio review charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T53 + T81 + T90 + T178`
- `hypothesis`: future review simulation 必须冻结成 review charter
- `asset_requirement`: review template、slot matrix、decision tree、simulation packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 仍像 release；slot/decision tree 冲突；simulation 包不闭环
- `expected_artifact`: review charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T201. portfolio freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T190 + T191 + T192 + T196`
- `hypothesis`: canon freeze 必须升级成 charter，后续才不会再次扩表
- `asset_requirement`: freeze charter template、rehearsal、control matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不稳；rehearsal 未通过；control matrix 冲突
- `expected_artifact`: freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T202. portfolio audit checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T193 + T197 + T200 + T201`
- `hypothesis`: 需要终版 audit checklist 才能防止未来 drift 回流
- `asset_requirement`: audit template、boundary pack、review charter、freeze charter
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；charter 冲突；boundary pack 不闭环
- `expected_artifact`: final audit checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T203. portfolio handoff doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T116 + T163 + T189 + T198`
- `hypothesis`: 需要 doctrine 级 handoff 文档，保证多轮接手不损失状态
- `asset_requirement`: handoff atlas、resume bundle、route-control atlas
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不完整；atlas/bundle 冲突；不可执行
- `expected_artifact`: portfolio handoff doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T204. final portfolio audit canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T193-T203`
- `hypothesis`: 需要一本 portfolio audit canon，把 blocked planning 彻底压成可审计对象
- `asset_requirement`: canon template、audit map、control matrix、doctrine、charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: canon 不稳；charters 冲突；doctrine 不可执行
- `expected_artifact`: portfolio audit canon
- `current_gate`: `blocked by release-review / audit canon unresolved`

### T205. canon freeze enforcement map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T201 + T204`
- `hypothesis`: 最终 frozen planning canon 需要 enforcement map
- `asset_requirement`: enforcement template、freeze charter、audit canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: enforcement template 不稳；charter/canon 冲突；不可执行
- `expected_artifact`: freeze enforcement map
- `current_gate`: `blocked by release-review / enforcement map unresolved`

### T206. canon freeze SLA 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T205`
- `hypothesis`: 需要 SLA 约束 future updates，防止再度无边界扩表
- `asset_requirement`: SLA template、enforcement map、owner matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: SLA 不可执行；owner 未定义；enforcement map 不闭环
- `expected_artifact`: freeze SLA
- `current_gate`: `blocked by release-review / SLA unresolved`

### T207. canon freeze exception matrix 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T206`
- `hypothesis`: 所有可能例外情况都应先定义，防止例外成为后门
- `asset_requirement`: exception template、SLA、owner matrix、approval matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: exception 范围过宽；SLA 不稳；approval 规则不闭环
- `expected_artifact`: exception matrix
- `current_gate`: `blocked by release-review / exception matrix unresolved`

### T208. canon freeze escalation matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T207`
- `hypothesis`: 需要 escalation matrix 处理 future conflicts，但不能触发 release
- `asset_requirement`: escalation template、exception matrix、review charter
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: escalation 像 release 流程；template 含糊；charter 冲突
- `expected_artifact`: escalation matrix
- `current_gate`: `blocked by release-review / escalation unresolved`

### T209. canon freeze monitoring spec 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T91 + T199`
- `hypothesis`: frozen canon 也需要 monitoring spec，但只是 planning 监控，不是运行监控
- `asset_requirement`: monitoring template、heartbeat sim、retention charter
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: spec 像真实运行监控；template 不稳；charter 冲突
- `expected_artifact`: monitoring spec
- `current_gate`: `blocked by release-review / monitoring spec unresolved`

### T210. canon freeze telemetry schema 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T209`
- `hypothesis`: telemetry 语义必须先冻结，未来才不会反复改 schema
- `asset_requirement`: telemetry schema、monitoring spec、artifact taxonomy
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 漂移；spec 不稳；taxonomy 冲突
- `expected_artifact`: telemetry schema
- `current_gate`: `blocked by release-review / telemetry unresolved`

### T211. canon freeze compliance pack 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T202 + T206 + T207 + T208 + T210`
- `hypothesis`: frozen canon 最终需要 compliance pack，才能真正阻止 drift
- `asset_requirement`: compliance template、audit checklist、SLA、exception matrix、telemetry schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: compliance 不闭环；matrix 冲突；template 不稳
- `expected_artifact`: compliance pack
- `current_gate`: `blocked by release-review / compliance unresolved`

### T212. canon freeze dry-run 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T211`
- `hypothesis`: 在真正冻结前，需要一次 dry-run 验证 canon 完整性
- `asset_requirement`: dry-run template、compliance pack、resume bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: dry-run 不通过；compliance 不稳；resume bundle 冲突
- `expected_artifact`: freeze dry-run packet
- `current_gate`: `blocked by release-review / dry-run unresolved`

### T213. canon freeze verdict packet 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T212`
- `hypothesis`: 最终需要一份明确 verdict packet：为什么仍 blocked、为什么仍 no-release
- `asset_requirement`: verdict template、dry-run packet、no-release proofs
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 仍可被误解；template 不闭环；proof 冲突
- `expected_artifact`: freeze verdict packet
- `current_gate`: `blocked by release-review / verdict unresolved`

### T214. canon freeze registry 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T181-T213`
- `hypothesis`: 需要一份 registry 记录所有 canon 级别产物
- `asset_requirement`: registry schema、index packet、ledger、verdict packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: registry 缺字段；ledger 不闭环；verdict 不稳
- `expected_artifact`: freeze registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T215. canon freeze handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T214`
- `hypothesis`: 需要一份 handbook 才能让后续轮次只读不重写
- `asset_requirement`: handbook template、registry、route atlas、doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: handbook 不稳；registry 冲突；仍需继续扩表
- `expected_artifact`: freeze handbook
- `current_gate`: `blocked by release-review / handbook unresolved`

### T216. final frozen planning canon v2 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T215`
- `hypothesis`: 需要最终 v2 版 frozen planning canon，彻底结束继续扩表
- `asset_requirement`: v2 canon template、freeze handbook、hard boundaries、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；v2 template 不稳；registry/handbook 不闭环
- `expected_artifact`: final frozen planning canon v2
- `current_gate`: `blocked by release-review / canon v2 not frozen`

### T217. portfolio canon audit map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T193-T216`
- `hypothesis`: 需要一份 canon audit map，把 240 项 blocked planning 变成长期可审计对象
- `asset_requirement`: canon audit template、planning canon v2、portfolio audit canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: audit map 不稳；canon 冲突；模板不可执行
- `expected_artifact`: canon audit map
- `current_gate`: `blocked by release-review / audit map unresolved`

### T218. portfolio canon registry 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T214 + T217`
- `hypothesis`: 需要 registry 级别冻结 canon 产物，后续才可只读
- `asset_requirement`: registry schema、canon audit map、existing ledgers
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 漂移；registry 不完整；ledger 冲突
- `expected_artifact`: canon registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T219. portfolio canon glossary 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T182 + T218`
- `hypothesis`: terminology 需要最终 canon glossary，后续才不会反复改名词
- `asset_requirement`: glossary template、term bindings、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: term 漂移；bindings 不稳；registry 不闭环
- `expected_artifact`: canon glossary final
- `current_gate`: `blocked by release-review / glossary unresolved`

### T220. portfolio canon owner-doctrine matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T183 + T196 + T203`
- `hypothesis`: owner/approval/handoff 需要收敛成 owner-doctrine matrix
- `asset_requirement`: owner matrix、control matrix、handoff doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 矩阵不闭环；owners 冲突；handoff 仍模糊
- `expected_artifact`: owner-doctrine matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T221. portfolio canon route-doctrine atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T198 + T203 + T217`
- `hypothesis`: route atlas 和 doctrine 需要合并，后续才可真正导航
- `asset_requirement`: atlas template、route-control atlas、handoff doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；doctrine 不稳；映射冲突
- `expected_artifact`: route-doctrine atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T222. portfolio canon scoreboards v2 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T176 + T186 + T218`
- `hypothesis`: 需要统一 v2 scoreboards，避免多个 scoreboard 漂移
- `asset_requirement`: scoreboard schema、existing scoreboards、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 不稳；旧 scoreboards 冲突；registry 不闭环
- `expected_artifact`: scoreboards v2
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T223. portfolio canon no-release doctrine 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T172 + T179 + T213`
- `hypothesis`: 需要把 no-release 证明升级成 doctrine，避免未来误放行
- `asset_requirement`: doctrine template、anti-release proof、verdict packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 仍可误读；proof 不闭环；verdict 漂移
- `expected_artifact`: no-release doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T224. portfolio canon quarterly-yearly bridge 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T133-T137 + T161`
- `hypothesis`: quarter/year/multi-year 需要桥接层，否则 roadmap 会断裂
- `asset_requirement`: bridge template、quarter plans、year roadmap、multi-year roadmap
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: bridge 不闭环；roadmap 冲突；template 不稳
- `expected_artifact`: quarter-year bridge packet
- `current_gate`: `blocked by release-review / bridge unresolved`

### T225. portfolio canon horizon-campaign bridge 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T125-T128 + T121-T124 + T169-T180`
- `hypothesis`: horizons 和 campaigns 需要一个终版桥接层
- `asset_requirement`: horizon/campaign maps、bridge schema、scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: bridge 不稳；maps 冲突；scoreboards 不可用
- `expected_artifact`: horizon-campaign bridge
- `current_gate`: `blocked by release-review / bridge unresolved`

### T226. portfolio canon slot-campaign bridge 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T130 + T131 + T147 + T171`
- `hypothesis`: slots 和 campaigns 也需要终版桥接，否则 future simulation 仍会漂移
- `asset_requirement`: slot ladder、campaign maps、bridge schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: bridge 不闭环；slot ladder 冲突；campaign map 漂移
- `expected_artifact`: slot-campaign bridge
- `current_gate`: `blocked by release-review / bridge unresolved`

### T227. portfolio canon exception doctrine 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T207 + T208 + T226`
- `hypothesis`: exception/escalation 需要 doctrine 级约束，避免未来变成 release 通道
- `asset_requirement`: exception matrix、escalation matrix、bridge packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 仍像 release；matrices 冲突；bridge 不稳
- `expected_artifact`: exception doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T228. final portfolio canon audit charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T193-T227`
- `hypothesis`: portfolio canon 最终需要 audit charter 约束
- `asset_requirement`: charter template、audit packets、bridges、doctrines
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不稳；doctrines 冲突；审计包不闭环
- `expected_artifact`: portfolio audit charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T229. frozen operations doctrine draft 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T204 + T228`
- `hypothesis`: 最终需要 operations doctrine draft，把 blocked planning 变成长期操作章程
- `asset_requirement`: doctrine template、audit charter、portfolio audit canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: draft 不稳；canon/charter 冲突；不可执行
- `expected_artifact`: operations doctrine draft
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T230. frozen operations doctrine review 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T229`
- `hypothesis`: operations doctrine 需要单独 review 层，但不能等于 release review
- `asset_requirement`: doctrine draft、review template、negative boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: review 仍像 release；negative boundaries 不够硬；draft 不闭环
- `expected_artifact`: operations doctrine review
- `current_gate`: `blocked by release-review / review unresolved`

### T231. frozen operations doctrine audit 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T230`
- `hypothesis`: operations doctrine 需要审计层确认，后续才可冻结
- `asset_requirement`: doctrine review、audit checklist、compliance pack
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 审计不通过；checklist 不全；review 不稳
- `expected_artifact`: operations doctrine audit
- `current_gate`: `blocked by release-review / audit unresolved`

### T232. frozen operations doctrine scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T231`
- `hypothesis`: operations doctrine 也需要 scoreboard 才能长期导航
- `asset_requirement`: scoreboard schema、audit packet、existing scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 不稳；audit 不闭环；导航无效
- `expected_artifact`: operations doctrine scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T233. frozen operations doctrine atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T232 + T198 + T221`
- `hypothesis`: operations doctrine 需要 atlas，后续才可长期接手
- `asset_requirement`: atlas template、doctrine packets、route atlas
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；doctrine 不稳；route atlas 冲突
- `expected_artifact`: operations doctrine atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T234. frozen operations doctrine handoff 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T233 + T203`
- `hypothesis`: 需要一个最终 operations doctrine handoff，后续轮次才不会再扩表
- `asset_requirement`: handoff template、doctrine atlas、resume fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: handoff 仍需重分析；atlas 不稳；resume 字段缺失
- `expected_artifact`: operations doctrine handoff
- `current_gate`: `blocked by release-review / handoff unresolved`

### T235. frozen operations doctrine freeze checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T234`
- `hypothesis`: 需要一个最终 checklist 确认 operations doctrine 已可冻结
- `asset_requirement`: checklist template、handoff packet、audit packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；handoff/audit 冲突；冻结条件不闭环
- `expected_artifact`: operations doctrine freeze checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T236. frozen operations doctrine freeze rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T235`
- `hypothesis`: 最终冻结前需要一轮 rehearsal，防止冻结后仍需改结构
- `asset_requirement`: rehearsal template、freeze checklist、handoff packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；checklist 不通过；handoff 不稳
- `expected_artifact`: operations doctrine rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T237. frozen operations doctrine verdict 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T236`
- `hypothesis`: 需要一份最终 verdict，说明为何结构已冻结但执行仍 blocked
- `asset_requirement`: verdict template、rehearsal packet、no-release doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 含糊；doctrine 不稳；rehearsal 失败
- `expected_artifact`: operations doctrine verdict
- `current_gate`: `blocked by release-review / verdict unresolved`

### T238. frozen operations doctrine registry 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T237`
- `hypothesis`: 需要 registry 把 operations doctrine 级产物记录下来
- `asset_requirement`: registry schema、verdict packet、atlas、handoff
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: registry 缺字段；verdict 不稳；atlas 冲突
- `expected_artifact`: operations doctrine registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T239. final frozen operations doctrine handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T238`
- `hypothesis`: 需要一本 handbook 才能让 future rounds 彻底停止扩结构
- `asset_requirement`: handbook template、registry、verdict、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: handbook 不稳；registry/verdict 不闭环；边界不够硬
- `expected_artifact`: operations doctrine handbook
- `current_gate`: `blocked by release-review / handbook unresolved`

### T240. final frozen operations doctrine canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T239`
- `hypothesis`: 需要最终 operations doctrine canon，后续轮次不再继续扩 planning 结构
- `asset_requirement`: canon template、handbook、registry、resume fields、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；canon 不稳；hard boundaries/resume fields 不完整
- `expected_artifact`: final frozen operations doctrine canon
- `current_gate`: `blocked by release-review / canon not frozen`

### T241. canon integrity map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T181-T240`
- `hypothesis`: 需要一份 integrity map 确认 canon 所有层都已闭环
- `asset_requirement`: integrity template、registries、ledgers、handbooks
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: map 不闭环；registries 冲突；handbooks 不稳
- `expected_artifact`: canon integrity map
- `current_gate`: `blocked by release-review / integrity map unresolved`

### T242. canon integrity ledger 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T194 + T214 + T241`
- `hypothesis`: 需要一份终版 integrity ledger 记录所有冻结对象
- `asset_requirement`: ledger schema、registries、integrity map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: ledger 缺字段；schema 漂移；integrity map 不稳
- `expected_artifact`: integrity ledger
- `current_gate`: `blocked by release-review / ledger unresolved`

### T243. canon integrity glossary 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T182 + T219 + T242`
- `hypothesis`: 最终 glossary 需要与 integrity ledger 严格绑定
- `asset_requirement`: glossary template、term bindings、integrity ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 术语不稳；bindings 冲突；ledger 不闭环
- `expected_artifact`: integrity glossary
- `current_gate`: `blocked by release-review / glossary unresolved`

### T244. canon integrity owner matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T183 + T220 + T242`
- `hypothesis`: owner 责任边界必须在冻结前完全显式
- `asset_requirement`: owner schema、doctrine matrix、integrity ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: owner 缺失；矩阵冲突；ledger 不稳
- `expected_artifact`: integrity owner matrix
- `current_gate`: `blocked by release-review / owner matrix unresolved`

### T245. canon integrity approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T184 + T244`
- `hypothesis`: approval 语义必须在冻结前完全定稿
- `asset_requirement`: approval schema、owner matrix、decision tree
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: approval 含糊；owner 冲突；decision tree 漂移
- `expected_artifact`: integrity approval matrix
- `current_gate`: `blocked by release-review / approval matrix unresolved`

### T246. canon integrity route map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T187 + T221 + T241`
- `hypothesis`: 需要 route map 把 atlas/graph/roadmap 完全统一
- `asset_requirement`: route template、atlas、graph handbook、roadmaps
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: route 图不稳；atlas/graph 冲突；roadmap 不闭环
- `expected_artifact`: integrity route map
- `current_gate`: `blocked by release-review / route map unresolved`

### T247. canon integrity retention matrix 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T185 + T199 + T242`
- `hypothesis`: retention 需要升级成 integrity 级 matrix
- `asset_requirement`: retention schema、charters、integrity ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: retention 不稳；charters 冲突；ledger 不闭环
- `expected_artifact`: integrity retention matrix
- `current_gate`: `blocked by release-review / retention unresolved`

### T248. canon integrity scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T186 + T222 + T242`
- `hypothesis`: 需要一个 integrity scoreboard 统一衡量冻结程度
- `asset_requirement`: metrics schema、existing scoreboards、integrity ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；scoreboards 冲突；ledger 不闭环
- `expected_artifact`: integrity scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T249. canon integrity doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T223 + T229-T240`
- `hypothesis`: no-release doctrine 与 operations doctrine 需要统一成 integrity doctrine
- `asset_requirement`: doctrine template、no-release doctrine、operations doctrine canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 冲突；template 不稳；canon 不闭环
- `expected_artifact`: integrity doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T250. canon integrity audit checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T202 + T211 + T241-T249`
- `hypothesis`: 最终 audit checklist 需要升级成 integrity 版本
- `asset_requirement`: checklist schema、compliance pack、integrity packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；integrity 包冲突；compliance 不稳
- `expected_artifact`: integrity audit checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T251. canon integrity rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T212 + T250`
- `hypothesis`: 需要一次完整 integrity rehearsal，确认所有冻结层可协同
- `asset_requirement`: rehearsal template、audit checklist、integrity doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；doctrine 不稳；checklist 不通过
- `expected_artifact`: integrity rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T252. final canon integrity charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T241-T251`
- `hypothesis`: 需要一份最终 integrity charter 作为冻结总边界
- `asset_requirement`: charter template、integrity packets、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不稳；packets 冲突；边界不够硬
- `expected_artifact`: integrity charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T253. blocked universe map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T169-T252`
- `hypothesis`: 需要一张 blocked universe map，把 240+ 规划统一成一个宇宙图
- `asset_requirement`: universe template、all maps、all charters、all ledgers
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: universe 图不稳；多图冲突；模板不可读
- `expected_artifact`: blocked universe map
- `current_gate`: `blocked by release-review / universe map unresolved`

### T254. blocked universe ledger 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T242 + T253`
- `hypothesis`: 需要一份 universe ledger 收口所有对象
- `asset_requirement`: ledger template、universe map、registries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: ledger 不全；registries 冲突；universe map 不稳
- `expected_artifact`: blocked universe ledger
- `current_gate`: `blocked by release-review / ledger unresolved`

### T255. blocked universe doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T249 + T252 + T253`
- `hypothesis`: 需要一份最终 doctrine，把 no-release / operations / integrity 全部统一
- `asset_requirement`: doctrine template、integrity charter、operations doctrine canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不闭环；charter 冲突；canon 不稳
- `expected_artifact`: blocked universe doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T256. blocked universe owner-approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T244 + T245 + T255`
- `hypothesis`: universe 级 owner/approval 需要完全冻结
- `asset_requirement`: owner/approval schemas、doctrine、universe ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: owner/approval 不稳；schema 冲突；ledger 不闭环
- `expected_artifact`: universe owner-approval matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T257. blocked universe scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T248 + T254`
- `hypothesis`: 需要一组 universe scoreboards 作为最终导航面
- `asset_requirement`: scoreboard template、universe ledger、integrity metrics
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: scoreboards 冲突；metrics 不稳；ledger 不闭环
- `expected_artifact`: blocked universe scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T258. blocked universe route atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T246 + T253`
- `hypothesis`: 需要 route atlas 作为 universe 导航终版
- `asset_requirement`: atlas template、route maps、universe map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；maps 冲突；模板不稳
- `expected_artifact`: blocked universe route atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T259. blocked universe audit pack 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T250 + T252 + T257`
- `hypothesis`: 需要一份 universe audit pack 证明 blocked 体系完全封口
- `asset_requirement`: audit template、charter、scoreboards、checklists
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 审计不闭环；scoreboards 不稳；charter 冲突
- `expected_artifact`: blocked universe audit pack
- `current_gate`: `blocked by release-review / audit unresolved`

### T260. blocked universe handoff doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T203 + T234 + T258`
- `hypothesis`: 需要一份最终 handoff doctrine，后续轮次不再重分析
- `asset_requirement`: doctrine template、route atlas、handoff packets、resume bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不稳；atlas/packets 冲突；resume 不完整
- `expected_artifact`: blocked universe handoff doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T261. blocked universe freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T252 + T255 + T259`
- `hypothesis`: 需要一份最终 freeze charter，限制任何后续结构变更
- `asset_requirement`: charter template、doctrine、audit pack
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不够硬；doctrine 不闭环；audit 未通过
- `expected_artifact`: blocked universe freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T262. blocked universe freeze registry 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T254 + T261`
- `hypothesis`: 需要 registry 记录冻结后的全部 universe 对象
- `asset_requirement`: registry schema、universe ledger、freeze charter
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: registry 缺字段；ledger 冲突；charter 不稳
- `expected_artifact`: blocked universe freeze registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T263. blocked universe freeze handbook 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T262`
- `hypothesis`: 需要最终 handbook 让 future rounds 只读不扩
- `asset_requirement`: handbook template、freeze registry、horizon/campaign maps、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: handbook 不稳；registry 不闭环；仍需扩结构
- `expected_artifact`: blocked universe freeze handbook
- `current_gate`: `blocked by release-review / handbook unresolved`

### T264. final frozen blocked universe canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T263`
- `hypothesis`: 需要最终 frozen blocked universe canon，彻底停止继续扩表
- `asset_requirement`: canon template、freeze handbook、registry、resume fields、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需继续扩表；canon 不稳；resume/boundaries 不完整
- `expected_artifact`: final frozen blocked universe canon
- `current_gate`: `blocked by release-review / canon not frozen`

### T265. universe audit doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T193-T264`
- `hypothesis`: blocked universe 需要独立的 audit doctrine 才能长期维持一致性
- `asset_requirement`: audit doctrine template、universe canon、audit pack
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不稳；canon/pack 冲突；模板不可执行
- `expected_artifact`: universe audit doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T266. universe audit registry 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T214 + T254 + T265`
- `hypothesis`: 需要 registry 记录所有 universe 级冻结对象与审计对象
- `asset_requirement`: registry schema、universe ledger、audit doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: registry 缺字段；schema 漂移；doctrine 不闭环
- `expected_artifact`: universe audit registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T267. universe audit glossary 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T243 + T266`
- `hypothesis`: 最终 universe 语义需要 glossary 收口，避免跨轮次漂移
- `asset_requirement`: glossary template、registry、term bindings
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: glossary 漂移；bindings 冲突；registry 不稳
- `expected_artifact`: universe audit glossary
- `current_gate`: `blocked by release-review / glossary unresolved`

### T268. universe audit owner matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T244 + T256 + T266`
- `hypothesis`: universe 级 owner 责任边界必须最终定稿
- `asset_requirement`: owner schema、audit registry、owner matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: owner 不明；schema 冲突；registry 不闭环
- `expected_artifact`: universe audit owner matrix
- `current_gate`: `blocked by release-review / owner matrix unresolved`

### T269. universe audit approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T245 + T268`
- `hypothesis`: approval 语义需要在 universe 层冻结
- `asset_requirement`: approval schema、owner matrix、decision tree
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: approval 含糊；owner 冲突；decision tree 漂移
- `expected_artifact`: universe audit approval matrix
- `current_gate`: `blocked by release-review / approval matrix unresolved`

### T270. universe route-doctrine atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T198 + T221 + T258`
- `hypothesis`: route/atlas/doctrine 需要在 universe 层统一
- `asset_requirement`: atlas template、route maps、doctrines
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；doctrine 冲突；route map 不闭环
- `expected_artifact`: universe route-doctrine atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T271. universe doctrine scoreboard 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T248 + T257 + T266`
- `hypothesis`: 需要统一 scoreboard 观察 universe 冻结状态
- `asset_requirement`: metrics schema、registries、existing scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 冲突；registries 不稳；scoreboards 无法聚合
- `expected_artifact`: universe doctrine scoreboard
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T272. universe no-release charter 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T172 + T179 + T223 + T265`
- `hypothesis`: no-release 需要在 universe 层再次冻结
- `asset_requirement`: charter template、anti-release doctrine、audit doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 仍不够硬；doctrines 冲突；可被误解
- `expected_artifact`: universe no-release charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T273. universe audit checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T202 + T250 + T259 + T272`
- `hypothesis`: 最终 audit checklist 需要 universe 版本
- `asset_requirement`: checklist template、charters、scoreboards、registries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；charters 不闭环；registries 冲突
- `expected_artifact`: universe audit checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T274. universe handoff doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T203 + T260 + T270`
- `hypothesis`: 需要最终 universe handoff doctrine 保证超长期接手
- `asset_requirement`: doctrine template、route-doctrine atlas、handoff packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不稳；atlas 冲突；handoff 仍需重分析
- `expected_artifact`: universe handoff doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T275. universe freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T261 + T272 + T273`
- `hypothesis`: 最终 universe 需要 freeze charter 约束所有未来变更
- `asset_requirement`: freeze charter、no-release charter、audit checklist
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不硬；audit 不闭环；未来变更条件不清
- `expected_artifact`: universe freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T276. final universe audit canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T265-T275`
- `hypothesis`: 需要一份最终 universe audit canon 作为 Wave 24 总收口
- `asset_requirement`: canon template、audit packets、freeze charter、doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: canon 不稳；packets 冲突；doctrine 不闭环
- `expected_artifact`: universe audit canon
- `current_gate`: `blocked by release-review / canon unresolved`

### T277. frozen constitution draft 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T216 + T240 + T264 + T276`
- `hypothesis`: 需要 draft 级 constitution 才能停止继续扩表
- `asset_requirement`: constitution template、existing canons、audit canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: draft 不稳；canons 冲突；模板不可执行
- `expected_artifact`: frozen constitution draft
- `current_gate`: `blocked by release-review / draft unresolved`

### T278. frozen constitution review 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T277`
- `hypothesis`: constitution 需要 review，但不能变成 release review
- `asset_requirement`: review template、constitution draft、negative boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: review 像 release；draft 不闭环；边界不够硬
- `expected_artifact`: constitution review
- `current_gate`: `blocked by release-review / review unresolved`

### T279. frozen constitution audit 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T278`
- `hypothesis`: constitution 需要 audit 才能冻结
- `asset_requirement`: audit template、review packet、audit checklists
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 审计不通过；review 不稳；checklist 冲突
- `expected_artifact`: constitution audit
- `current_gate`: `blocked by release-review / audit unresolved`

### T280. frozen constitution scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T279`
- `hypothesis`: constitution 需要 scoreboards 才能长期导航
- `asset_requirement`: scoreboard schema、constitution audit、global metrics
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；audit 不闭环；scoreboards 无效
- `expected_artifact`: constitution scoreboards
- `current_gate`: `blocked by release-review / scoreboard unresolved`

### T281. frozen constitution atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T233 + T258 + T280`
- `hypothesis`: constitution 需要 atlas 作为最终导航层
- `asset_requirement`: atlas template、route maps、constitution scoreboards
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；scoreboards 冲突；maps 不闭环
- `expected_artifact`: constitution atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T282. frozen constitution owner matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T244 + T256 + T279`
- `hypothesis`: constitution 层需要最终 owner 责任边界
- `asset_requirement`: owner schema、audit packet、existing matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: owner 不明；matrices 冲突；audit 不闭环
- `expected_artifact`: constitution owner matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T283. frozen constitution approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T245 + T282`
- `hypothesis`: approval 规则需要在 constitution 层完全冻结
- `asset_requirement`: approval schema、owner matrix、decision tree
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: approval 含糊；owner 冲突；decision tree 漂移
- `expected_artifact`: constitution approval matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T284. frozen constitution handoff bundle 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T189 + T234 + T274 + T281`
- `hypothesis`: 需要终版 handoff bundle 让 future rounds 彻底只读
- `asset_requirement`: handoff packets、atlases、resume fields、doctrines
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: bundle 不稳；atlases/doctrines 冲突；resume 不完整
- `expected_artifact`: constitution handoff bundle
- `current_gate`: `blocked by release-review / bundle unresolved`

### T285. frozen constitution checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T190 + T202 + T273 + T284`
- `hypothesis`: 需要一份终版 checklist，确定 constitution 可冻结
- `asset_requirement`: checklist template、handoff bundle、audit packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；bundle 不稳；audit 冲突
- `expected_artifact`: constitution checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T286. frozen constitution rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T191 + T285`
- `hypothesis`: 冻结前需要一次最终 rehearsal，避免 frozen 后仍需改结构
- `asset_requirement`: rehearsal template、constitution checklist、handoff bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；checklist 不通过；bundle 冲突
- `expected_artifact`: constitution rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T287. frozen constitution verdict 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T286`
- `hypothesis`: 需要一份最终 verdict，明确为什么 planning 结构到此冻结
- `asset_requirement`: verdict template、rehearsal packet、charters、doctrines
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 含糊；charters/doctrines 不稳；rehearsal 失败
- `expected_artifact`: constitution verdict
- `current_gate`: `blocked by release-review / verdict unresolved`

### T288. final frozen planning constitution 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T287`
- `hypothesis`: 需要最终 frozen planning constitution，后续轮次彻底不再继续扩结构
- `asset_requirement`: constitution template、verdict、handoff bundle、hard boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需扩结构；verdict 不稳；bundle/boundaries 不完整
- `expected_artifact`: final frozen planning constitution
- `current_gate`: `blocked by release-review / constitution not frozen`

### T289. universe governance map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T193-T288`
- `hypothesis`: 需要一份 universe governance map，把各层治理对象完全对齐
- `asset_requirement`: governance map template、existing canons、doctrines、charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: map 不闭环；canons 冲突；doctrines 漂移
- `expected_artifact`: universe governance map
- `current_gate`: `blocked by release-review / map unresolved`

### T290. universe governance ledger 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T194 + T218 + T242 + T254 + T289`
- `hypothesis`: 需要一个 governance ledger 统一记录所有冻结治理对象
- `asset_requirement`: ledger schema、maps、registries、existing ledgers
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: ledger 冲突；schema 不稳；maps 不闭环
- `expected_artifact`: universe governance ledger
- `current_gate`: `blocked by release-review / ledger unresolved`

### T291. universe governance glossary 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T243 + T267 + T290`
- `hypothesis`: 需要一份 governance glossary，冻结所有治理语义
- `asset_requirement`: glossary template、term bindings、governance ledger
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: glossary 漂移；bindings 冲突；ledger 不稳
- `expected_artifact`: universe governance glossary
- `current_gate`: `blocked by release-review / glossary unresolved`

### T292. universe governance matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T196 + T220 + T244 + T268 + T290`
- `hypothesis`: owner/approval/control 需要在 universe 治理层彻底收束
- `asset_requirement`: governance matrix schema、owner/approval matrices、control matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 矩阵不闭环；schema 冲突；治理边界不清
- `expected_artifact`: universe governance matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T293. universe governance doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T249 + T255 + T265 + T289`
- `hypothesis`: no-release / audit / operations doctrine 需要在 universe 层统一
- `asset_requirement`: doctrine templates、maps、charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 冲突；charter 不稳；maps 不闭环
- `expected_artifact`: universe governance doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T294. universe governance audit checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T202 + T250 + T273 + T290-T293`
- `hypothesis`: governance 层需要终版 audit checklist
- `asset_requirement`: checklist template、governance packets、doctrines、matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；governance 包冲突；doctrines 不稳
- `expected_artifact`: universe governance audit checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T295. universe governance route atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T198 + T221 + T246 + T258 + T289`
- `hypothesis`: route/atlas 需要在 universe 治理层统一成终版导航图
- `asset_requirement`: atlas template、route maps、governance map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；route 冲突；governance map 漂移
- `expected_artifact`: universe governance route atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T296. universe governance handoff bundle 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T203 + T234 + T260 + T274 + T295`
- `hypothesis`: 需要一个 governance handoff bundle，保证超长期接手仍无损
- `asset_requirement`: handoff templates、route atlas、doctrine packets、resume bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: bundle 不稳；atlas/doctrine 冲突；resume 不完整
- `expected_artifact`: universe governance handoff bundle
- `current_gate`: `blocked by release-review / bundle unresolved`

### T297. universe governance freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T201 + T261 + T275 + T293 + T296`
- `hypothesis`: 需要一份终版 freeze charter，约束所有治理层的未来修改
- `asset_requirement`: freeze charters、governance doctrine、handoff bundle
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不硬；doctrine 不闭环；bundle 不稳
- `expected_artifact`: universe governance freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T298. universe governance verdict 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T213 + T297`
- `hypothesis`: 需要一份 universe governance verdict，说明为何到此仍 blocked/no-release
- `asset_requirement`: verdict template、freeze charter、no-release proofs
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 含糊；proof 漂移；charter 不稳
- `expected_artifact`: universe governance verdict
- `current_gate`: `blocked by release-review / verdict unresolved`

### T299. universe governance registry 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T214 + T262 + T298`
- `hypothesis`: governance registry 需要终版冻结，便于只读接手
- `asset_requirement`: registry schema、verdict packet、freeze charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 不稳；verdict 冲突；charter 不闭环
- `expected_artifact`: universe governance registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T300. final universe governance canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T289-T299`
- `hypothesis`: 需要一份终版 universe governance canon 作为 Wave 26 的总收口
- `asset_requirement`: canon template、governance packets、registry、charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: canon 不稳；registry/charters 冲突；仍需扩结构
- `expected_artifact`: universe governance canon
- `current_gate`: `blocked by release-review / canon unresolved`

### T301. planning republic draft 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T216 + T240 + T264 + T288 + T300`
- `hypothesis`: 需要一个 republic draft 统一前面所有 frozen canon
- `asset_requirement`: republic template、existing canons、governance canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: draft 不稳；canons 冲突；模板不可执行
- `expected_artifact`: planning republic draft
- `current_gate`: `blocked by release-review / draft unresolved`

### T302. planning republic review 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T301`
- `hypothesis`: republic 需要单独 review，但仍然不能变成 release review
- `asset_requirement`: review template、republic draft、negative boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: review 像 release；draft 不稳；边界不够硬
- `expected_artifact`: planning republic review
- `current_gate`: `blocked by release-review / review unresolved`

### T303. planning republic audit 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T302`
- `hypothesis`: republic 需要 audit，后续才可彻底冻结
- `asset_requirement`: audit template、review packet、governance canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 审计不通过；review 不稳；canon 冲突
- `expected_artifact`: planning republic audit
- `current_gate`: `blocked by release-review / audit unresolved`

### T304. planning republic atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T281 + T295 + T303`
- `hypothesis`: 需要一个 republic atlas 作为最终导航层
- `asset_requirement`: atlas template、existing atlases、audit packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；audit 不稳；旧 atlas 冲突
- `expected_artifact`: planning republic atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T305. planning republic owner-approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T282 + T283 + T303`
- `hypothesis`: republic 层需要最终 owner/approval 责任边界
- `asset_requirement`: matrices、audit packet、owner fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: matrices 冲突；owner 不明；audit 不闭环
- `expected_artifact`: planning republic owner-approval matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T306. planning republic scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T280 + T304`
- `hypothesis`: 需要一组最终 scoreboards 把 republic 层完全导航化
- `asset_requirement`: scoreboards、atlas、metrics schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；atlas 冲突；scoreboards 无效
- `expected_artifact`: planning republic scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T307. planning republic handoff doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T284 + T296 + T304`
- `hypothesis`: 需要终版 handoff doctrine，未来轮次彻底只读
- `asset_requirement`: doctrine template、handoff bundles、republic atlas
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不稳；bundles 冲突；atlas 不闭环
- `expected_artifact`: planning republic handoff doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T308. planning republic freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T297 + T307`
- `hypothesis`: republic 需要 freeze charter 终止后续扩表
- `asset_requirement`: freeze charter、handoff doctrine、governance canon
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不硬；doctrine 不稳；canon 冲突
- `expected_artifact`: planning republic freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T309. planning republic checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T285 + T302 + T303 + T308`
- `hypothesis`: 需要一个终版 checklist，确保 planning republic 可冻结
- `asset_requirement`: checklist template、charter、audit packets、review packet
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；charter/packets 冲突；review 不稳
- `expected_artifact`: planning republic checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T310. planning republic rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T286 + T309`
- `hypothesis`: 需要最终 rehearsal，确认 planning republic 冻结后无需再扩结构
- `asset_requirement`: rehearsal template、checklist、republic atlas、handoff doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；atlas/doctrine 不稳；checklist 不通过
- `expected_artifact`: planning republic rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T311. planning republic verdict 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T287 + T298 + T310`
- `hypothesis`: 需要一份最终 verdict，明确为什么 planning republic 到此冻结
- `asset_requirement`: verdict template、rehearsal packet、no-release proofs、governance verdicts
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 含糊；proof 漂移；rehearsal 失败
- `expected_artifact`: planning republic verdict
- `current_gate`: `blocked by release-review / verdict unresolved`

### T312. final frozen planning republic 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T311`
- `hypothesis`: 需要最终 frozen planning republic，彻底终止后续扩表
- `asset_requirement`: republic template、verdict、handoff doctrine、hard boundaries、resume fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需扩表；republic 不稳；hard boundaries/resume fields 不完整
- `expected_artifact`: final frozen planning republic
- `current_gate`: `blocked by release-review / republic not frozen`

### T313. republic audit map 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T289-T312`
- `hypothesis`: 需要一份 republic audit map，把 republic 层对象完全显式化
- `asset_requirement`: audit map template、republic canons、governance packets
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: map 不稳；canons 冲突；packets 不闭环
- `expected_artifact`: republic audit map
- `current_gate`: `blocked by release-review / audit map unresolved`

### T314. republic audit registry 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T218 + T266 + T313`
- `hypothesis`: republic 层需要独立 registry，后续才可只读接手
- `asset_requirement`: registry schema、audit map、existing registries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: registry 不完整；schema 漂移；audit map 不稳
- `expected_artifact`: republic audit registry
- `current_gate`: `blocked by release-review / registry unresolved`

### T315. republic audit glossary 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T243 + T267 + T314`
- `hypothesis`: 共和国层需要 glossary 终版，防止语义再漂移
- `asset_requirement`: glossary template、registry、term bindings
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: glossary 不稳；bindings 冲突；registry 不闭环
- `expected_artifact`: republic audit glossary
- `current_gate`: `blocked by release-review / glossary unresolved`

### T316. republic governance matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T292 + T305 + T314`
- `hypothesis`: owner/approval/control 需要在 republic 层再收束一次
- `asset_requirement`: governance matrix、owner/approval matrices、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: matrix 冲突；owner 不明；registry 不稳
- `expected_artifact`: republic governance matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T317. republic doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T293 + T307 + T313`
- `hypothesis`: no-release / operations / republic 需要统一成 doctrine
- `asset_requirement`: doctrine templates、audit map、handoff doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 冲突；handoff 不稳；audit map 漂移
- `expected_artifact`: republic doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T318. republic route atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T295 + T304 + T313`
- `hypothesis`: route/atlas 需要在 republic 层冻结成单一导航层
- `asset_requirement`: atlas template、route maps、audit map
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；route 冲突；audit map 不稳
- `expected_artifact`: republic route atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T319. republic scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T306 + T314`
- `hypothesis`: 需要 republic 层的终版 scoreboards
- `asset_requirement`: scoreboard schema、existing scoreboards、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: schema 不稳；scoreboards 冲突；registry 不闭环
- `expected_artifact`: republic scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T320. republic no-release charter 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T272 + T317`
- `hypothesis`: no-release 需要在 republic 层升级成 charter
- `asset_requirement`: charter template、no-release doctrine、republic doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不够硬；doctrines 冲突；仍可误读
- `expected_artifact`: republic no-release charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T321. republic freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T275 + T308 + T320`
- `hypothesis`: 需要一份 republic freeze charter 收束全部 future 变更
- `asset_requirement`: freeze charters、no-release charter、governance matrix
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charters 冲突；matrix 不稳；未来变更条件含糊
- `expected_artifact`: republic freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T322. republic audit checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T273 + T294 + T309 + T320`
- `hypothesis`: 需要 republic 层最终 checklist 作为冻结门槛
- `asset_requirement`: checklist template、charters、audit packets、registry
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；charters 冲突；registry 不稳
- `expected_artifact`: republic audit checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T323. republic freeze rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T310 + T322`
- `hypothesis`: 冻结前需要最后一次 republic rehearsal
- `asset_requirement`: rehearsal template、checklist、route atlas、doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；checklist 不通过；atlas/doctrine 冲突
- `expected_artifact`: republic rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T324. final republic audit canon 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T313-T323`
- `hypothesis`: 需要一份最终 republic audit canon 作为 Wave 28 收口
- `asset_requirement`: canon template、audit packets、rehearsal、charters
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: canon 不稳；packets 冲突；charters 不闭环
- `expected_artifact`: republic audit canon
- `current_gate`: `blocked by release-review / canon unresolved`

### T325. frozen commonwealth draft 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T288 + T312 + T324`
- `hypothesis`: 需要一个 commonwealth draft 统一所有 frozen planning 共和国层成果
- `asset_requirement`: draft template、republic canon、planning constitution
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: draft 不稳；canons 冲突；模板不可执行
- `expected_artifact`: commonwealth draft
- `current_gate`: `blocked by release-review / draft unresolved`

### T326. frozen commonwealth review 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T325`
- `hypothesis`: commonwealth 需要单独 review，但仍不是 release review
- `asset_requirement`: review template、commonwealth draft、negative boundaries
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: review 像 release；draft 不稳；边界不够硬
- `expected_artifact`: commonwealth review
- `current_gate`: `blocked by release-review / review unresolved`

### T327. frozen commonwealth audit 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T326`
- `hypothesis`: commonwealth 需要 audit 后才可冻结
- `asset_requirement`: audit template、review packet、commonwealth draft
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 审计不通过；review 不稳；draft 冲突
- `expected_artifact`: commonwealth audit
- `current_gate`: `blocked by release-review / audit unresolved`

### T328. frozen commonwealth atlas 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T304 + T318 + T327`
- `hypothesis`: 需要一个 final commonwealth atlas 统一所有导航层
- `asset_requirement`: atlas template、existing atlases、commonwealth audit
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: atlas 不可检索；导航冲突；audit 不稳
- `expected_artifact`: commonwealth atlas
- `current_gate`: `blocked by release-review / atlas unresolved`

### T329. frozen commonwealth owner-approval matrix 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T305 + T316 + T327`
- `hypothesis`: commonwealth 需要最终 owner/approval 责任边界
- `asset_requirement`: matrices、audit packet、owner fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: matrices 冲突；owner 不明；audit 不闭环
- `expected_artifact`: commonwealth owner-approval matrix
- `current_gate`: `blocked by release-review / matrix unresolved`

### T330. frozen commonwealth scoreboards 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T319 + T327`
- `hypothesis`: commonwealth 需要终版 scoreboards 作为最终导航面
- `asset_requirement`: scoreboards、audit packet、metrics schema
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: metrics 不稳；scoreboards 冲突；audit 不闭环
- `expected_artifact`: commonwealth scoreboards
- `current_gate`: `blocked by release-review / scoreboards unresolved`

### T331. frozen commonwealth handoff doctrine 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-3`
- `predecessor_pack`: `T307 + T328`
- `hypothesis`: commonwealth 需要终版 handoff doctrine，确保未来只读接手
- `asset_requirement`: doctrine template、handoff packets、commonwealth atlas
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: doctrine 不稳；atlas/packets 冲突；不可执行
- `expected_artifact`: commonwealth handoff doctrine
- `current_gate`: `blocked by release-review / doctrine unresolved`

### T332. frozen commonwealth freeze charter 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T321 + T331`
- `hypothesis`: commonwealth 需要最终 freeze charter，约束所有未来修改
- `asset_requirement`: freeze charter、handoff doctrine、republic canons
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: charter 不硬；doctrine 不稳；canon 冲突
- `expected_artifact`: commonwealth freeze charter
- `current_gate`: `blocked by release-review / charter unresolved`

### T333. frozen commonwealth checklist 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T322 + T332`
- `hypothesis`: 需要一份 commonwealth checklist 确保最终冻结前不漏项
- `asset_requirement`: checklist template、charter、audit packet、matrices
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: checklist 不全；charter 不稳；audit 不闭环
- `expected_artifact`: commonwealth checklist
- `current_gate`: `blocked by release-review / checklist unresolved`

### T334. frozen commonwealth rehearsal 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T323 + T333`
- `hypothesis`: 最终冻结前需要一次 commonwealth rehearsal
- `asset_requirement`: rehearsal template、checklist、atlas、doctrine
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: rehearsal 失败；checklist 不通过；atlas/doctrine 冲突
- `expected_artifact`: commonwealth rehearsal
- `current_gate`: `blocked by release-review / rehearsal unresolved`

### T335. frozen commonwealth verdict 批次

- `status`: `blocked`
- `tier`: `Tier A`
- `release_slot`: `R0`
- `stop_class`: `SC-2`
- `predecessor_pack`: `T334`
- `hypothesis`: 需要一份 verdict 说明为何到此必须停止继续扩结构
- `asset_requirement`: verdict template、rehearsal packet、charters、doctrines
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: verdict 含糊；charters/doctrines 不稳；rehearsal 失败
- `expected_artifact`: commonwealth verdict
- `current_gate`: `blocked by release-review / verdict unresolved`

### T336. final frozen planning commonwealth 批次

- `status`: `blocked`
- `tier`: `Tier D`
- `release_slot`: `R0`
- `stop_class`: `SC-2/SC-3`
- `predecessor_pack`: `T335`
- `hypothesis`: 需要最终 frozen planning commonwealth，彻底终止后续扩表
- `asset_requirement`: commonwealth template、verdict、handoff doctrine、hard boundaries、resume fields
- `compute_budget`: `planning batch only`, future `<= 2 GPUh`
- `stop_conditions`: 仍需扩表；commonwealth 不稳；boundaries/resume 不完整
- `expected_artifact`: final frozen planning commonwealth
- `current_gate`: `blocked by release-review / commonwealth not frozen`

## Non-GPU Prerequisites

以下动作不是 GPU heavy item，但它们比任何 GPU 项更先决：

- 固定 intake 事实
- 锁定 canonical comparator 定义
- 锁定 compute budget 模板
- 锁定 stop conditions 模板
- 锁定 release-review / admission gate

这些前置项没有完成前，不得从本文件中提名任何 GPU heavy item。
