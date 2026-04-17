# Leader Research Delivery Snapshot
更新日期：`2026-04-18`

## 1. Leader 一页主讲表

| 线索 | 模型 | 攻击 | 防御 | AUC | ASR | TPR@1%FPR | TPR@0.1%FPR | 能说的话 | 不能说的话 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 黑盒 | `Stable Diffusion v1.5 + DDIM` | `recon DDIM public-100 step30` | `none` | `0.849` | `0.51` | `1.0` | `n/a` | 最弱权限下风险已可观测，当前黑盒主证据成立 | 不说成最终 exploit，不外推到完整真实世界 benchmark |
| 灰盒 | `CIFAR-10 DDPM` | `PIA GPU512 baseline` | `stochastic-dropout(all_steps)` | `0.841339 -> 0.828075` | `0.786133 -> 0.767578` | `0.058594 -> 0.052734` | `0.011719 -> 0.009766` | 当前最成熟的攻击-防御闭环，攻击信号与 defended pairing 都已 admitted，且必须与 bounded repeated-query adaptive 边界一起按四指标阅读 | 不说成已 paper-faithful fully reproduced，不说 provenance blocker 已解除，不说 validated privacy protection |
| 白盒 | `CIFAR-10 DDPM / DPDM` | `GSA 1k-3shadow` | `W-1 strong-v3 full-scale` | `0.998192 / 0.488783` | `0.9895 / 0.4985` | `0.987 / 0.009` | `0.432 / 0.0` | 展示高权限风险上界与 defended contrast | 不说成 final paper benchmark，不把 closed-frozen bridge 写成新执行线 |

这张一页表本身也必须遵守灰盒四指标合同，不能让读者先看到 `AUC / ASR`、再在正文里补 low-FPR 与 adaptive 边界。

## 2. 已 admit 的主线成果（可直接拿到 4C 材料）

- **黑盒 risk evidence（`recon DDIM public-100 step30`）**：AUC 0.849、ASR 0.51、TPR@1%FPR 1.0，在 controlled/public-subset/proxy-shadow-member 数据上运行，是当前最弱权限下观测成员泄露的 admitted 证据。对外侧重“risk exists”而非“终极 benchmark”。  
- **灰盒 PIA 基线与 defended（GPU512 baseline，G-1 stochastic-dropout all_steps defended）**：baseline `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR = 0.841339 / 0.786133 / 0.058594 / 0.011719`，defended `= 0.828075 / 0.767578 / 0.052734 / 0.009766`。这组结果必须按四指标一起读，并同时读作：`workspace-verified + adaptive-reviewed + bounded repeated-query adaptive boundary + paper-alignment blocked by checkpoint/source provenance`，不能被误读成“已复现论文版本”或“已验证隐私保护”。  
- **白盒 GSA + W-1 bridge**：GSA 1k-3shadow 攻击跑出 AUC 0.998192 / ASR 0.9895，W-1 strong-v3 full-scale defended run AUC 0.488783 / ASR 0.4985，放在 same-protocol “closed-frozen” comparator 上，用于展示高权限下的风险上界与 defended contrast，而不是声称已完成 benchmark。

每条主线都应配上 `track/attack/defense/auc/asr/evidence_level` 字段以及 `boundary`（如 recon = fine-tuned+controlled, PIA = workspace-verified + blocked provenance, GSA/W-1 = admitted bridge but not final paper benchmark），此结构已写在 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`，Platform/Local-API 可以直接取用。

## 3. 今天必须同步的非 admitted 变化

- admitted 三线主讲表不变，不替换指标，不替换 headline。
- `I-C` 边界层必须更新：
  - 当前最强诚实口径只能是 `translated-contract-only + negative falsifier`
  - 可以说 translated gray-box alias execution 已存在
  - 不可以说 cross-permission unified support 已成立
  - 不可以把这条线写成新的 GPU release 或新主讲创新
- gray-box challenger 层必须更新：
  - 攻击 challenger 仍以 `TMIA-DM late-window` 为最强参考
  - defended challenger 现在应优先写成 `TMIA + temporal-striding(stride=2)`，不再优先引用 `TMIA + dropout`
- black-box boundary 层必须更新：
  - `CLiD` 只能写成 `evaluator-near local clip-only corroboration`
  - `variation` 只能写成 `contract-ready blocked`
  - `served-image-sanitization` 已经是第一条明确试过并 rejected 的 serving-side mitigation no-go
- white-box breadth 层必须更新：
  - 当前 repo 仍只有 `W-1 = DPDM` 这一条可执行 defended family
  - `Finding NeMo` 与 `Local Mirror` 目前都不能写成第二 defended family
  - `Finding NeMo` 当前最强诚实口径也已不再只是 `hold`：
    - one actual bounded admitted fixed-mask packet now exists
    - current packet verdict is `negative but useful`
    - do not promote this into defense-positive or benchmark-ready wording
- `I-D` conditional future-surface 层也必须更新：
  - 当前最强诚实口径是：
    - `SD1.5 + celeba_partial_target/checkpoint-25000` local canary contract
    - first bounded `CFG` packet = `positive but bounded`
    - first actual runner-level hidden-guidance-jitter rerun = `negative but useful`
    - current honest bounded successor lane = `none`
  - 可以说 conditional branch now has:
    - one frozen contract
    - one bounded attack packet
    - one negative actual defense rerun
  - 不可以说：
    - conditional diffusion 已被全面覆盖
    - `g=7.5` 已经成为普适最优规则
    - hidden-guidance jitter 已经是 release-grade defense
    - 当前还有 ready successor lane
    - low-FPR 或 adaptive robustness 已成立

这些变化会影响 Leader 的讲法、答辩补充口径和 challenger queue 说明，但**不**改变 admitted 主表。

## 4. 创新/差异化亮点（直接可在 4C 材料中提及）

1. Recon 证据线不是孤立 payload，而是可反复 eyeball 的 evidence chain：主证据 + best single metric reference + metadata 全部写进 unified table，能说明“我们用一条可复现流程在证明风险存在”。  
2. PIA 用 adaptive reviewer + stochastic-dropout 交付攻防三件套（attack signal 解释 + defended prototype + provenance boundary），并且当前高层口径必须显式带四指标与 bounded repeated-query adaptive 边界，比单纯复现论文更有 narrative depth。  
3. GSA/W-1 white-box bridge 提供 “admitted comparator + defended contrast”，展示研究深度；同一份 evidence table 里也在记录 expected comparator gating。  
4. 除 admitted 主讲线外，我们已经把 challenger/boundary 层做成可管理的队列，而不是把所有非 admitted 结果都混成“待定”：`TMIA temporal-striding`、`variation contract-ready blocked`、`CLiD evaluator-near`、`black-box mitigation no-go` 都已经有明确位置和说法。
5. `I-D` 现在也不再只是口号：它已经有真实 conditional packet truth，但仍然只能作为 future-surface supporting evidence，而不是 admitted 第四主线。

## 5. Replay / comparator 资产绝不可冒充 admitted

- 任何 replay/artifact（如 recon replay/demo traces、`Research/experiments/*` 中的复查脚本）只能在 Local-API 里标记为 “debug/replay/comparator trace”，不得用来支撑 admitted claim。  
- 与 SMP-LoRA 相关的 optimizer/lr frontier（`2026-04-13-smp-lora-t06-optimizer-lr-frontier-admission-packet`）已经被定为 `closed-mixed-no-go`，对应的 AdamW/SGD evaluation 结果都低于 baseline anchor，属于 comparator-only hypothesis。  
- `2026-04-14-baseline-smp-lora-w1-comparator-admission-packet` 明确要求 “no new GPU question” 直到 comparator verdict；SMP-LoRA 只能说是 comparator candidate，不能写成 admitted run、放 GPU 释放，或者主讲线。  
- `SecMI / 外来 zip 或 checkpoint` 当前都只能停在 `blocked baseline / hold / intake only / reference-only` 语义，不能与 admitted 三线并列。
- `Finding NeMo` 现在应单独写成：
  - `non-admitted actual bounded falsifier`
  - not `zero-GPU hold`
  - not `defense-positive`
- `TMIA-DM` 现在必须拆开写：
  - 可以写成 challenger queue 的有效成员
  - 不能写成 admitted 第四主线
  - defended 侧优先参考 `temporal-striding`，不是旧的 `dropout`

## 6. 如果评委追问 SMP-LoRA

1. 明确说：当前 GPU 状态是 “none”，T06 optimizer/lr frontier 已判 closed-mixed-no-go，后续行动已经 pivot 到 comparator 设计。  
2. 讲述我们在做的只是 “baseline vs SMP-LoRA vs W-1 comparator”，目的是要么打穿 W-1 防线，要么把 SMP-LoRA 归入 no-go narrative；任何 comparator output 都要附加 `boundary` 字段说明 “comparison-only / requires separate hypothesis review”。  
3. 解释为什么 Leader 现阶段只讲 recon/PIA/GSA-W1：它们有 admitted 指标、统一的 evidence table 和 definitive boundary，在 4C 材料里能形成完整 narrative；SMP-LoRA 目前还只是在 comparator gate 里打磨，等 verdict 出来之后再决定是否可以转为 admitted。

## 7. 交付给 Leader 的下一步

- 已经可以把上面三条主线的关键指标、boundary 语句、innovation bullet 直接迁入 4C 材料（例如 PPT/答辩提纲、材料页、口播词）。  
- 任何提到 SMP-LoRA 时都附上 “comparator gate / planning mode / no new GPU question” 的限定语。  
- 立即补一轮 wording-only sync：
  - `TMIA defended challenger = temporal-striding`
  - `CLiD = evaluator-near local clip-only corroboration`
  - `variation = contract-ready blocked`
  - `served-image-sanitization = rejected no-go`
  - `white-box defense breadth = only W-1 executable`
- 需要 Developer/Leader 复查 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`、`Research/docs/competition-evidence-pack.md`、`Research/docs/research-boundary-card.md`，确认 metrics/boundary 字段能无缝对接平台展示与答辩口径。
