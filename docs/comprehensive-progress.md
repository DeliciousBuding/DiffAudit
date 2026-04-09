# Comprehensive Progress

这份文档是 `Project` 研究仓库的综合进度入口。

它不替代 [reproduction-status.md](reproduction-status.md) 的逐线细节，也不替代 [mia-defense-research-index.md](mia-defense-research-index.md) 的文献整理；它的职责是把“当前最能讲的攻击线、最缺的防御线、最短执行路径”放到一页里。

## 当前一句话

当前仓库已经具备三条攻击线的基本骨架，但真正最成熟、最适合打成“攻击 + 防御”主讲闭环的，是灰盒 `PIA`。黑盒 `recon` 负责提供最强风险证据，白盒 `GSA` 负责提供深度与上界，防御线整体仍明显落后于攻击线。

## 进度总览

| 维度 | 当前判断 | 备注 |
| --- | --- | --- |
| 黑盒攻击 | `较成熟` | `recon` 是当前最强证据线 |
| 灰盒攻击 | `最成熟` | `PIA` 已进入 real-asset runtime mainline |
| 白盒攻击 | `已拿到强结果 + 有增强 rerun 在跑` | `GSA` 已完成 `1k-3shadow` paper-aligned runtime，AUC 达到 `0.97514`；独立资产根上的 `epoch300 rerun1` 正在后台训练 |
| 黑盒防御 | `基本未落地` | `B-1 / B-2` 仍在设计层 |
| 灰盒防御 | `已进入 provisional G-1 + adaptive gate completed` | `PIA GPU128/GPU256/GPU512` 三档与一次 `GPU512` 同档 repeat 都显示 `stochastic-dropout` 压低指标；新的 `GPU512` adaptive-reviewed baseline + `all_steps / late_steps_only` 已落地 |
| 白盒防御 | `已有 strong-v2 主结果，strong-v3 已推进到 full-scale` | `DPDM` 已完成 stronger defended comparator，当前已拿到 stronger rung 的 full-scale GPU 结果 |
| 统一评估表 | `已有第一版` | 已新增 admitted main results 的跨盒总表 |

## 攻击主线

### 黑盒

- 主线：`recon`
- 次主线候选：`variation`（对应 `Towards Black-Box`）
- 当前能说的话：
  - 公开资产上的 black-box 风险已经有可引用主证据
  - `variation` 已能在本地 CPU 上重复跑 synthetic smoke
  - `variation` 的真实 API 资产 probe 已确认 blocked，当前缺 query image root
  - 新归档 `TMIA-DM` 已证明时间相关噪声 / 梯度信号也是正式文献方向，但它当前不属于严格黑盒执行面
- 当前不能说的话：
  - 还不能把 black-box 防御讲成已有结果
  - 还不能把 `variation` 写成真实 API 闭环
  - 还不能把 `TMIA-DM` 写成黑盒新主线
- 当前用途：
  - 作为申报和答辩里的“风险存在”主证据
  - `variation` 适合作为第二黑盒候选线补充进申报叙事
  - 黑盒最终口径现在应区分 `main evidence`、`best single metric reference` 和 `secondary track`

### 灰盒

- 主线：`PIA`
- baseline：`SecMI`（当前已判定为 `blocked baseline`）
- 当前能说的话：
  - `PIA` 已经不是 smoke，而是真实资产 mainline
  - `PIA GPU128 / GPU256 / GPU512` 已拿到同口径 baseline + defense 对照，且 defense 指标连续三档都低于 baseline
  - `PIA GPU512` 同档 repeat 也继续维持 defense 优于 baseline
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

### 白盒

- 主线：`GSA`
- 扩展：`Finding NeMo`
- 当前能说的话：
  - 白盒闭环已经打通
  - 资产根、checkpoint-*、bucket 已进入规范结构
  - `DPDM` 已从环境阻塞推进到真实 CUDA checkpoint
  - 当前白盒防御的主要技术问题是评估桥接，不是训练缺失
  - `GSA` 已跑出第一版强白盒结果
  - 一条更强配置的 `GSA epoch300 rerun1` 已在独立资产根上启动，当前仍在 `target` 训练阶段
  - `DPDM` target-only comparator 当前接近随机，方向上支持防御有效
  - `DPDM` multi-shadow comparator 当前也接近随机，方向上继续支持防御有效
  - `DPDM` 在 defended target-member checkpoint 上仍接近随机，白盒防御信号更明确
  - `DPDM` 的 defended-target + defended-shadows `strong-v2` comparator 为 `AUC = 0.541199`，仍显著弱于 `GSA = 0.97514`
  - `DPDM` 的 `strong-v2 max512` comparator 为 `AUC = 0.537201`，说明更大评估规模下趋势仍未反转
  - `DPDM` 的 `strong-v2 3-shadow max512` comparator 为 `AUC = 0.462799`，这是当前最接近 defended `1k-3shadow` 结构的本地结果
  - `DPDM` 的 `strong-v2 3-shadow full-scale` comparator 为 `AUC = 0.490813`，仍明显弱于 `GSA` 主线
  - `DPDM` 的 `strong-v3 3-shadow max128` comparator 为 `AUC = 0.537048`，说明 stronger training rung 已经能在 GPU 上稳定出第一条 defended 结果
  - `DPDM` 的 `strong-v3 3-shadow max256` comparator 为 `AUC = 0.522339`，说明这条更强训练 rung 已经推进到中规模 GPU defended 结果
  - `DPDM` 的 `strong-v3 3-shadow max512` comparator 为 `AUC = 0.5`，说明 stronger training rung 已推进到更大规模 GPU defended 结果
  - `DPDM` 的 `strong-v3 3-shadow full-scale` comparator 为 `AUC = 0.488783`，说明 stronger training rung 已完成 full-scale defended 结果
- 当前不能说的话：
  - 还不能说白盒论文级复现成功
  - 还不能说白盒 defense 比较已经完成
  - 还不能把 `epoch300 rerun1` 写成新的攻击结果，因为它还没有 `summary.json`
  - 还不能把 `DPDM` target-only comparator写成同口径白盒攻击结果
  - 还不能把当前 `DPDM strong-v2 defended-target multi-shadow comparator` 写成最终白盒 defense benchmark
- 当前用途：
  - 作为技术深度补充线

## 防御主线

### 当前建议

| 轨道 | 当前最合理防御路线 | 当前判断 |
| --- | --- | --- |
| 黑盒 | `B-1 / B-2` | 设计方向成立，但还没有正式实现 |
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

1. 固定 `PIA provisional G-1`
2. `recon` 主证据口径收口
3. `variation / Towards` 保持 formal local secondary track，并明确 real-API assets blocked
4. 低频监控白盒 `epoch300 rerun1`
5. 固定白盒 defended 主结果口径
6. 基于第一版统一总表继续补质量 / 成本列，并保持灰盒机理说明与 adaptive gate 一致

## 申报 / PPT 应该怎么讲

当前最合理的讲法是：

1. 扩散模型存在成员泄露风险
2. 我们已经在黑盒、灰盒、白盒三种权限下建立了攻击验证能力
3. 当前最成熟的是灰盒 `PIA`
4. 我们已经拿到一个 `provisional G-1` 灰盒防御闭环
5. 白盒 `GSA + W-1` 已经进入“强攻击结果已出，strong-v2 / strong-v3 defended full-scale comparator 都已完成”的阶段

## 关联文档

- 逐线状态：[reproduction-status.md](reproduction-status.md)
- 防御文档索引：[mia-defense-research-index.md](mia-defense-research-index.md)
- 防御执行清单：[mia-defense-execution-checklist.md](mia-defense-execution-checklist.md)
- 研究仓路线图：[../ROADMAP.md](../ROADMAP.md)
