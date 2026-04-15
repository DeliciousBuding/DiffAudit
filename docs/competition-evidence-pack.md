# 比赛主讲模型证据包

更新时间：`2026-04-14 00:33 +08:00`

## 1. 结论先行

当前适合 4C 比赛主讲的模型口径已经固定为：

- 黑盒：`recon`
- 灰盒：`PIA`
- 白盒：`GSA + W-1`

当前最适合主讲的算法闭环仍是：

- `PIA` 作为主讲攻防闭环
- `recon` 作为风险存在性主证据
- `GSA + W-1` 作为深度与上界补充线

当前不应再把 `SMP-LoRA` 写成比赛主讲，也不应再把它写成当前应继续放行的 GPU 主问题。原因不是“这条线没价值”，而是截至 `2026-04-13`，`T06 optimizer/lr frontier` 已经给出新的收口裁决：`closed-mixed-no-go`，下一步应转向 comparator 设计，而不是继续做 optimizer / lr / seed / epoch 类救活式试跑。

## 2. 本包采用的证据优先级

本包按以下顺序取证并裁决：

1. 运行态与当前机器状态
2. 最新实验工件与 `evaluation.json`
3. 当前 admission packet / intake packet
4. 主文档与路线图

因此，对于“下一条 GPU 问题是什么”这个问题，本包不再沿用 `2026-04-10` 文档里“`T06` 仍待放行”的旧状态，而以 `2026-04-13` 的 `T06` packet 和对应输出目录为准。

## 3. 比赛主讲口径

### 3.1 一句话口径

DiffAudit 当前最可辩护的比赛叙事是：我们已经在扩散模型成员推断风险上建立了按权限分层的审计框架，黑盒 `recon` 证明风险真实存在，灰盒 `PIA` 提供当前最成熟、最适合主讲的攻击-防御闭环，白盒 `GSA + W-1` 给出高权限条件下的风险上界与防御比较。

### 3.2 对外讲法

1. 扩散模型存在可观测的成员泄露风险。
2. 我们已经具备黑盒、灰盒、白盒三种权限下的攻击验证能力。
3. 当前最成熟的主讲闭环是灰盒 `PIA`。
4. `PIA` 的攻击信号可解释为 `epsilon-trajectory consistency`，当前防御原型 `stochastic-dropout` 能在同协议下稳定压低攻击指标。
5. 白盒 `GSA + W-1` 已经给出强攻击与 defended comparator，证明系统不仅能看“是否泄露”，还能看“上界有多高、对抗后是否下降”。

### 3.3 主讲边界

- `recon` 是风险证据线，不是比赛中的主讲攻防闭环。
- `PIA` 是比赛中的算法主讲线，但 strongest claim 仍必须写成：
  - `workspace-verified + adaptive-reviewed + paper-alignment blocked by checkpoint/source provenance`
- `GSA + W-1` 是深度补充线，不应被写成“论文级白盒 benchmark 已完成”。

## 4. 当前 admitted 结果

### 4.1 黑盒 admitted 主证据

- `attack`: `recon DDIM public-100 step30`
- `track`: `black-box`
- `AUC`: `0.849`
- `ASR`: `0.51`
- `TPR@1%FPR`: `1.0`
- `定位`: `current black-box main evidence`
- `用途`: 证明最弱权限下已可观测成员风险

### 4.2 灰盒 admitted 主结果

灰盒当前应该成对主讲：

- baseline：
  - `PIA GPU512 baseline`
  - `AUC = 0.841339`
  - `ASR = 0.786133`
- defended：
  - `PIA GPU512 + provisional G-1 stochastic-dropout(all_steps)`
  - `AUC = 0.828075`
  - `ASR = 0.767578`

当前最重要的讲法不是“绝对数值多高”，而是：

- 主结果来自真实资产 mainline，而不是 smoke
- `all_steps` defended line 在 repeated-query adaptive review 后仍保持同向下降
- `GPU128 / GPU256` 的 8GB portability pair 继续支撑“这不是只在单一大卡档位成立的偶然现象”

### 4.3 白盒 admitted 主结果

- attack：
  - `GSA 1k-3shadow`
  - `AUC = 0.998192`
  - `ASR = 0.9895`
- defended：
  - `W-1 strong-v3 full-scale`
  - `AUC = 0.488783`
  - `ASR = 0.4985`

当前这条线可以支持的讲法是：

- 白盒强攻击上界已明确跑出
- defended comparator 已有 full-scale 结果
- same-protocol bridge 只保留为 `closed-frozen` 的诊断治理决策，不进入比赛主结果升级

## 5. 不可主讲内容

以下内容当前不能进入比赛主讲：

- `SMP-LoRA`
  - 只保留为探索线
  - 不写成当前比赛主讲方法
  - 不写成当前最值得继续烧 GPU 的主问题
- `Finding NeMo`
  - 当前仅是 `adapter-complete zero-GPU hold`
  - 不能写成 execution-ready、benchmark-ready 或已获 GPU release
- `SecMI`
  - 当前仍是 `blocked baseline`
  - blocker 是 `real flagfile + matching checkpoint root`
- `TMIA-DM`
  - 当前仍是 `protocol-and-asset decomposition intake only`
- `variation real API`
  - 真实 API 资产仍 blocked
- `white-box same-protocol bridge`
  - 当前只能写成 `closed-frozen`
  - 不能写成 benchmark 已完成

## 6. 给 Developer 的结构化研究资产接口

Developer 当前只应消费 admitted 面，不应消费 candidate / intake 面。

### 6.1 admitted 统一出口

- `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`
  - 当前比赛主结果统一表
  - `Local-API` / `Platform` 应优先从这里取黑盒、灰盒、白盒 headline 指标

### 6.2 应消费的研究语义

- 黑盒：
  - `main evidence = recon DDIM public-100 step30`
  - `best single metric reference = recon DDIM public-50 step10`
  - `secondary track = variation / Towards`
- 灰盒：
  - `PIA baseline + defended(mainline all_steps)`
  - provenance 口径必须随结果一起暴露为边界字段
- 白盒：
  - `GSA admitted attack main result`
  - `W-1 strong-v3 full-scale defended main rung`

### 6.3 不应暴露成系统正式结果的面

- `Research/workspaces/intake/phase-e-candidates.json`
- 各类 `intake` packet
- `Finding NeMo` observability canary
- `batch32 diagnostic comparator`
- 任意 `SMP-LoRA` exploratory outputs

### 6.4 给 Developer 的最小字段建议

系统展示层至少要能稳定读出：

- `track`
- `attack`
- `defense`
- `auc`
- `asr`
- `evidence_level`
- `note`
- `source`
- `boundary`

其中 `boundary` 当前至少要覆盖：

- `recon = fine-tuned / controlled / public-subset / proxy-shadow-member`
- `PIA = workspace-verified + adaptive-reviewed + paper-alignment blocked by checkpoint/source provenance`
- `GSA + W-1 = admitted white-box mainline, but not final paper-level benchmark`

## 7. 给 Leader 的 4C 材料摘要

### 7.1 可以直接进入比赛材料的主摘要

- 我们不是只复现一篇攻击论文，而是建立了按权限层级组织的扩散模型隐私审计框架。
- 黑盒 `recon` 证明最弱权限下风险已可观测。
- 灰盒 `PIA` 是当前最成熟的主讲闭环，具备“攻击信号解释 + 防御原型 + 结构化证据”三件套。
- 白盒 `GSA + W-1` 提供高权限下的风险上界与防御比较，体现研究深度。

### 7.2 Leader 写 4C 材料时应固定的边界句

- 本项目当前最强灰盒主张是 `workspace-verified + adaptive-reviewed`，并同时显式携带 `paper-alignment blocked by checkpoint/source provenance`。
- 黑盒结论成立于受控、微调、公开子集协议，不等于真实预训练模型版权取证已成立。
- 白盒 bridge 当前是治理冻结，不是 benchmark 完成。

### 7.3 PPT / 答辩不应说的话

- 不说 `SMP-LoRA` 是当前主讲线
- 不说白盒 benchmark 已完成
- 不说 `Finding NeMo` 已进入执行主线
- 不说 `PIA` 已 paper-faithful fully reproduced

## 8. 当前 GPU 放行判断

## 8.1 当前判断

当前不应继续放行新的 GPU 问题。

### 8.2 原因

1. `active GPU question` 的运行态检查结果仍是 `none`，且当前 GPU 上没有研究计算型进程，只有桌面/C+G 占用。
2. `T06 optimizer/lr frontier` 已不是“待放行问题”，而是已完成并收口的问题：
   - `AdamW + lower lr` 的 `evaluation.json` 为 `AUC=0.5923295454545454`
   - `SGD(momentum)` 的 `evaluation.json` 为 `AUC=0.42113095238095233`
   - 二者都没有超越历史 `Adam` anchor，无法支持继续把 `SMP-LoRA` 升回 released GPU main question
3. 当前比赛主讲线已经固定为 `recon + PIA + GSA/W-1`，继续为 `SMP-LoRA` 烧 GPU 不会给 4C 主讲材料带来最高信息增量。
4. `Finding NeMo / SecMI / TMIA-DM` 仍分别处于 `zero-GPU hold / blocked baseline / intake only`，现在放 GPU 只会越过既定 gate。
5. 当前更高价值的是把 admitted 结果、边界条件、Developer 接口和 Leader 材料摘要彻底压实，而不是再开一条 exploratory run。

### 8.3 当前唯一合理的后继动作

如果后续一定要继续推进一条高信息增量 run，下一步也不应直接放 GPU，而应先把下面这件事写成单独 packet：

- `baseline vs SMP-LoRA vs W-1 comparator`

它当前只应被视为：

- `next design target`
- `not released`
- `requires separate hypothesis/budget/stop-condition review`

换句话说，当前真实裁决不是“继续放行下一题”，而是：

- `do not release new GPU question now`
- `pivot from optimizer/lr rescue to comparator design`

## 9. 本轮执行证据

### 9.1 指定文档已读取

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\reproduction-status.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\artifacts\unified-attack-defense-table.json`

### 9.2 运行态检查

- `git status`：
  - `D:\Code\DiffAudit` 不是 git worktree，`git status --short` 返回 `fatal: not a git repository`
- `GPU`：
  - `nvidia-smi` 显示 `RTX 4070 Laptop GPU` 当前约 `3186MiB / 8188MiB`
  - 进程列表仅见 `Codex.exe / Claude.exe / Edge / Explorer` 等 `C+G` 进程
  - 没有可见 CUDA 训练/评估进程
- `当前 run`：
  - 进程命令行中未发现指向 `D:\Code\DiffAudit\Research` 的训练脚本
  - 最新与 GPU 决策相关的有效工件来自 `2026-04-13` 的 `SMP-LoRA T06` 输出目录

### 9.3 直接用于裁决的最新工件

- `Research/workspaces/intake/2026-04-13-smp-lora-t06-optimizer-lr-frontier-admission-packet.md`
  - `status = closed-mixed-no-go`
  - `next recommended move = pivot to comparator`
- `Research/outputs/smp-lora-t06-rank1-lambda01-ep10-batch14-throughput-adamw-lowlr-20260413-201437/evaluation.json`
  - `AUC = 0.5923295454545454`
  - `Accuracy = 0.5789473684210527`
- `Research/outputs/smp-lora-t06-rank1-lambda01-ep10-batch14-throughput-sgd-matchedlr-20260413-202327/evaluation.json`
  - `AUC = 0.42113095238095233`
  - `Accuracy = 0.3684210526315789`

## 10. 当前真实判断

截至本轮，比赛主讲模型证据包已经足以固定为：

- 主讲主线：`PIA`
- 风险证据线：`recon`
- 深度补充线：`GSA + W-1`

截至本轮，当前最值得推进的不是新 GPU run，而是：

- 把 admitted 证据、边界口径、系统消费字段和 4C 材料摘要固定成统一口径

截至本轮，当前不应放行新的 GPU 问题；若未来要继续推进 `SMP-LoRA`，应先改写为 comparator packet，而不是继续做 optimizer/lr rescue。
