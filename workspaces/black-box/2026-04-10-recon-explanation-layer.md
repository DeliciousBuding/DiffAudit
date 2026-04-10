# 2026-04-10 Recon Explanation Layer

## Status Panel

- `owner`: `research_leader`
- `track`: `black-box`
- `artifact_type`: `decision-grade explanation / boundary artifact`
- `decision_scope`: `writing-only, non-GPU`
- `main_evidence_anchor`: `recon DDIM public-100 step30`
- `best_single_metric_reference`: `recon DDIM public-50 step10`
- `secondary_track`: `variation / Towards = formal local secondary track + blocked real-API assets`
- `CopyMark_classification`: `boundary layer`
- `frequency_paper_classification`: `explanation layer`
- `gpu_release`: `none`
- `admitted_change`: `none`

## 目的

这份文档不改变黑盒 admitted 主结果。

它只负责把两类新增文献吸收进 `recon` 的解释层：

1. `CopyMark / Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models`
2. 频域视角论文 `Unveiling Impact of Frequency Components on Membership Inference Attacks for Diffusion Models`

## 当前固定口径

当前黑盒三层口径继续冻结为：

- `main evidence`
  - `recon DDIM public-100 step30`
- `best single metric reference`
  - `recon DDIM public-50 step10`
- `secondary track`
  - `variation / Towards = formal local secondary track + blocked real-API assets`

这份文档不改上面三层，只解释它们的外推边界和后续分析设计。

## 当前正式裁决

当前正式推荐分支固定为：

- `retain recon main evidence`
- `absorb CopyMark as external-validity boundary`
- `absorb frequency paper as post-hoc explanation layer`

当前固定不做：

- 不新开黑盒 GPU 问题
- 不把新增文献写成 admitted upgrade
- 不把 explanation / boundary 偷换成 execution line

## 一、`CopyMark` 的使用方式

### 当前定位

`CopyMark` 不进入当前黑盒执行主线。

它的作用是：

- 作为现实设定边界说明
- 作为外推风险提醒
- 作为为什么当前 `recon` 结果不能直接讲成“真实预训练模型版权取证已成立”的证据

### 当前应如何写

允许写：

- 当前 `recon` 结果证明：在当前受控协议与资产条件下，成员信号确实可观测
- `CopyMark` 提醒：既有 diffusion MIA 在更真实 benchmark 下可能明显变弱，因此当前结果不能自动外推到预训练大模型现实版权审计
- `CopyMark` 当前作用是收紧 `external-validity boundary`，不是推翻本地主证据

不允许写：

- `CopyMark` 已经推翻当前 `recon` 主证据
- 当前黑盒主线必须立刻切到大模型真实重跑

### 对外叙事边界

以后任何黑盒材料都建议显式带上：

- `threat model`
- `asset semantics`
- `evidence level`
- `external-validity boundary`

其中 `external-validity boundary` 当前至少要写清：

- 当前结果是否建立在 fine-tuned / controlled / public subset 协议上
- 当前结果是否直接适用于真实预训练模型

## 二、频域论文的使用方式

### 当前定位

频域论文不进入当前攻击主线，也不变成新 GPU 题。

它的作用是：

- 进入 `post-hoc analysis`
- 解释为什么某些 member / non-member 样本会出现排序异常或 false negative
- 为未来 attack-defense 解释层提供新的观察维度

### 当前建议分析设计

未来只允许做低成本分析设计，不新增 admitted 主结果：

1. 对 `recon` 结果做频率分组统计
   - 比较高频样本与低频样本的 membership score 分布
2. 把当前 black-box 主证据按频率复杂度做 error slice
   - 观察 false negative 是否集中在高频样本
3. 把频域解释写成“结果解释层”，而不是“攻击能力升级”

### 当前固定写法

允许写：

- 频域论文为当前 `recon` 结果提供 `post-hoc explanation hypothesis`
- 它优先解释：
  - false negative / false positive 的 error slice
  - member / non-member 样本复杂度与 score 分布偏移

不允许写：

- 我们已经得到新的 `frequency-aware black-box mainline`
- 频域论文已经证明当前 admitted 主证据应被重排
- 当前必须为频域分析新开 smoke 或 GPU run

### 当前不做

- 不把频域论文写成新 black-box attack family
- 不因为频域论文新开 GPU 任务
- 不把当前 admitted 结果改写成 frequency-aware mainline

## 三、与现有黑盒矩阵的关系

当前黑盒矩阵仍然保持：

- `recon` 是主证据线
- `variation / Towards` 是 API-only local secondary track
- `TMIA-DM` 不进入黑盒执行面

新增文献的归类应固定为：

- `CopyMark`：边界层
- 频域论文：解释层

## 四、当前结论

当前对 `recon` 最值钱的推进不是新 run，而是：

1. 把 `CopyMark` 吸收到外推边界说明
2. 把频域论文吸收到 post-hoc explanation design
3. 持续保持三层黑盒口径不漂移

## 五、未来触发条件

只有在下面条件出现时，才允许重新讨论新的黑盒执行动作：

1. 当前三层黑盒口径已经在主线文档与状态页完全一致
2. 出现新的、明确的 black-box hypothesis，而不是解释层或边界层补丁
3. 该 hypothesis 写清：
   - threat model
   - asset semantics
   - evidence level
   - external-validity boundary
   - compute budget
   - expected artifact

在这些条件出现前，当前 black-box 默认推进继续固定为：

- `decision-grade explanation / boundary writing only`
