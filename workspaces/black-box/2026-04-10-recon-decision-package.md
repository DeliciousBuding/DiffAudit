# 2026-04-10 Recon Decision Package

## Status Panel

- `owner`: `research_leader`
- `review_gate`: `reviewer_audit_leader + gpu_scheduler_leader`
- `track`: `black-box`
- `artifact_type`: `decision-grade black-box package`
- `decision_scope`: `writing-only, non-GPU + live non-GPU reverify`
- `main_evidence`: `recon DDIM public-100 step30`
- `best_single_metric_reference`: `recon DDIM public-50 step10`
- `secondary_track`: `variation / Towards = formal local secondary track + blocked real-API assets`
- `boundary_layer`: `CopyMark`
- `explanation_layer`: `frequency paper`
- `semantic_gate.current_state`: `proxy-shadow-member`
- `semantic_gate.paper_aligned`: `false`
- `gpu_release`: `none`
- `admitted_change`: `none`
- `phase_e_change`: `none`

## 本轮裁决

本轮把当前黑盒固定包正式压成一份可直接复述、可审查、可冻结的 decision-grade artifact。

当前唯一允许对外复述的黑盒五件套固定为：

1. `main evidence = recon DDIM public-100 step30`
2. `best single metric reference = recon DDIM public-50 step10`
3. `secondary track = variation / Towards = formal local secondary track + blocked real-API assets`
4. `CopyMark = boundary layer`
5. 频域论文 = `explanation layer`

这份 package 不改变 admitted hierarchy，不释放 GPU，也不触发新的 black-box execution line。

## 本轮真实执行

### 1. 公共包语义审计

本轮重新执行：

```powershell
conda run -n diffaudit-research python -m diffaudit audit-recon-public-bundle `
  --bundle-root D:\\Code\\DiffAudit\\Download\\black-box\\supplementary\\recon-assets\\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models
```

返回：

- `status = ready`
- `semantic_gate.current_state = proxy-shadow-member`
- `semantic_gate.paper_aligned = false`
- `allowed_claim = local-semantic-chain-ready`

解释：

- 当前公开包已经是 `machine-audited + locally self-consistent`
- 但它仍受 `proxy-shadow-member` 语义约束
- 因此 strongest claim 不能升级成 `paper-aligned`

### 2. 非 GPU artifact-mainline 复算

本轮重新执行：

```powershell
conda run -n diffaudit-research python -m diffaudit run-recon-artifact-mainline `
  --artifact-dir D:\Code\DiffAudit\Research\experiments\recon-runtime-mainline-ddim-public-100-step30\score-artifacts `
  --workspace D:\Code\DiffAudit\Research\experiments\recon-artifact-mainline-public-100-step30-reverify-20260410-round28 `
  --repo-root D:\Code\DiffAudit\Research\external\Reconstruction-based-Attack
```

新产物：

- `experiments/recon-artifact-mainline-public-100-step30-reverify-20260410-round28/summary.json`

复算结果：

- `auc = 0.849`
- `asr = 0.51`
- `tpr@1%fpr = 1.0`

解释：

- 这是 `non-GPU artifact-mainline reverify`
- 它只证明现有 score artifacts 可以再次稳定桥接到同一主结论
- 它不是新的 admitted upgrade，也不是新的黑盒主结果

## 为什么 `public-100 step30` 仍是主证据

必须保留的 strongest wording 是：

- `public-100 step30 is the main evidence because its artifact chain is the current most complete and most defensible.`
- `public-50 step10 is not the main evidence; it is only the current best single AUC reference.`

当前保留 `public-100 step30` 的理由是：

1. 它已经同时具备 `runtime-mainline + artifact-mainline + bundle audit` 三段证据链
2. 本轮的 non-GPU reverify 又证明现有 score artifacts 可以稳定复算到相同 headline metrics
3. 当前裁决依据是“证据链完整、语义门清晰、可复核”，不是“所有单指标都最高”

当前明确不允许写成：

- `step30` 证明采样步数越多越强
- `step30` 全面强于 `step10`

当前只允许写成：

- `step30` 是当前更完整、更可辩护的主证据
- 它不构成“采样步数单调提升”的规律宣告

## 最佳单指标参考与次轨边界

### `best single metric reference`

`recon DDIM public-50 step10` 当前仍保留 `AUC = 0.866` 的最佳单指标参考地位。

但它不能替代主证据。

因此任何正式材料都必须把下面两句分开写：

- `main evidence = recon DDIM public-100 step30`
- `best single metric reference = recon DDIM public-50 step10`

### `secondary track`

`variation / Towards` 当前只能写成：

- `formal local secondary track + blocked real-API assets`

当前 blocker 仍是：

- `query_image_root / query images` 缺失

因此它不能被写成：

- 真实可运行的 real-API black-box path
- 当前黑盒并行执行主线

## 外推边界与解释层

### `CopyMark = boundary layer`

`CopyMark` 当前只承担一个职责：

- 收紧 `external-validity boundary`

允许写：

- 当前 `recon` 结果证明成员信号在当前受控协议下可观测
- `CopyMark` 提醒：真实 benchmark 下既有 diffusion MIA 可能明显变弱，因此当前结果不能自动外推到真实预训练模型版权审计

不允许写：

- `CopyMark` 已推翻当前 `recon` 主证据
- 黑盒主线必须立刻切到真实大模型重跑

### 频域论文 = `explanation layer`

频域论文当前只承担：

- `post-hoc explanation hypothesis`
- error-slice / false-negative / false-positive 的解释视角

允许写：

- 频域论文为当前 `recon` 结果提供解释层
- 它优先用于频率复杂度与 score 分布偏移的分析设计

不允许写：

- 我们已经得到新的 `frequency-aware black-box mainline`
- 频域论文已经把 admitted hierarchy 改写
- 因为频域论文必须新开 smoke 或 GPU run

## 当前固定边界

当前黑盒 package 必须显式带出四项边界：

### `threat model`

- fine-tuned diffusion model 上的 black-box 成员推断

### `asset semantics`

- `fine-tuned / controlled / public-subset / proxy-shadow-member`

### `evidence level`

- `black-box main evidence`

### `external-validity boundary`

- 当前结果证明成员信号在受控协议下可观测
- 不等于真实预训练模型版权取证已成立

## 当前不升级什么

这份 package 明确不升级下面任何一项：

1. 不把 `machine-audited local semantic chain` 升级成 `paper-aligned`
2. 不把 `non-GPU artifact-mainline reverify` 升级成新的 admitted 结果
3. 不把 `CopyMark` 升级成反证或执行切换信号
4. 不把频域论文升级成新攻击主线、新 benchmark 或新 GPU 题
5. 不把 `variation / Towards` 升级成 real-API runnable line
6. 不把 `TMIA-DM` 塞回黑盒 hierarchy
7. 不触发 `Phase E` 变化

## 下一步

当前黑盒线的正确动作已经从“继续找 run”切到“保持 fixed package 冻结并同步外围文档”。

本轮完成后，下一条最值得推进的唯一目标切回：

- `PIA provenance` 的 `release/source identity + split provenance + paper protocol mapping` CPU 补件

黑盒线在出现新的单问题 hypothesis 之前，继续保持：

- `writing-only / non-GPU`
- `admitted_change = none`
- `gpu_release = none`

