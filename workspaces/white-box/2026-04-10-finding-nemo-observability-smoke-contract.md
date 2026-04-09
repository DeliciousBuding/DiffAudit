# 2026-04-10 Finding NeMo Portable Observability Smoke Contract

## 目的

这份文档把 `Finding NeMo + local memorization + FB-Mem` 的下一步收口成一份 `portable observability smoke` 入口合同。

它只回答：

- 如果未来要重审一次最小 `validation-smoke`，入口必须长成什么样
- 在当前 admitted 白盒资产上，哪些输入、层级、样本和输出是固定的

它不授权：

- 当前运行
- 长 GPU 任务
- benchmark
- admitted 升级
- bridge 重开

## 固定字段

- `owner`: `research_leader`
- `track`: `white-box`
- `artifact_type`: `portable observability smoke contract`
- `decision_scope`: `Stage 1 planning only`
- `candidate`: `Finding NeMo + local memorization + FB-Mem`
- `gpu_release`: `none`
- `admitted_change`: `none`
- `current_verdict`: `review-ready contract + read-only probe implemented`

## 当前固定边界

- `active_gpu_question = none`
- `white-box same-protocol bridge = closed-frozen`
- `paper-faithful NeMo on current admitted white-box assets = no-go`
- 当前允许的目标只剩：
  - `portable observability exists or not`
- 当前不允许的目标包括：
  - `mechanism evidence`
  - `localized memorization neurons`
  - `explains W-1 defended gap`
  - `default next GPU run`

## Admitted Anchors

### 固定资产根

- `assets_root = workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1`

### 固定 checkpoint root

- `checkpoint_root = workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/checkpoints/target`

### 固定 admitted 攻击锚点

- `GSA epoch300 rerun1`
- `canonical_summary = workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`

### 固定 admitted defended 锚点

- `W-1 strong-v3 full-scale`
- 当前只作为对照锚点，不进入本合同的 hook 入口

## Portable Entry Contract

### 当前状态

- `status = planned-only`
- `implementation = not yet present in repo`

也就是说，下面定义的是未来入口的合同形状，不是假装当前仓库已经有现成命令。

### 解释器合同

基础 `python` 当前不含 `diffusers`，因此入口必须显式绑定研究环境。

允许的 portable 解释器方式只有两类：

1. `DIFFAUDIT_RESEARCH_PYTHON=<absolute-path-to-research-python>`
2. `conda run -n diffaudit-research python`

不允许把本机 scheduler 写成运行前置条件。

### 计划中的命令形状

保留的未来命令 ID：

- `run-gsa-observability-smoke`

计划中的命令形状：

```powershell
conda run -n diffaudit-research python scripts/export_gsa_observability_smoke.py `
  --assets-root workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1 `
  --checkpoint-root workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/checkpoints/target `
  --split target-member `
  --sample-id target-member/<class>/<file> `
  --layer-selector mid_block.attentions.0.to_v `
  --signal-type activations `
  --output-dir workspaces/white-box/runs/finding-nemo-observability-smoke-planned
```

这里的命令当前只作为合同保留，不代表脚本已经实现。

## Layer Naming Map

### 命名来源

在 `diffaudit-research` 环境中，按当前 `GSA DDPM` 的 `UNet2DModel` 结构探针，已确认一组稳定模块前缀存在：

- `down_blocks.4.attentions.*`
- `up_blocks.1.attentions.*`
- `mid_block.attentions.0.*`

并且 `to_q / to_k / to_v / to_out` 子模块都存在。

### 本合同固定的 selector 策略

只允许一层一信号：

- `default_layer_selector = mid_block.attentions.0.to_v`

允许作为 future review 的备选 selector，但不进入当前默认值：

- `down_blocks.4.attentions.0.to_v`
- `up_blocks.1.attentions.0.to_v`

当前不允许：

- 通配多层 sweep
- 同次请求切换多个 selector
- `cross-attention intervention`
- neuron ablation / reactivation

## Sample Binding Rule

### 固定 split

当前合同只允许从 admitted 攻击资产的以下两个 split 中取样：

- `target-member`
- `target-nonmember`

对应路径：

- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-member`
- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-nonmember`

### sample_id 规则

固定规则：

- admitted 资产当前是 flat split layout，不存在强制的 `<class_name>/` 子目录
- canonical `sample_id = <split>/<relative_file_name>`
- compatibility alias 只允许作为读取时的兼容输入：
  - `<split>:<file_stem>`
- 任何 writeback / summary / records 都必须回写 canonical `sample_id`

例子：

- `target-member/00-data_batch_1-00965.png`
- `target-nonmember/00-data_batch_1-00467.png`
- 兼容 alias：
  - `target-member:00-data_batch_1-00965`
  - `target-nonmember:00-data_batch_1-00467`

### 最小 smoke 样本策略

当前合同只允许：

- 每个 split 取按字典序排序后的第一个可用样本
- 最多形成 `1 member + 1 nonmember` 的最小 pair

不允许：

- 随机抽样
- 多批次 sweep
- 扩展到 shadows
- 扩展到 defended 数据面

## Signal Scope

### 当前允许

1. `activations`
2. `grad_norm` 仅可作为可选补充基线

### 当前禁止

1. `cross-attention intervention`
2. neuron-level 干预
3. template / foreground-background 机制证明
4. 任何从 observability 直接跳到 mechanism claim 的解释

## Output Schema

输出必须是机器可审查的小合同，不是研究结论段落。

固定字段：

- `sample_id`
- `split`
- `checkpoint_root`
- `layer_id`
- `layer_selector`
- `signal_type`
- `timestep`
- `tensor_shape`
- `summary_stat`
- `artifact_path`
- `status`
- `notes`

### summary_stat 最小集合

- `mean`
- `std`
- `min`
- `max`

### artifact_path 约束

- 必须是 repo-relative path
- 必须落在本次 smoke workspace 内
- 不允许写机器外绝对路径

## Success Criteria

这份合同未来若被执行，成功只表示：

1. 入口可 portable 地运行
2. 固定 selector 的 sample-level observability 导出存在
3. 输出 schema 完整、可审查

它不表示：

1. `Finding NeMo` 已复现
2. 记忆神经元已定位
3. `W-1` defended gap 已解释
4. GPU 已放行为下一条主线

## Stop Conditions

出现下面任一情况，未来 smoke 即应判为 `not-yet / no-go`：

1. `checkpoint_root` 不能稳定解析到 admitted target checkpoint
2. `layer_selector` 无法绑定到已确认存在的模块路径
3. `sample_id` 规则无法稳定重放
4. 输出 schema 缺字段
5. 入口仍依赖 `LocalOps/paper-resource-scheduler`
6. 需要把 `cross-attention` 干预引入首轮才“看起来有结果”

## Scheduler Section

### 1. Scope And Entry Boundary

`LocalOps/paper-resource-scheduler` is the only GPU scheduling entry for this candidate.

This section governs only:

- `Finding NeMo + local memorization + FB-Mem`
- `Stage 0: protocol reconciliation`
- `Stage 1: portable observability smoke planning`

It does not authorize:

- any run now
- any long GPU job
- any benchmark
- any admitted result upgrade
- any reopening of the closed-frozen white-box bridge

Current fixed state:

- `active_gpu_question = none`
- `white-box same-protocol bridge = closed-frozen`
- `Finding NeMo = zero-GPU-preferred candidate`
- `paper-faithful NeMo on current admitted white-box assets = no-go`

### 2. Zero-GPU Default Path

Default path is `no GPU request`.

Before any future GPU reconsideration, the line must stay in document-only / CPU-safe planning scope:

- fix one admitted `checkpoint_root`
- write one reviewable `layer naming map`
- write one reviewable `sample binding rule`
- write one reviewable `output schema`
- keep the runtime entry portable and reproducible without `LocalOps/paper-resource-scheduler`
- keep target claim limited to `observability exists or not`, not `mechanism proven`

Until all of the above are `review-ready`, scheduler state remains:

- `gpu_release = none`
- `queue_state = not-requestable`

### 3. Exact Condition For A Future One-Shot Validation-Smoke Request

A future GPU request may be reconsidered only if all conditions below are simultaneously true:

1. `Phase D` remains `closed-frozen`, and no other active main GPU question has been released.
2. `GSA epoch300 rerun1` remains the admitted white-box attack main result.
3. `W-1 strong-v3 full-scale` remains the admitted defended main rung.
4. `batch32 diagnostic comparator` remains `runtime-smoke / diagnostic`.
5. This portable smoke-entry contract is `review-ready` and pins all four items:
   - `checkpoint_root`
   - `layer naming map`
   - `sample binding rule`
   - `output schema`
6. The proposed run is explicitly scoped as:
   - `one-shot`
   - `single-card`
   - `short validation-smoke`
   - `activations` first, optional `grad_norm`
   - no `cross-attention` intervention
   - no new training
   - no full-dataset sweep
   - no second GPU round assumed in advance
7. Expected artifact is fixed in advance as one of:
   - `portable observability smoke artifact`
   - `remain intake / not-yet`
   - `no-go`
8. Boss / research review confirms the request is still a `new-question intake validation`, not a disguised bridge continuation or benchmark escalation.

If any one condition above is false, the request is not eligible.

### 4. Required Queue Metadata

Any future eligible request must be recorded only in `LocalOps/paper-resource-scheduler/gpu-resource-requests.md`.

Scheduler-required fields remain mandatory:

- `agent`
- `category=model`
- `state`
- `resource=gpu`
- `requested_at`
- `note`

For this candidate, `note` must additionally encode the following contract metadata in a reviewable way:

- `candidate=Finding NeMo + local memorization + FB-Mem`
- `request_kind=validation-smoke`
- `scope=one-shot single-card short-run`
- `admitted_anchor_attack=GSA epoch300 rerun1`
- `admitted_anchor_defense=W-1 strong-v3 full-scale`
- `checkpoint_root=<single fixed path>`
- `layer_selector=<single fixed selector>`
- `sample_binding_rule=<fixed rule>`
- `signal_type=activations|grad_norm`
- `budget=<single short-run budget>`
- `stop_conditions=<fixed>`
- `expected_artifact=<fixed>`
- `portable_entry=yes`
- `scheduler_not_required_for_repo_repro=yes`

Any request missing any mandatory scheduler field or any contract metadata above must be rejected as `intake incomplete`.

### 5. Scheduling Priority And Release Discipline

Scheduler priority remains:

- `model > markdown`

For this candidate that means:

- if a valid eligible `model` request is ever released in the future, `markdown` GPU work must yield
- this priority does not itself grant Finding NeMo a GPU slot
- no one may bypass the request ledger, skip `requested -> running -> released`, or leave stale `running` state behind

### 6. No-Go Triggers

The request is automatic `no-go` if any of the following is true:

- the proposal is framed as `paper-faithful NeMo` on current admitted white-box assets
- the portable entry still depends on `LocalOps/paper-resource-scheduler` as a repo/runtime prerequisite
- `checkpoint_root`, `layer naming map`, `sample binding rule`, or `output schema` is still missing or unstable
- the first run requires `cross-attention` intervention, neuron ablation/reactivation, extra training, or protocol rewrite
- the plan assumes full-sweep statistics, benchmark comparison, or admitted-boundary claims
- the request would reopen the closed-frozen white-box bridge
- another main GPU question has already been released
- the expected artifact is not limited to smoke-entry evidence and intake decision
- wording drifts into `mechanism evidence` 或 `W-1 defended gap` 解释

## Current Decision

- `gpu_release = none`
- `default_path = zero-GPU`
- `Finding NeMo = eligibility-gated for one minimal validation-smoke only`
- `current action = finish contract, do not run`
