# 2026-04-10 Finding NeMo Activation Export Adapter Review

## 目的

这份文档只记录一个事实：

- `Finding NeMo migrated DDPM observability route` 现在已经在 `Project` 内获得了一个受限的 `activation export adapter`

它不是：

- run 授权
- GPU 申请单
- benchmark 结果
- admitted 升级

## 当前裁决

- `decision_scope`: `Project-only bounded adapter implementation`
- `implementation_status`: `implemented`
- `track`: `white-box`
- `gpu_release`: `none`
- `active_gpu_question`: `none`
- `admitted_change`: `none`
- `paper-faithful NeMo on current admitted white-box assets`: `no-go`

## 本轮实际落地

已落地代码：

- `src/diffaudit/attacks/gsa_observability.py`
- `src/diffaudit/cli.py`
- `tests/test_gsa_observability_adapter.py`

已实现的最小能力：

1. 固定 admitted `checkpoint_root`
2. 固定 canonical / compatibility `sample_id` 绑定
3. 固定单值 `layer_selector`
4. CPU-only 的 sample-pair activation export
5. 输出：
   - `summary.json`
   - `records.jsonl`
   - `tensors/.../*.pt`

## 当前允许主张

当前只允许写成：

- `read-only contract-probe implemented`
- `cpu-only activation-export adapter implemented`
- `code path available`

当前不允许写成：

- `validation-smoke released`
- `mechanism evidence ready`
- `Finding NeMo execution-ready`
- `benchmark-ready`
- `new GPU question released`

## 验证

代码级验证：

- `conda run -n diffaudit-research python -m pytest D:\Code\DiffAudit\Project\tests\test_gsa_observability_adapter.py -q`
  - `5 passed`

真实 admitted 资产验证：

- fixed member sample:
  - `target-member/00-data_batch_1-00965.png`
- fixed control sample:
  - `target-nonmember/00-data_batch_1-00467.png`
- fixed selector:
  - `mid_block.attentions.0.to_v`
- fixed checkpoint:
  - `checkpoint-9600`
- device:
  - `cpu`
- result:
  - `status = ready`
  - `records_written = 2`
  - `tensor_artifacts_written = 2`

## 仍然 blocked 的项

1. 任何 GPU release
2. 任何 `validation-smoke` 放行
3. 任何 benchmark / admitted claim
4. 任何 `cross-attention` 干预
5. 任何 neuron ablation / reactivation
6. 任何 `FB-Mem explains W-1 gap` 叙事

## 下一步

下一步不再是继续补实现，而是做一份 decision-grade review：

- 当前这个 `activation export adapter` 是否已经足够让 `Finding NeMo` 维持在 `zero-GPU adapter-complete hold`
- 还是还需要继续留在 `not-yet`

在这份 review 完成前，固定结论保持：

- `gpu_release = none`
- `active_gpu_question = none`
- `white-box same-protocol bridge = closed-frozen`
