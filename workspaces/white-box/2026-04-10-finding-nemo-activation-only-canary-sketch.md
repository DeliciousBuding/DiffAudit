# 2026-04-10 Finding NeMo Activation-Only Canary Sketch

## 目的

这份文档只回答一个问题：

如果未来要在不释放 GPU 长跑、不越过现有合同边界的前提下，为 `Finding NeMo migrated DDPM observability route` 补一个最小实现草案，它应该长成什么样。

它不是：

- 运行计划
- GPU 申请单
- benchmark 提案
- admitted 升级文档

## 固定字段

- `owner`: `research_leader`
- `track`: `white-box`
- `artifact_type`: `activation-only canary implementation sketch`
- `decision_scope`: `implementation sketch only`
- `candidate`: `Finding NeMo + local memorization + FB-Mem`
- `gpu_release`: `none`
- `admitted_change`: `none`

## 当前前提

- `paper-faithful NeMo on current admitted white-box assets = no-go`
- `portable observability smoke contract = review-ready`
- 当前唯一允许验证的目标：
  - sample-level `activations` 导出链是否可被 portable 地实现
- 当前不允许的目标：
  - `mechanism evidence`
  - neuron localization
  - `FB-Mem` 解释
  - `cross-attention intervention`
  - GPU run release

## 草案范围

本草案只覆盖：

1. 未来拟新增的脚本 / 模块名
2. 最小输入参数
3. 样本绑定逻辑
4. 层选择器解析逻辑
5. 输出 `summary.json + records.jsonl + tensor artifact` 的生成逻辑

本草案明确不覆盖：

1. 真正的实现提交
2. 真正的 smoke 执行
3. 任何 GPU 预算消耗
4. 任何 `cross-attention` 或 neuron-level 干预

## 现有可复用入口

### 已存在

1. `src/diffaudit/attacks/gsa.py`
   - 已有 `probe_gsa_assets`
   - 已有 `run_gsa_runtime_mainline`
   - 已有 `checkpoint-*` 自动解析
2. `src/diffaudit/cli.py`
   - 已有 `probe-gsa-assets`
   - 已有 `run-gsa-runtime-mainline`
3. `workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py`
   - 已证明 `DDPM UNet2DModel` 可被加载并处理 admitted 资产

### 当前不足

1. 没有 sample-level file metadata 回传
2. 没有 `activation hook` 导出路径
3. 没有 `records.jsonl` 级机器合同
4. 没有 `layer_selector` 解析逻辑

## 拟新增文件形状

### 1. Python 模块

建议新增：

- `src/diffaudit/attacks/gsa_observability.py`

职责只限于：

1. 解析合同输入
2. 解析 `sample_id`
3. 解析 `layer_selector`
4. 做单样本 activation 导出
5. 写出 `summary.json`
6. 写出 `records.jsonl`

### 2. CLI 命令

建议新增：

- `run-gsa-observability-smoke`

当前只是命令占位设计，不表示应立即实现。

### 3. 脚本入口

如果不先改 CLI，也可先保留脚本草案名：

- `scripts/export_gsa_observability_smoke.py`

这应被视为实现草案中的过渡入口，不应被写成长期主入口。

## 最小参数合同

### 必填参数

- `--workspace`
- `--repo-root`
- `--assets-root`
- `--checkpoint-root`
- `--checkpoint-dir`
- `--split`
- `--sample-id`
- `--control-sample-id`
- `--layer-selector`
- `--signal-types`
- `--timestep`
- `--noise-seed`
- `--prediction-type`
- `--device`
- `--provenance-status`

### 当前固定默认值

- `assets_root = workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1`
- `checkpoint_root = workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/checkpoints/target`
- `checkpoint_dir = .../checkpoint-9600`
- `layer_selector = mid_block.attentions.0.to_v`
- `signal_types = activations`
- `device = cpu`

## 样本绑定逻辑

### 目标

把当前 GSA 的 dataloader/batch 驱动路径，收紧成可复审的单样本绑定路径。

### 规则

1. 不允许依赖 dataloader `shuffle=True` 的输出顺序
2. 不允许依赖 batch index
3. 只允许按合同里的 `sample_id` 反解到真实文件

### 建议实现步骤

1. 从 `assets_root/datasets/<split>` 扫描真实文件
2. 构建：
   - `sample_id -> dataset_relpath`
3. 对 `sample_id` 做唯一解析
4. 失败即停止，不进入模型前向

### 当前固定 pair

- `target-member` 第一条按字典序的 canary
- `target-nonmember` 第一条按字典序的 control canary

当前不允许：

- 批量 pair
- 随机 pair
- shadow split

## Layer Selector 解析逻辑

### 命名源

只允许使用模型真实 `named_modules()` 或 `state_dict` 前缀。

### 解析步骤

1. 加载与 GSA 一致的 `UNet2DModel`
2. 遍历 `named_modules()`
3. 对 `layer_selector` 做唯一匹配
4. 若命中 0 个或多个模块，直接 `blocked`

### 默认策略

- `mid_block.attentions.0.to_v`

备选只作为常量保留，不自动扩展：

- `down_blocks.4.attentions.0.to_v`
- `up_blocks.1.attentions.0.to_v`

## Activation Export 草案

### 目标

只导出一次 sample-level activation tensor，不做机制解释。

### 建议逻辑

1. 构造单样本 dataset
2. 绑定单个 hook 到选定模块
3. 做一次前向
4. 捕获 activation tensor
5. 写出：
   - tensor artifact
   - `records.jsonl`
   - `summary.json`

### 当前明确不做

1. 不做 aggregation over dataset
2. 不做 neuron ranking
3. 不做 ablation
4. 不做 reactivation
5. 不做解释文本生成

## 输出文件草案

### 1. `summary.json`

职责：

- 只记录 contract 是否被满足

必须包含：

- `schema`
- `status`
- `asset_root`
- `checkpoint_root`
- `resolved_checkpoint_dir`
- `sample_binding_rule`
- `layer_map_strategy`
- `requested`
- `checks`
- `artifacts`

### 2. `records.jsonl`

职责：

- 每行 1 条样本级导出记录

必须包含：

- `sample_id`
- `split`
- `dataset_relpath`
- `checkpoint_root`
- `resolved_checkpoint_dir`
- `signal_type`
- `layer_id`
- `timestep`
- `tensor_shape`
- `summary_stat`
- `artifact_path`

### 3. tensor artifact

建议目录：

- `tensors/<sample_id_sanitized>/<layer_selector>_t<timestep>.pt`

## 成功 / 失败判据

### 成功

只表示：

1. `sample_id` 可以唯一解析
2. `layer_selector` 可以唯一解析
3. 至少写出 1 条 sample-level activation record
4. 合同里的 `summary.json + records.jsonl + tensor artifact` 都存在

### 失败

任一情况都应判 `blocked / no-go`：

1. `sample_id` 无法唯一解析
2. `layer_selector` 命中 0 或多个模块
3. 仍依赖 batch index 才能解释输出
4. 输出缺机器合同字段
5. 需要 GPU 才能验证入口本身

## 明确未实现项

为了避免误读，这里必须显式保留：

1. 当前没有真实实现
2. 当前没有新 CLI 子命令
3. 当前没有真正的 activation hook adapter
4. 当前没有 run
5. 当前没有新的 summary artifact

## Scheduler Note

当前这份 activation-only canary sketch 仍然是 `zero-GPU`。

原因不是“还没排到卡”，而是它当前仍属于 `portable observability smoke` 的实现草案层：目标只允许收敛到“固定 admitted 资产上是否存在可重放的 sample-level activation 导出入口”，不允许越级成运行申请、benchmark、机制证明，或对白盒 bridge 的变相续跑。

按现有边界，`Finding NeMo` 仍只是 `eligibility-gated for one minimal validation-smoke only`，且 `portable observability smoke` 仍停留在合同/实现草案层，未获 run release。

在未来连一次 one-shot smoke 都重新评审之前，必须先审实现而不是审结果。最低 review points 仍包括：

- `checkpoint_root` 是否固定到单一 admitted target 路径
- `layer_selector` 是否固定为单层单信号且能稳定绑定
- `sample_binding_rule` 是否能把 `sample_id` 稳定重放到 admitted split
- 导出 schema 是否完整并保持 repo-relative artifact
- 入口是否显式绑定研究解释器而不是依赖 `LocalOps/paper-resource-scheduler`
- 首轮 scope 是否仍严格限定为：
  - `activations first`
  - 可选 `grad_norm`
  - `one-shot / single-card / short validation-smoke`
  - 不引入 `cross-attention` 干预
  - 不引入 ablation/reactivation
  - 不引入训练
  - 不引入 full sweep
  - 不引入任何 admitted / mechanism claim

未来如果真的进入重审申请，仍只能走 `LocalOps/paper-resource-scheduler/gpu-resource-requests.md`，并继续服从：

- `model > markdown`
- `requested -> running -> released`

但这份 sketch 本身不构成任何 run 授权。

## 当前 Verdict

- `implementation_sketch = ready`
- `code_change = not started`
- `gpu_release = none`
- `next_required_decision = whether to stop at sketch or draft a minimal adapter patch`

当前结论固定为：

这份文档只把未来 `activation-only canary` 的实现草案写成 review-ready 形状，不构成 run release，也不构成实现已开始。
