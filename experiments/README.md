# 实验输出说明

这里存放可提交的最小实验输出和 smoke 结果。

## 当前内容

- `secmi-smoke/summary.json`：最小 smoke pipeline 输出
- `secmi-synth-smoke/summary.json`：CPU synthetic smoke 输出
- `secmi-synth-smoke-gpu/summary.json`：GPU synthetic smoke 输出
- `pia-runtime-smoke-cpu/summary.json`：CPU PIA runtime smoke 输出
- `pia-runtime-smoke-gpu/summary.json`：GPU PIA runtime smoke 输出
- `pia-synth-smoke-cpu/summary.json`：CPU PIA synthetic smoke 输出
- `pia-synth-smoke-gpu/summary.json`：GPU PIA synthetic smoke 输出
- `clid-dry-run-smoke/summary.json`：CLiD dry-run smoke 输出
- `clid-artifact-summary/summary.json`：CLiD 上游 artifact 复算输出
- `recon-eval-smoke/summary.json`：纯黑盒 reconstruction 评估 smoke 输出
- `recon-artifact-summary/summary.json`：reconstruction score artifact 复算输出
- `recon-upstream-eval-smoke/summary.json`：reconstruction 上游 `test_accuracy.py` smoke 输出
- `variation-synth-smoke/summary.json`：API-only 黑盒 variation synthetic smoke 输出
- `blackbox-status/summary.json`：当前黑盒所有结果的统一汇总
- `templates/`：统一结果结构模板

## 原则

- 不提交私有数据集或大体积中间产物
- 可以提交最小 `summary.json` 作为运行证据
- 真实 benchmark 结果要明确写清楚资产前提和攻击假设

## 当前建议

1. 新增实验目录时，优先只提交 `summary.json`。
2. 如果一个 smoke 会生成临时配置、权重或图片，默认在命令结束时清理。
3. 目录命名尽量直接表达：
   - 方法
   - 模式
   - 设备或场景

例如：`pia-runtime-smoke-gpu`、`clid-artifact-summary`、`recon-eval-smoke`。
