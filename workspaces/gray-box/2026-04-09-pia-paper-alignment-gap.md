# 2026-04-09 PIA Paper-Alignment Gap

## 当前结论

`PIA` 当前已经是 `workspace-verified` 的灰盒主讲线，但还不是 `paper-aligned`。

当前唯一仍未闭环的阻塞不是运行链，也不是 `all_steps / late_steps_only` 的选择，而是 `checkpoint/source provenance`。

截至 `2026-04-10 round-29`，这个 blocker 进一步收紧为两类硬问题：

- `release/source identity unresolved`
- `CIFAR10 random-four-split / four-model tau-transfer protocol mismatch vs local single fixed ratio0.5 split`

## 已经闭环的部分

- canonical local checkpoint、dataset root、member split 已可被当前适配器稳定加载
- baseline 与 `provisional G-1 = stochastic-dropout (all_steps)` 已形成 admitted 主讲闭环
- adaptive review、quality/cost、manifest summary、unified table、contract-specific best summary 已经能互相对齐
- 当前口径可以安全写成 `workspace-verified`

## 仍然不能升级为 `paper-aligned` 的原因

- 当前 checkpoint/source provenance 仍是导入后保留的本地来源，不是已经被单独核准的 paper-faithful release bundle
- 现有 evidence 能证明“本地真实资产主线可运行”，不能证明“checkpoint/source provenance 已与论文声明一一对齐”
- 当前新增的 strict redo 还证明：`2026-04-09` 的 strict-clean 只能算历史快照，当前 repo 已因 pycache 漂移而不再 strict-clean
- 因此当前只允许写：
  - `workspace-verified`
  - `provisional G-1`
  - `paper-alignment blocker = checkpoint/source provenance`

不允许写：

- `paper-aligned defense benchmark`
- `paper-faithful reproduction complete`
- `validated privacy win`

## 当前主讲口径

- attack baseline: `pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive`
- defended mainline: `pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive`
- `late_steps_only` 只保留为质量优先消融，不替代 defended mainline

## 下一步边界

只有在 checkpoint/source provenance 被单独核准之后，`PIA` 才允许从 `workspace-verified` 升级到 `paper-aligned`。

该 blocker 的可审查 dossier 已收口在：

- `workspaces/gray-box/2026-04-09-pia-provenance-dossier.md`
- `workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md`

截至 `2026-04-10` 的当前 closed decision 为：

- `remain long-term blocker`

在此之前，任何系统、文档、PPT、状态页都必须把这条线写成：

- `workspace-verified`
- `paper-aligned blocked by checkpoint/source provenance`
