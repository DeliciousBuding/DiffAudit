# Research Repo Cleanup Status

更新时间：`2026-04-13`

## 当前判断

`Research` 仓库已经有一批与 `SMP-LoRA` 稳定性探针直接相关的改动完成提交并推送，但剩余脏树仍然很大，不能继续靠肉眼处理。当前最合理的做法是把剩余改动分成可执行批次，而不是尝试一次性“洗干净整仓”。

## 已完成的清理动作

- 已将本轮 `SMP-LoRA` 研究批次单独提交到：
  - `020b37f research: seal smp-lora stability probes`
- 已将 `outputs/` 下默认生成的训练目录加入忽略规则，只保留 `evaluation.json`
- 已将 `workspaces/white-box/assets/` 默认纳入忽略规则，只保留 `manifest*.json / README.md / HANDOFF.md`

## 当前剩余脏树分层

### 1. `docs/` 大批量文档改动

- 数量最多
- 主要是论文报告、索引、协作文档、Feishu 关联文档
- 风险：
  - 这些文件跨度大
  - 很可能混有多轮历史改动
  - 不适合和研究主线实验批次混在一个提交里

### 2. `experiments/` summary.json 批量变更

- 多数是现成实验目录下的 `summary.json`
- 更像历史实验状态回写
- 风险：
  - 很难在当前轮次验证每个 summary 的语义边界
  - 应该按实验线单独核对，而不是仓库清洁时顺手提交

### 3. `workspaces/` 历史研究文档与资产清单

- 包含 gray-box / white-box / intake / implementation 多条线
- 很多属于长期文档治理，不是一次研究 run 的自然副产物

### 4. `scripts/` / `tests/` / `src/` 存量未跟踪文件

- 这里是最需要第二轮精确筛选的区域
- 既可能包含有效工具链，也可能混有半成品

## 推荐的后续清理顺序

1. `scripts/ + tests/ + src/`
   - 只处理真正进入当前研究主线的工具链
2. `docs/`
   - 先处理主线/治理文档
   - 论文报告与 OCR 文档后置
3. `experiments/`
   - 只按主线证据包逐条核对
4. `workspaces/`
   - 按 black-box / gray-box / white-box / intake 分批清理

## 当前不做什么

- 不尝试一次性清空整个脏树
- 不把几百个历史文档混成一个提交
- 不把未核验的 `summary.json` 批量推上去
- 不为追求“看起来干净”而回滚已有改动

## 下一步建议批次

### Batch A: toolchain hygiene

- 范围：
  - `scripts/`
  - `tests/`
  - `src/`
- 目标：
  - 锁定哪些工具已进入当前研究主线
  - 哪些只是临时脚本

### Batch B: research docs core

- 范围：
  - `ROADMAP.md`
  - `docs/comprehensive-progress.md`
  - `docs/reproduction-status.md`
  - `docs/mainline-narrative.md`
  - `workspaces/intake/`

### Batch C: historical docs/experiments triage

- 范围：
  - `docs/paper-reports/`
  - `experiments/`
  - 其他长期存量文档

## 备注

这份文档的目的不是宣称仓库已完全干净，而是把“剩余为什么脏、下一步先动哪一层”固化下来，避免下轮重新从 `git status` 的海量输出里找方向。
