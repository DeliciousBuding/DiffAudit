# 灰盒工作区

用于存放：

- 灰盒或半白盒论文阅读笔记
- 部分可观测信息下的攻击设定
- 解释增强类实验思路
- 灰盒方向任务认领

## 当前建议任务

1. 当前灰盒主讲线是 `PIA`，先看 `2026-04-07-pia-runtime-mainline.md` 和 `assets/pia/manifest.json`
2. `SecMI` 现在是独立 corroboration line，不再是单纯 blocked placeholder；先看 `2026-04-15-pia-vs-secmi-graybox-comparison.md` 和 `2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
3. `CDI` 当前应理解为 `gray-box collection-level audit extension`：
   - first internal canary 已落地
   - repaired `2048` paired surface 已落地
   - `control-z-linear` 现在是 default internal paired scorer
   - 但它还不是 headline scorer，也不是外部证据口径
4. 当前最重要的新增工作不是再补 naive fusion，而是保持 `PIA` 的 defended mainline、补第二防御或新 family verdict
5. `PIA` 论文必须自己读懂，先写清楚它依赖什么信号完成攻击，再设计为什么 `G-1` 应该有效
6. 当前 `PIA + SecMI` 的 simple ensemble 已判定 `no-go`，没有新 hypothesis 前不继续烧这条预算
7. 综合进度口径统一看 `../../docs/comprehensive-progress.md`
