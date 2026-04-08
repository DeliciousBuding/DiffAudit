# 灰盒工作区

用于存放：

- 灰盒或半白盒论文阅读笔记
- 部分可观测信息下的攻击设定
- 解释增强类实验思路
- 灰盒方向任务认领

## 当前建议任务

1. 当前灰盒主讲线是 `PIA`，先看 `2026-04-07-pia-runtime-mainline.md` 和 `assets/pia/manifest.json`
2. 当前灰盒 baseline 是 `SecMI`，目标是尽快判定它能否进入真实资产闭环
3. 当前最重要的新增工作不是再补 smoke，而是围绕 `GPU128` 对照继续验证现有 defense prototype 能否正式定义成 `G-1`
4. `PIA` 论文必须自己读懂，先写清楚它依赖什么信号完成攻击，再设计为什么 `G-1` 应该有效
5. 在 `PIA baseline + defended` 的正式对照表没出来前，不扩 `CelebA-HQ / ImageNet-64`
6. 综合进度口径统一看 `../../docs/comprehensive-progress.md`
