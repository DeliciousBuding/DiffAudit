# Research Storage Boundary

这份文档只回答一个问题：

`Research` 里的外部代码、数据集、权重、lane 资产、run 结果，分别应该放在哪里？

---

## 一句话规则

- `external/` 放 **外部代码仓 / 本地探索 clone**
- `third_party/` 放 **仓库真实依赖的最小 vendored 代码**
- `D:\Code\DiffAudit\Download\` 放 **原始下载物**
- `workspaces/<lane>/assets/` 放 **lane 归一化后的 admitted 资产入口**
- `workspaces/<lane>/runs/` 放 **运行产物和 evidence**

不要把这些角色混在一起。

---

## 1. `Research/external/`

放什么：

- 上游论文代码仓 clone
- 本地 exploratory mirror
- 不准备直接提交进主代码的外部实现

例如：

- `external/PIA`
- `external/CLiD`
- `external/Reconstruction-based-Attack`
- `external/DiT`

不该放什么：

- 你的 canonical run 结果
- 仓库自己的主代码
- 长期堆积的原始数据集压缩包
- 大量无结构的 checkpoint 仓库
- `downloads/` 这类 raw-intake 聚合目录

规则：

- `external/` 是“上游代码工作区”，不是“数据仓库”
- 能通过 `--repo-root` 指过去的上游实现，优先放这里
- 除非某个上游 release 与代码目录强耦合，否则不要把大数据集长期塞进这里
- 如果看到 `Research/external/downloads/`，默认就是边界漂移；应迁回 `D:\Code\DiffAudit\Download\`

---

## 2. `Research/third_party/`

放什么：

- 已经被当前仓库代码真实依赖的最小 vendored 子集

例如：

- `third_party/secmi/`

不该放什么：

- 整个上游大仓随手复制
- 仅用于临时试验的 clone
- 数据集、权重、训练结果

规则：

- 只有“必须 vendored 才方便维护或集成”的最小代码子集才进 `third_party/`
- 这里是 repo 内长期维护代码，不是临时探索区

---

## 3. `D:\Code\DiffAudit\Download\`

放什么：

- 原始下载的 dataset
- 原始下载的 model weights
- zip / tar / supplementary release
- author release mirror
- 尚未归一化的 checkpoint 或论文附件

例如：

- `Download/shared/datasets/...`
- `Download/shared/weights/...`
- `Download/gray-box/weights/...`
- `Download/black-box/supplementary/...`
- `Download/white-box/supplementary/...`

规则：

- `Download/` 是机器本地的“原始 intake 层”
- 这里负责保存“大、原始、可替换、可重下”的东西
- 不要把 `Download/` 当成 run evidence 层
- 不要在 `Download/` 里写研究结论

---

## 4. `Research/workspaces/<lane>/assets/`

放什么：

- 某条 lane 已经归一化、可被当前 repo 消费的 admitted 资产入口
- manifest
- split metadata
- 小型 contract 文件
- 必要时的 lane-local normalized asset tree

例如：

- `workspaces/gray-box/assets/pia/manifest.json`
- `workspaces/white-box/assets/gsa/...`

规则：

- 这里是“研究仓当前认领并消费的资产层”
- 它可以引用 `Download/` 的原始物，也可以放少量 lane 归一化结果
- 但不要把所有原始下载物再复制一遍

简单判断：

- 如果东西还是“原始下载物”，放 `Download/`
- 如果它已经变成“当前 lane 的稳定 contract 入口”，放 `workspaces/<lane>/assets/`

---

## 5. `Research/workspaces/<lane>/runs/`

放什么：

- 每次实验/评估/run 的输出
- `summary.json`
- 导出的 scores / board / packet summaries
- 当前任务的 machine-readable evidence

规则：

- `runs/` 只承载 evidence，不承载上游代码
- 不要把原始数据集或大模型仓库塞进 `runs/`
- 每个任务都应能指向一个 canonical evidence anchor

---

## 6. 当前仓库里的特例

### `SecMI`

- `external/SecMI/` = full upstream clone，用于上游 config / split / layout reference
- `third_party/secmi/` = 当前 DiffAudit 集成真正使用的最小 vendored surface

两者可以并存，但角色必须显式区分，不能混称成“SecMI 就在 external 或 third_party”。

### `CLiD`

- `external/CLiD/` = 本地 working clone / `--repo-root`
- `D:\Code\DiffAudit\Download\black-box\supplementary\clid-mia-supplementary\` = raw supplementary mirror

也就是说：

- 跑本地桥接或看上游代码，去 `external/CLiD`
- 读 raw supplementary / `inter_output`，去 `Download`

### `recon-assets`

`recon-assets` 已经从 `external/` 迁出，当前 canonical raw bundle 位置是：

- `D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets\`

它不再属于代码 clone 层。

## 7. 当前最容易乱的地方

### 情况 A：把数据集塞进 `external/`

短期看方便，长期最乱。

正确做法：

- 原始下载先进 `D:\Code\DiffAudit\Download\`
- 当前 lane 真正消费的入口再在 `workspaces/<lane>/assets/` 或 manifest 里指过去

### 情况 B：把 run 结果写回 `Download/`

不对。

`Download/` 只保存原始 intake，不保存研究 verdict。

### 情况 C：把 repo 真实依赖代码和探索 clone 混在一起

正确分法：

- 当前仓库真实依赖、要一起维护的最小子集 -> `third_party/`
- 上游整仓、本地 exploratory clone -> `external/`

---

## 8. 建议的今后纪律

1. 新拿到一个外部代码仓：
   - 先放 `external/`
2. 新拿到一个原始数据集/权重/附件：
   - 先放 `D:\Code\DiffAudit\Download\`
3. 某条 lane 要把原始物变成稳定消费入口：
   - 在 `workspaces/<lane>/assets/` 建 manifest 或归一化入口
4. 真正执行后的证据：
   - 只写到 `workspaces/<lane>/runs/`
5. 只有在“当前仓库代码必须长期直接 import / patch”的情况下：
   - 才把最小必要代码放进 `third_party/`

---

## 9. 当前结论

如果你问“外部代码和数据集到底该放哪”：

- 外部代码默认放 `Research/external/`
- 原始数据集和原始权重默认放 `D:\Code\DiffAudit\Download\`
- 当前研究主线真正消费的 lane 入口放 `Research/workspaces/<lane>/assets/`
- 运行结果放 `Research/workspaces/<lane>/runs/`
- 只有最小 vendored 依赖才进 `Research/third_party/`



