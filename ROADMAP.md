# DiffAudit Roadmap

## 目标

这份路线图只面向 `Project` 研究仓库。

统一目标：

- 让黑盒、灰盒、白盒三条攻击线进入同一套研究与证据规划
- 让攻击、资产、manifest、summary、防御结果都能被统一记录
- 在当前阶段形成“至少一条可辩护攻击主线 + 至少一条可比较防御原型”

当前不追求“所有论文都碰一下”，而追求“把最值得讲、最有证据的几条线真正打穿”。

## 当前综合判断

截至 `2026-04-07`：

- 黑盒：
  - `recon` 是当前最强证据线
  - `CLiD`、`variation` 已有证据，但都还不是当前最强执行主线
- 灰盒：
  - `PIA` 是当前最成熟、最适合做“攻击 + 防御”主讲闭环的一条线
  - `SecMI` 仍然更像 baseline，而不是当前最该押注的主线
- 白盒：
  - `GSA` 已到 real-asset closed loop ready
  - 但现在更像“闭环打通”，还不是论文级稳定攻击结果
- 防御：
  - 当前最接近正式结果的是灰盒 `G-1` 近似原型
  - `W-1` 的候选实现 `DPDM` 已在本地，但还没接成正式 baseline
  - `B-1`、`B-2`、`W-2`、`G-2` 仍主要停留在设计层

## 当前优先级

固定顺序：

1. `PIA` 攻击基线稳定化
2. `PIA + G-1` 正式防御比较
3. `recon` 主证据口径固化
4. `GSA` 统计稳定化
5. `W-1 / DPDM` 第一版基线
6. `SecMI` 真实资产闭环或明确降级
7. 统一 attack-defense 对比表

## 主路线

### Phase 1: 黑盒证据线稳态化

目标：

- 维持 `recon` 作为当前最强 black-box evidence line

当前主线：

- `recon`

当前要求：

1. 固化 `public-100 step10 / step30` 的解释边界
2. 继续收口 `target/shadow/member/non-member` 的最可辩护语义
3. 暂不继续无节制扩模型覆盖
4. 将 `B-1 / B-2` 只登记为 defense backlog，不伪装成已落地结果

完成标准：

- `recon` 主证据、最佳单指标结果、对照基线三者不再混写

### Phase 2: 灰盒主讲线打穿

目标：

- 让灰盒形成当前最稳的“攻击 + 防御”可讲闭环

当前主线：

- `PIA`

当前 baseline：

- `SecMI`

当前要求：

1. 继续固化 `PIA` 的 canonical roots、provenance 与 summary 口径
2. 重新跑更大样本、更稳定重复次数的 `PIA runtime mainline`
3. 把当前 dropout-style defense prototype 正式定义成 `G-1`
4. 在同一主基线上做出 `PIA baseline vs defended` 正式对比
5. 给 `SecMI` 一个明确判定：
   - 要么推进到真实资产闭环
   - 要么明确降级为 baseline / blocked

完成标准：

- 至少一条灰盒攻击线具备稳定 `runtime mainline`
- 至少一条灰盒防御线具备同口径前后对比

### Phase 3: 白盒深度线补强

目标：

- 让白盒从“闭环已打通”推进到“统计上更可信”

当前主线：

- `GSA`

当前防御候选：

- `W-1 = DPDM / Diffusion-DP`

当前要求：

1. 扩大 `GSA` 的 `target/shadow + member/non-member` bucket 规模
2. 提高 `checkpoint-*` 的训练强度
3. 把 `DPDM` 接成正式 `W-1` 基线
4. 暂缓 `W-2`，直到 `W-1` 有正式结果

完成标准：

- `GSA` 结果不再只是极小样本闭环
- `W-1` 至少有一版可运行 baseline

### Phase 4: 统一评估表

目标：

- 把不同线的结果收口成一份综合进度和统一对比表

统一字段：

- `track`
- `attack`
- `defense`
- `dataset`
- `model`
- `AUC`
- `ASR`
- `TPR@low-FPR`
- `quality/cost`
- `evidence_level`

完成标准：

- 黑盒、灰盒、白盒可以放进同一张对比表
- 攻击与防御前后结果可以直接比较

## 工作区分工口径

- `workspaces/black-box/`
  - 负责 black-box evidence line
- `workspaces/gray-box/`
  - 负责当前主讲线 `PIA` 与灰盒防御
- `workspaces/white-box/`
  - 负责 `GSA` 与白盒防御候选

## 当前执行纪律

- 不把 `smoke / preview / toy` 写成论文复现成功
- `PIA` 与 `GSA` 论文必须自己读懂，不得只跑代码
- 在 `CIFAR-10 + DDPM` 的攻击-防御对比表没站稳前，不扩 `CelebA-HQ / ImageNet-64`
- 每次重要状态变化都必须同步：
  - [docs/reproduction-status.md](docs/reproduction-status.md)
  - [docs/comprehensive-progress.md](docs/comprehensive-progress.md)
  - 对应工作区 README 或主线文档

## 下一步

当前最短路径：

1. 给 `PIA` 补一份真正的“攻击依赖信号”笔记，并据此正式定义 `G-1`
2. 重新跑 `PIA baseline + defended`，扩大样本量与重复次数
3. 对 `SecMI` 做 promote / block 决策
4. 扩大 `GSA` bucket 与训练强度
5. 把 `DPDM` 接成 `W-1` 第一版白盒防御 baseline
6. 产出统一 attack-defense 总表

综合进度入口见 [docs/comprehensive-progress.md](docs/comprehensive-progress.md)。
