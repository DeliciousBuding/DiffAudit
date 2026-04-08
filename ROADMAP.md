# DiffAudit Roadmap

## 目标

这份路线图只面向 `Project` 研究仓库。

统一目标：

- 让黑盒、灰盒、白盒三条攻击线进入同一套研究与证据规划
- 让攻击、资产、manifest、summary、防御结果都能被统一记录
- 在当前阶段形成“至少一条可辩护攻击主线 + 至少一条可比较防御原型 + 一张 admitted 统一总表”

当前不追求“所有论文都浅尝一下”，而追求“把最值得讲、最有证据的几条线真正打穿并冻结口径”。

## 当前综合判断

截至 `2026-04-09`：

- 黑盒：
  - `recon` 已形成冻结主证据口径
  - `variation` 已形成正式本地次主线，但真实 API 资产 blocked
  - `CLiD` 保留为补充材料
- 灰盒：
  - `PIA` 是当前最成熟、最适合做“攻击 + 防御”主讲闭环的一条线
  - `PIA` 已完成 `GPU128 / GPU256 / GPU512` baseline + defense 与一次 `GPU512` repeat
  - `PIA` provenance 当前已可写成 `workspace-verified`
  - `PIA` 攻击依赖信号与灰盒成本口径现已补齐到正式说明
  - 新归档 `TMIA-DM` 进一步补强了时间相关噪声 / 梯度信号这一灰盒文献轴
  - `SecMI` 已明确为 `blocked baseline`
- 白盒：
  - `GSA 1k-3shadow` 已形成强攻击结果
  - `W-1 = DPDM` 已完成 `strong-v2 full-scale` 与 `strong-v3 full-scale` defended comparator
  - 当前 defended 主口径冻结为 `strong-v3 full-scale`
- 防御：
  - 灰盒 `G-1` 当前固定为 `provisional G-1 = stochastic-dropout`
  - 白盒 `W-1` 已进入 admitted 总表
  - `B-1`、`B-2`、`W-2`、`G-2` 仍在 backlog

## 当前优先级

固定顺序：

1. 维持 `PIA + provisional G-1` admitted 主讲口径
2. 维持 `recon` 冻结口径
3. 维持 `variation = secondary track + blocked real-API assets`
4. 维持 `GSA + W-1 strong-v3 full-scale` 白盒口径
5. 继续维持统一 attack-defense 总表中的 `quality/cost` 与主讲线机理说明一致
6. 让 `Local-API` 继续消费 admitted 结果

## 主路线

### Phase 1: 黑盒证据线稳态化

目标：

- 维持 `recon` 作为当前 black-box 主证据线

当前主线：

- `recon`

当前要求：

1. 固化 `main evidence / best single metric reference / secondary track`
2. 不再把 `variation` 写成真实 API 闭环
3. 暂不继续无节制扩模型覆盖
4. 暂不把 `B-1 / B-2` 伪装成已落地结果

完成标准：

- 黑盒文档、总表、状态页对同一口径描述一致

### Phase 2: 灰盒主讲线冻结

目标：

- 让灰盒形成当前最稳的 admitted “攻击 + 防御”闭环

当前主线：

- `PIA`

当前 baseline：

- `SecMI`

当前要求：

1. 维持 `PIA` 的 canonical roots、manifest、summary 口径
2. 把当前 provenance 固定为 `workspace-verified`
3. 把 `stochastic-dropout` 固定为 `provisional G-1`
4. 保持 `SecMI = blocked baseline`
5. 当前不为重复已知结论继续重跑 `PIA`

完成标准：

- 灰盒 admitted 主讲线稳定
- 下一轮是否重启 GPU 具有明确触发条件

### Phase 3: 白盒深度线冻结

目标：

- 让白盒保持“强攻击 + defended 对照”结构，而不是继续做边际重跑

当前主线：

- `GSA`

当前防御候选：

- `W-1 = DPDM / Diffusion-DP`

当前要求：

1. 固定 `GSA 1k-3shadow` 为攻击主结果
2. 固定 `strong-v3 full-scale` 为 defended 主 rung
3. 保留 `strong-v2 full-scale` 为参考 rung
4. 暂缓 `W-2`
5. 当前不继续抢 GPU

完成标准：

- 白盒主讲口径稳定
- 统一总表只引用必要 rung

### Phase 4: 统一评估表

目标：

- 把不同线的 admitted 结果收口成一份综合对比表

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

- admitted 结果可直接被系统读取
- blocked / smoke / secondary 状态不混入主对比表

## 工作区分工口径

- `workspaces/black-box/`
  - 负责 black-box evidence line
- `workspaces/gray-box/`
  - 负责 `PIA` 与灰盒防御
- `workspaces/white-box/`
  - 负责 `GSA` 与 `W-1`
- `workspaces/implementation/`
  - 负责 admitted 统一总表

## 当前执行纪律

- 不把 `smoke / preview / toy` 写成论文复现成功
- admitted 结果优先写 machine-readable `manifest / summary / table`
- 在新的真实资产到位前，不扩 `SecMI` 和 `variation` 的真实运行面
- 每次重要状态变化都同步：
  - `docs/reproduction-status.md`
  - `docs/comprehensive-progress.md`
  - `docs/local-api.md`
  - 对应工作区主文档

## 下一步

当前最短路径：

1. 持续维持 admitted 口径一致
2. 保持 `PIA` 主讲线的机理说明、成本列和状态页一致
3. 把 `TMIA-DM` 保持为灰盒候选论文，不误写成黑盒主线
4. 继续让 `Local-API` 读取 admitted 结果，而不是只盯 `recon`
5. 等新的真实资产或新的研究问题出现后，再安排下一轮 GPU 任务

综合进度入口见 [docs/comprehensive-progress.md](docs/comprehensive-progress.md)。
