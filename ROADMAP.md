# DiffAudit Roadmap

## 目标

这份路线图面向一个明确目标：

- 先把黑盒主线做成项目里第一个接近论文级复现的板块
- 再补齐 `Stable Diffusion / Kandinsky / DiT` 三条模型线的证据层级
- 最后再扩展到第二黑盒路线、灰盒强化和白盒准备

当前不追求“所有论文都碰一下”，而追求“先把一条主线做深做硬”。

## 当前状态

截至 `2026-04-06`，项目已经从研究整理阶段进入真实证据积累阶段。

已完成的关键里程碑：

- `Stable Diffusion v1.5 + DDIM`
  - `1-sample runtime-mainline`
  - `10-sample public runtime-mainline`
  - `25-sample public runtime-mainline`
  - `50-sample public runtime-mainline`
  - `100-sample public runtime-mainline`
- `Kandinsky v2.2`
  - `1-sample runtime-mainline`
- `DiT-XL/2`
  - 官方 checkpoint `step10 sample-smoke`
- 黑盒统一汇总、状态文档、Feishu 在线索引已经打通

当前最大公开子集证据：

- `recon-runtime-mainline-ddim-public-100-step10`
  - `auc = 0.788`
  - `asr = 0.63`
  - `tpr@1%fpr = 0.99`

当前最佳判别指标仍来自：

- `recon-runtime-mainline-ddim-public-50-step10`
  - `auc = 0.866`
  - `asr = 0.51`
  - `tpr@1%fpr = 1.0`

当前主阻塞：

- 公开资产的 `target/shadow/member/non-member` 语义映射仍未最终核准
- `variation` 与 `CLiD` 还没有真实资产闭环
- `DiT` 还只有官方采样证据，没有进入成员推断协议
- 白盒仍缺 checkpoint / gradient / activation 访问条件

## 总路线

### Phase 1: 做硬黑盒主线

目标：

- 把 `NDSS 2025 recon` 这条线从“小规模可运行”推进到“公开中样本稳定复现”

执行顺序：

1. 将 `DDIM public-25` 固化为当前基准
2. 扩到 `DDIM public-50`
3. 视资源扩到 `DDIM public-100`
4. 在每一档记录：
   - 样本规模
   - 采样步数
   - 总耗时
   - 生成图片数量
   - score artifact 体积
   - `auc / asr / tpr@1%fpr`

完成标准：

- 至少一档 `50+` 样本的 `Stable Diffusion + DDIM runtime-mainline`
- 所有结果都有 `runtime-mainline + artifact-mainline + blackbox-status` 三层证据

### Phase 2: 补齐模型覆盖

目标：

- 把“三个模型都跑过”提升为“三个模型都有明确证据等级”

执行顺序：

1. `Kandinsky`
   - 从 `1-sample` 扩到 `10/10`
   - 如资源允许，再扩到 `25/25`
2. `DiT`
   - 从 `step10 sample-smoke` 提升到更高步数
   - 优先 `step50`
   - 再评估 `step100` 或 `512` 分辨率

完成标准：

- `SD/DDIM`: 中样本 `runtime-mainline`
- `Kandinsky`: 不再只是最小 smoke
- `DiT`: 不再只是 `step10`

### Phase 3: 第二黑盒路线闭环

目标：

- 推进 `Towards Black-Box Membership Inference Attack for Diffusion Models`

执行原则：

- 这条线现在是第二优先级
- 在 `NDSS recon` 形成中样本稳定结果前，不抢主线资源

推进内容：

- 搜索真实可用 variation / image editing 接口或等价本地代理协议
- 明确 query budget、评估协议和对照指标
- 补可复用的数据输入规范

完成标准：

- 不是只有文档理解，而是进入真实资产可运行阶段

### Phase 4: 灰盒与白盒不断档维护

目标：

- 保持后续扩展空间，但不让它们抢黑盒主线

灰盒：

- `SecMI / PIA` 维持代码可运行
- 有真实 checkpoint 时优先补单条高质量复现，不平均铺开

白盒：

- 继续积累 checkpoint、训练配置、梯度接口信息
- 等真实访问条件具备后再进主执行队列

## 执行原则

- 主线优先级固定为：
  - `NDSS recon > DiT/Kandinsky 模型覆盖 > variation/CLiD > gray-box > white-box`
- 每次只推进一个主实验和一个次实验
- 每次实验完成后必须同步：
  - `experiments/*/summary.json`
  - `experiments/blackbox-status/summary.json`
  - `docs/reproduction-status.md`
  - `workspaces/black-box/plan.md`
  - `docs/paper-reports/master-feishu-index.md`
  - Feishu 在线文档
- GPU 任务必须严格走调度器申请与释放
- 每个阶段至少形成一次小提交并立即推送

## 下一步

当前最短路径：

1. 核准公开资产的 `target/shadow/member/non-member` 语义映射，并明确 `public-50` 与 `public-100` 的解释口径
2. 把 `Kandinsky` 从 `1-sample` 扩到 `10/10`
3. 把 `DiT` 从 `step10` 扩到 `step50`

如果必须只做一件事：

- 先做 `DiT step50` 或 `Kandinsky 10/10`

因为公开 `recon` 子集已经推进到 `100-sample` 上限，下一步更有价值的是补齐模型覆盖，并回头解释 `public-50` 与 `public-100` 指标差异。
