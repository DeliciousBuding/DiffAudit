# DiffAudit Roadmap

## 目标

这份路线图面向一个统一目标：

- 把黑盒、灰盒、白盒三条攻击线都纳入同一套研究与证据规划
- 先把黑盒主线做成项目里第一个接近论文级复现的板块
- 在不打断黑盒主执行队列的前提下，持续维护灰盒可运行性与白盒研究准备度
- 再按证据成熟度补齐 `Stable Diffusion / Kandinsky / DiT` 等模型线和后续攻击线

当前不追求“所有论文都碰一下”，而追求“统一规划三条线，但先把黑盒第一主线做深做硬”。

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
  - 官方 checkpoint `step50 sample-smoke`
- 黑盒统一汇总、状态文档、Feishu 在线索引已经打通

当前最大公开子集证据：

- `recon-runtime-mainline-ddim-public-100-step30`
  - `auc = 0.849`
  - `asr = 0.51`
  - `tpr@1%fpr = 1.0`

当前最佳判别指标仍来自：

- `recon-runtime-mainline-ddim-public-50-step10`
  - `auc = 0.866`
  - `asr = 0.51`
  - `tpr@1%fpr = 1.0`

当前主阻塞：

- 公开资产的 `target/shadow/member/non-member` 语义映射仍未最终核准
- `DDIM public-100 step10` 与 `step30` 的指标差异还没有形成统一解释口径
- `Kandinsky 10/10` 当前本机链路异常慢，且还没有拿到能定位首个阶段耗时的有效日志
- `variation` 与 `CLiD` 还没有真实资产闭环
- `DiT` 还只有官方采样证据，没有进入成员推断协议
- 白盒仍缺 checkpoint / gradient / activation 访问条件

## 总路线

统一规划原则：

- 黑盒、灰盒、白盒都属于正式研究路线，不再按“是否以后再做”区分，而按“当前执行优先级”和“资产成熟度”排队
- 共享能力统一复用 `config / planner / adapter / dry-run / summary` 这套仓库骨架，避免把黑盒主线写成唯一长期形态
- 当前执行顺序仍保持分层：先黑盒主证据，再黑盒扩线与模型覆盖，再灰盒高质量闭环，最后白盒进入主执行队列

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

- 把灰盒与白盒持续留在统一路线图内，但不让它们抢黑盒主线

灰盒：

- `SecMI / PIA` 维持代码可运行
- 有真实 checkpoint 时优先补单条高质量复现，不平均铺开
- 进入真实资产阶段后，按“单条高质量闭环优先于多论文铺开”执行

白盒：

- 继续积累 checkpoint、训练配置、梯度接口信息
- 继续约束研究问题、资产要求和最小可验证接口
- 等真实访问条件具备后再进主执行队列，不提前伪造结论

## 执行原则

- 主线优先级固定为：
  - `NDSS recon > DiT/Kandinsky 模型覆盖 > variation/CLiD > gray-box > white-box`
- 统一规划不等于平均分配资源；当前资源仍优先投向黑盒主线，灰盒和白盒按维护态或准备态推进
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

1. 固化公开资产的 `target/shadow/member/non-member` 语义映射，并明确 `public-100 step10` / `step30` 与 `public-50 step10` 的解释口径
2. 维持 `docs/reproduction-status.md`、`workspaces/black-box/plan.md` 与 `experiments/blackbox-status/summary.json` 的事实一致，并明确它们属于三线统一规划下的第一优先执行层
3. 在拿到有效阶段日志前继续暂停 `Kandinsky`，之后再决定是否恢复 `10/10`
4. 灰盒继续维持 `SecMI / PIA` 的可运行与状态可见性，白盒继续累积可访问资产条件，但都不改写当前黑盒优先级

如果必须只做一件事：

- 先把 `DDIM public-100 step30` 的结果口径和阻塞项写死

因为当前最大的缺口不是“缺一次新实验”，而是“已有主证据和状态文档还没完全收口”。`Kandinsky` 继续跑之前，先把 `recon` 公开子集的结论、边界和阻塞写清楚，才能避免后续继续在错误口径上追加证据。
