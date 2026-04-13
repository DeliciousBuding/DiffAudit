# Research AGENTS

这是 `Research/` 研究仓库的仓库级协作文件。

它面向两类协作者：

- 人类队友
- 在 `Research/` 内工作的 AGENT / Codex / Claude / 其他自动化代理

目标只有一个：

- 让新协作者拿到仓库后，能在不依赖某一台固定机器的前提下，继续推进黑盒、灰盒、白盒三条主线

---

## 1. 仓库边界

`Research` 是纯研究仓库。

它负责：

- 论文阅读与索引
- 攻击 / 防御研究代码
- 实验配置
- 实验产物与证据
- 研究工作区文档

它不负责：

- 本机飞书运维
- 服务器部署细节
- 平台前端展示逻辑
- Local-API 服务端发布规则

如果任务已经变成：

- 公网访问
- 平台页面
- 服务发布

那就应该切去 `Platform/` 或 `Services/Local-API/`，不要继续在 `Research/` 里扩。

---

## 2. 当前总判断

截至当前，`Research` 的算法线应该这样理解：

- `recon`
  - 当前最强 black-box 证据线
- `variation / Towards`
  - 当前正式本地 black-box 次主线
  - 已有本地 CPU synthetic smoke
- `PIA`
  - 当前最成熟、最适合作为“攻击 + 防御”主讲闭环的一条线
- `SecMI`
  - 当前 gray-box baseline
  - 需要尽快做 promote / block 判定
- `GSA`
  - 当前 white-box 主线
  - 已到 real-asset closed loop ready
- `DPDM`
  - 当前 `W-1` 防御候选

不要再用旧口径理解：

- 不是“仓库只做黑盒”
- 也不是“三条线平均推进”

当前执行顺序是：

1. `PIA`
2. `recon`
3. `variation / Towards`
4. `GSA`
5. `W-1 / DPDM`

---

## 3. 队友 / AGENT 接仓第一步

无论是人还是 AGENT，第一次进入仓库都先做这几步：

1. 阅读 [README.md](README.md)
2. 阅读 [docs/teammate-setup.md](docs/teammate-setup.md)
3. 阅读 [docs/comprehensive-progress.md](docs/comprehensive-progress.md)
4. 阅读 [docs/mentor-strict-reproduction-plan.md](docs/mentor-strict-reproduction-plan.md)
5. 进入自己负责的工作区：
   - [workspaces/black-box/README.md](workspaces/black-box/README.md)
   - [workspaces/gray-box/README.md](workspaces/gray-box/README.md)
   - [workspaces/white-box/README.md](workspaces/white-box/README.md)

不要一进来就直接改代码。

---

## 4. 环境与可移植性规则

### 4.1 环境原则

必须默认：

- 队友机器路径和你机器不同
- 队友未必已经装好 editable package
- 队友未必已有 `external/` 中的本地 clone
- 队友未必已有任何 checkpoint / dataset / query-image 资产

所以：

- 共享配置里不写个人真实路径
- 文档里不默认某个本机目录存在
- 命令优先写成仓库相对路径

### 4.2 标准环境流程

统一使用：

- `environment.yml`
- [scripts/bootstrap_research_env.py](scripts/bootstrap_research_env.py)
- [scripts/verify_env.py](scripts/verify_env.py)

推荐顺序：

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
python scripts/verify_env.py
python -m diffaudit --help
```

### 4.3 本地资产模板

不要在共享配置中填写本机路径。

统一使用：

- [configs/assets/team.local.template.yaml](configs/assets/team.local.template.yaml)

队友应复制为：

- `configs/assets/team.local.yaml`

并且：

- `team.local.yaml` 只在本地存在
- `team.local.yaml` 不提交

### 4.4 CLI 可移植性规则

如果你修改 [src/diffaudit/cli.py](src/diffaudit/cli.py)：

- 不要在顶层导入所有重模块
- 重依赖模块优先按需导入

原因：

- black-box 入口不应因为 white-box 的 `torch` / `gsa` 依赖而无法导入
- 单条方法线测试应尽量能独立通过

---

## 5. 三条线怎么继续

### 5.1 黑盒

当前目标：

- `recon` 继续做主证据线
- `variation / Towards` 继续做正式本地次主线

黑盒内部不要再混入：

- `SecMI`
- `PIA`

它们属于灰盒。

黑盒当前可直接引用的文档：

- [workspaces/black-box/plan.md](workspaces/black-box/plan.md)
- [workspaces/black-box/2026-04-08-variation-local-track.md](workspaces/black-box/2026-04-08-variation-local-track.md)

### 5.2 灰盒

当前目标：

- `PIA baseline + defended`
- `SecMI` promote / block 判定

灰盒当前不是去补更多 smoke，而是：

- 稳定 `PIA` baseline
- 正式定义 `G-1`
- 做前后对比

当前文档：

- [workspaces/gray-box/plan.md](workspaces/gray-box/plan.md)
- [workspaces/gray-box/2026-04-07-pia-runtime-mainline.md](workspaces/gray-box/2026-04-07-pia-runtime-mainline.md)

### 5.3 白盒

当前目标：

- `GSA` 扩样本、提训练强度
- `DPDM -> W-1`

白盒当前最重要的不是“能不能跑”，而是：

- 结果能不能脱离随机附近

当前文档：

- [workspaces/white-box/plan.md](workspaces/white-box/plan.md)
- [workspaces/white-box/2026-04-07-gsa-runtime-mainline.md](workspaces/white-box/2026-04-07-gsa-runtime-mainline.md)

---

## 6. 证据分级规则

必须严格区分：

- `research-ready`
- `code-ready`
- `evidence-ready`
- `asset-ready`
- `benchmark-ready`

禁止混写：

- `smoke`
- `preview`
- `toy closed-loop`
- `paper reproduction`

没有真实运行证据，不得写：

- “复现成功”
- “论文结果已验证”

---

## 7. 文档同步规则

新增主线、重大结果或重大判断变化时，至少同步：

- [README.md](README.md)
- [ROADMAP.md](ROADMAP.md)
- [docs/reproduction-status.md](docs/reproduction-status.md)
- [docs/comprehensive-progress.md](docs/comprehensive-progress.md)
- 对应工作区 `plan.md` 或 README

仓库必须始终能回答：

- 现在做到哪了
- 差什么
- 缺什么资产
- 用什么命令验证

---

## 8. 工作区规则

多人协作统一按工作区拆分：

- `workspaces/black-box/`
- `workspaces/gray-box/`
- `workspaces/white-box/`
- `workspaces/implementation/`

这些目录只放：

- 阅读笔记
- 复现计划
- 任务归属
- 阻塞记录
- 阶段总结

共享可执行代码统一放在：

- `src/diffaudit/`

---

## 9. 代码与第三方规则

### 9.1 优先做什么

优先补：

- 配置 schema
- planner / probe
- dry-run
- 最小 smoke
- 结果记录
- import 边界与测试

不要优先做：

- 前端展示
- 平台化用户系统
- 与真实攻击路径无关的大框架

### 9.2 第三方代码

规则：

- `third_party/`
  - 只放最小必要 vendored 子集
- `external/`
  - 只放本地探索 clone
  - 不提交

要求：

- 必须保留来源说明
- 只做必要补丁
- 不随意重写 upstream

---

## 10. 多人协作纪律

- 一次尽量只负责一个方向
- 共享接口变更前先补文档
- 提交要小步快跑
- 被资产阻塞时必须显式记录
- 不要把个人机器路径、token、私有资产写进共享文件

---

## 11. 输出风格

研究结论优先按这四项组织：

- 假设
- 方法
- 证据
- 阻塞项

不要使用这类模糊表述：

- “应该可以”
- “差不多能跑”
- “基本算复现了”

---

## 12. 对队友的直接判断

现在队友已经可以直接上手，但前提是按标准流程接仓：

- 先跑环境 bootstrap
- 先复制 `team.local` 模板
- 先跑各线 `plan / probe / dry-run`
- 再开始继续复现

不是“拉下来就能立刻跑全套主线”，但已经不是“必须依赖你这台机器手把手带”。
