# 贡献规范

DiffAudit 是一个多人协作研究仓库。目标是让多人可以并行推进论文阅读、实验设计、算法复现和工程实现，同时避免把仓库写成个人草稿堆。

## 一、工作区使用规则

按方向进入对应工作区：

- `workspaces/black-box/`
- `workspaces/white-box/`
- `workspaces/gray-box/`
- `workspaces/implementation/`

工作区主要放：

- 阅读笔记
- 复现计划
- 任务清单
- 阻塞说明
- 阶段总结

共享代码统一放：

- `src/diffaudit/`
- `configs/`
- `tests/`
- `scripts/`

## 二、分支管理

推荐分支命名：

- `black-box/<topic>`
- `white-box/<topic>`
- `gray-box/<topic>`
- `implementation/<topic>`
- `docs/<topic>`

示例：

- `black-box/secmi-cifar10`
- `white-box/paper-intake`
- `implementation/result-schema`

不建议大家直接在 `main` 上开发。

## 三、提交规范

推荐 commit 前缀：

- `feat:`
- `fix:`
- `docs:`
- `test:`
- `chore:`

要求：

- 一次提交只解决一个明确问题
- 不要把阅读笔记、代码重构、实验输出混在同一个 commit 里
- 能拆就拆，不要大包提交

## 四、Pull Request 规范

推荐流程：

1. 从 `main` 拉新分支
2. 在自己的方向分支工作
3. 提交 PR
4. 由队友 review 后合并

PR 描述至少写清楚：

- 做了什么
- 为什么做
- 怎么验证
- 还差什么

当前仓库默认 PR 模板位于：

- `.github/PULL_REQUEST_TEMPLATE.md`

提交 PR 前，建议先本地跑：

```powershell
python scripts/run_local_checks.py --fast
```

## 五、实验命名规范

配置文件建议命名：

- `configs/attacks/<method>_<purpose>.yaml`
- `configs/benchmarks/<method>_<purpose>.yaml`

实验输出统一写到：

- `experiments/<run-name>/`

推荐 run name：

- `secmi-smoke`
- `secmi-cifar10-blackbox`
- `whitebox-ddim-recon`

## 六、资产管理

不要提交这些内容：

- 私有数据集
- 未授权 checkpoint
- 本地临时 clone
- 个人 scratch 文件

允许提交的例外：

- 经过确认要共享的镜像参考资料
- 为了集成而保留的最小 third-party 子集

如果资产缺失，不要硬写“复现完成”。应当：

- 返回 `blocked`
- 记录缺失路径
- 保留 dry-run

## 七、第三方代码规则

`third_party/` 只放最小必要 vendored 子集。

要求：

- 保留来源
- 只做必要补丁
- 不要把上游整个仓库全拷进来

`external/` 是本地探索区，不进 git。

## 八、测试要求

提交前最少运行：

```powershell
conda run -n diffaudit-research python -m unittest tests.test_attack_registry tests.test_smoke_pipeline
```

新增能力时，优先补：

- 配置测试
- planner / adapter 测试
- dry-run 测试
- vendored import smoke test

仓库当前也已经有最小 GitHub Actions 工作流，默认跑：

- `python -m diffaudit --help`
- `python -m unittest tests.test_attack_registry tests.test_smoke_pipeline`
- `python scripts/bootstrap_research_env.py`
- `python scripts/render_team_local_configs.py --team-local configs/assets/team.local.template.yaml --output-dir tmp/configs/rendered-ci`
- `pytest tests/test_variation_attack.py tests/test_render_team_local_configs.py -q`

## 九、本地 hook / pre-commit

仓库当前提供：

- `.pre-commit-config.yaml`
- `scripts/run_local_checks.py`

推荐安装：

```powershell
conda activate diffaudit-research
pre-commit install
```

安装后，提交前会自动执行基础文本检查与最小本地检查链。

## 十、GitHub 协作建议

最合适的团队模式是：

- 队友加入仓库协作
- 给 `Write` 或 `Maintain` 权限
- `main` 开分支保护
- 默认通过 PR 合并

具体操作说明见 [docs/github-collaboration.md](docs/github-collaboration.md)。

根级 GitHub 设置基线见：

- `docs/github-settings-baseline.md`

## 十一、Copilot Review 规则

仓库已启用 Copilot code review。

使用规则：

- 把 Copilot 当成第一轮静态 reviewer，不把它当最终结论
- 高优先级关注 `src/diffaudit/`、`tests/`、`scripts/`、`configs/`
- 低优先级路径包括 `references/`、论文 PDF、长篇阅读笔记、纯实验产物
- 如果 Copilot 对研究状态、复现完成度、paper-aligned 口径提出质疑，优先认真核对
- 如果 Copilot 只给风格建议，而没有行为风险、复现风险或路径风险，可以低优先处理

仓库级审查指令位于：

- `.github/copilot-instructions.md`
