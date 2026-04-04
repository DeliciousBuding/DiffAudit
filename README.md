# DiffAudit

DiffAudit 是一个面向扩散模型的隐私风险审计研究仓库。

本项目的目标不是提升图像生成质量，而是研究扩散模型是否会对训练样本产生过强记忆，并围绕成员推断攻击、风险量化、证据整理和审计原型逐步形成一套可复现的研究工作流。

## 项目定位

一句话概括：

`DiffAudit = 面向扩散模型的隐私风险审计与成员推断复现仓库`

当前阶段是研究原型期。仓库会尽量做到代码可验证、流程可复现，但不默认声称已经完整复现所有论文结果。

## 研究方向

仓库当前围绕三条主线组织：

- `black-box`：黑盒成员推断，贴近真实 API 场景
- `white-box`：白盒成员推断，依赖模型内部信息，通常攻击信号最强
- `gray-box`：灰盒或半白盒攻击，介于黑盒与白盒之间

当前默认优先级是 `black-box`。白盒和灰盒要尽量复用相同的配置结构、adapter 结构、dry-run 机制和结果记录方式，避免每条线都各写一套。

## 多人协作模式

这是一个多人协作研究仓库。

仓库按工作区拆分协作：

- `workspaces/black-box/`
- `workspaces/white-box/`
- `workspaces/gray-box/`
- `workspaces/implementation/`

工作区主要用于：

- 阅读笔记
- 复现计划
- 任务认领
- 阻塞记录
- 阶段总结

共享可执行代码统一放在 `src/diffaudit/`。不要把个人草稿、临时实验说明直接堆进源码目录。

## 当前进展

仓库目前已经具备：

- 独立的 GPU 科研环境
- 配置驱动的 smoke pipeline
- `SecMI` 的计划层、资产解析、workspace 校验、adapter 准备和 dry-run 校验
- 最小化 vendored `SecMI` 集成子集
- 可持续扩展的测试基线

仓库目前还没有完成：

- 基于真实 checkpoint 的 `SecMI` 攻击执行
- 黑盒、白盒、灰盒三条线的系统化 benchmark 结果
- 完整的审计报告输出层

## 仓库结构

```text
configs/                 攻击、数据集、基准与实验配置
docs/                    环境说明、协作文档与项目文档
experiments/             实验产物与 smoke 输出
notebooks/               探索性 notebook
references/              参考资料索引与镜像材料
scripts/                 工具脚本与环境验证脚本
src/diffaudit/           主代码
tests/                   单元测试与集成级 smoke 测试
third_party/secmi/       vendored 的最小 SecMI 子集
external/                本地探索 clone，忽略不提交
workspaces/              多人协作工作区
```

## 推荐工作流

新增任何一条攻击线时，推荐按这个顺序推进：

1. 明确目标论文与攻击假设
2. 在 `configs/` 下增加或更新配置样例
3. 先做 planner / adapter
4. 先做 `dry-run`，再声称“可执行”
5. 跑 smoke 验证并记录到 `experiments/`
6. 资产齐全后再跑真实实验

对多人协作，还建议补一层纪律：

1. 先认领工作区
2. 先写计划再改共享代码
3. 共享代码尽量小步提交
4. 阻塞项必须显式记录

对 `SecMI`，推荐命令顺序是：

1. `plan-secmi`
2. `prepare-secmi`
3. `dry-run-secmi`
4. 资产到位后再尝试真实执行

## 环境搭建

创建并激活 conda 环境：

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

环境说明见 [docs/environment.md](/D:/Code/DiffAudit/Project/docs/environment.md)。

环境验证：

```powershell
$env:PYTHONPATH='src;.'
python scripts/verify_env.py
```

## 快速开始

运行 smoke pipeline：

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit run-smoke --config configs/benchmarks/secmi_smoke.yaml --workspace .
```

生成 `SecMI` 计划：

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml
```

准备 `SecMI` adapter 上下文：

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

运行 `SecMI` dry-run：

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

如果缺 checkpoint、flagfile、dataset_root 等真实资产，`dry-run-secmi` 会返回 `blocked` 并直接指出缺失路径。

## 参考资料

仓库已经镜像了一批调研期 PDF，位置在 `references/materials/`。

说明与索引见：

- [references/README.md](/D:/Code/DiffAudit/Project/references/README.md)
- [references/materials/README.md](/D:/Code/DiffAudit/Project/references/materials/README.md)

这些文件当前被纳入 git，是因为团队还处在调研期，需要共享同一套阅读上下文。

## 协作与分支管理

仓库级规范见：

- [AGENTS.md](/D:/Code/DiffAudit/Project/AGENTS.md)
- [CONTRIBUTING.md](/D:/Code/DiffAudit/Project/CONTRIBUTING.md)
- [docs/github-collaboration.md](/D:/Code/DiffAudit/Project/docs/github-collaboration.md)
- [docs/getting-started.md](/D:/Code/DiffAudit/Project/docs/getting-started.md)

推荐的 GitHub 协作方式是：

- 队友加入仓库协作
- 默认不直接推 `main`
- 每个人按方向开分支
- 通过 PR 合并
- `main` 开启分支保护

## 可复现性说明

很多实验依赖外部资产，仓库不会默认包含：

- 训练好的 checkpoint
- `flagfile.txt` 等训练期配置产物
- 真实数据集根目录
- 部分成员划分或私有实验资产

因此仓库支持应理解为三个阶段：

- `code-ready`
- `asset-ready`
- `experiment-ready`

不要把 smoke 或 dry-run 结果当作 benchmark。

## 近期路线图

- 打通 `SecMI` 的函数级执行层
- 增加真实黑盒 benchmark 配置与资产约束
- 接入一篇白盒论文与一篇灰盒论文
- 统一结果 schema 与报告层
- 稳定多人协作流程

## 致谢

本仓库构建在公开研究基础上，尤其参考了 `SecMI`：

- [SecMI](https://github.com/jinhaoduan/SecMI)
