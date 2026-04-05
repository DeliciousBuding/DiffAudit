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
- `PIA` 的计划层、资产解析、dry-run、runtime probe 和 synthetic smoke
- `CLiD` 的计划层、资产解析、dry-run、dry-run smoke 和 artifact summary
- `recon` 纯黑盒主线的计划层、资产解析、dry-run、分阶段 smoke 和统一 mainline smoke
- `variation` API-only 黑盒线的计划层、资产解析、dry-run 和 synthetic smoke
- 最小化 vendored `SecMI` 集成子集
- 可持续扩展的测试基线

仓库目前还没有完成：

- 基于真实 checkpoint 的 `SecMI` 攻击执行
- 黑盒、白盒、灰盒三条线的系统化 benchmark 结果
- `recon` 和 `variation` 的真实资产驱动执行
- `CLiD` 的真实 text-to-image 资产驱动执行
- 完整的审计报告输出层

当前方法线的最新状态见 [docs/reproduction-status.md](docs/reproduction-status.md)。
黑盒结果的统一机读汇总见 `experiments/blackbox-status/summary.json`。

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
2. `probe-secmi-assets`
3. `prepare-secmi`
4. `dry-run-secmi`
5. 资产到位后再尝试真实执行

对 `PIA`，当前推荐命令顺序是：

1. `plan-pia`
2. `probe-pia-assets`
3. `dry-run-pia`
4. `runtime-probe-pia`
5. `run-pia-runtime-smoke`
6. `run-pia-synth-smoke`
7. 资产到位后再尝试真实执行

对 `CLiD`，当前推荐命令顺序是：

1. `plan-clid`
2. `probe-clid-assets`
3. `dry-run-clid`
4. `run-clid-dry-run-smoke`
5. `summarize-clid-artifacts`
6. 资产到位后再评估真实 text-to-image 执行链

对 `recon` 这条纯黑盒主线，当前推荐命令顺序是：

1. `plan-recon`
2. `probe-recon-assets`
3. `dry-run-recon`
4. `run-recon-mainline-smoke`
5. 需要拆阶段排查时，再分别执行 `run-recon-eval-smoke` / `summarize-recon-artifacts` / `run-recon-upstream-eval-smoke`
6. 真实资产到位后再接生成、embedding 和攻击评估三阶段执行

对 `variation` 这条 API-only 黑盒线，当前推荐命令顺序是：

1. `plan-variation`
2. `probe-variation-assets`
3. `dry-run-variation`
4. `run-variation-synth-smoke`
5. 真实 API 凭据和调用预算到位后再接真实查询执行

## 环境搭建

创建并激活 conda 环境：

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

环境说明见 [docs/environment.md](docs/environment.md)。

环境验证：

```powershell
python scripts/verify_env.py
```

如果当前 shell 没有激活 conda 环境，也可以直接用：

```powershell
conda run -n diffaudit-research python scripts/verify_env.py
```

## 快速开始

运行 smoke pipeline：

```powershell
python -m diffaudit run-smoke --config configs/benchmarks/secmi_smoke.yaml --workspace .
```

生成 `SecMI` 计划：

```powershell
python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml
```

注意：`configs/attacks/secmi_plan.yaml` 现在是共享模板，不包含你的真实本地资产路径。正式运行前，请先参考 `configs/assets/example.local.yaml`，把 `dataset_root` 和 `model_dir` 改成你自己的本地值。

探测 `SecMI` 资产是否齐全：

```powershell
python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml
```

准备 `SecMI` adapter 上下文：

```powershell
python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

运行 `SecMI` dry-run：

```powershell
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

如果缺 checkpoint、flagfile、dataset_root 等真实资产，`dry-run-secmi` 会返回 `blocked` 并直接指出缺失路径。

如果只想先看缺什么资产，不想触发完整运行时导入，优先用 `probe-secmi-assets`。

如果你还没有填好本地路径，`probe-secmi-assets` 返回 `blocked` 是预期行为。

运行 `SecMI` runtime probe：

```powershell
python -m diffaudit runtime-probe-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

这个命令不会真正执行攻击，但会校验：

- 资产路径是否存在
- vendored `SecMI` 模块是否可导入
- `flagfile` 是否能被解析

生成本地 smoke 资产：

```powershell
python -m diffaudit bootstrap-secmi-smoke-assets --target-dir tmp/secmi-smoke-assets
```

生成 `PIA` 计划：

```powershell
python -m diffaudit plan-pia --config configs/attacks/pia_plan.yaml
```

探测 `PIA` 资产是否齐全：

```powershell
python -m diffaudit probe-pia-assets --config configs/attacks/pia_plan.yaml --member-split-root external/PIA/DDPM
```

运行 `PIA` dry-run：

```powershell
python -m diffaudit dry-run-pia --config configs/attacks/pia_plan.yaml --repo-root external/PIA --member-split-root external/PIA/DDPM
```

运行 `PIA` runtime probe：

```powershell
python -m diffaudit runtime-probe-pia --config configs/attacks/pia_plan.yaml --repo-root external/PIA --member-split-root external/PIA/DDPM --device cpu
```

运行 `PIA` runtime smoke：

```powershell
python -m diffaudit run-pia-runtime-smoke --workspace experiments/pia-runtime-smoke-cpu --repo-root external/PIA --device cpu
```

运行 `PIA` synthetic smoke：

```powershell
python -m diffaudit run-pia-synth-smoke --workspace experiments/pia-synth-smoke-cpu --repo-root external/PIA --device cpu
```

生成 `CLiD` 计划：

```powershell
python -m diffaudit plan-clid --config configs/attacks/clid_plan.yaml
```

探测 `CLiD` 资产是否齐全：

```powershell
python -m diffaudit probe-clid-assets --config configs/attacks/clid_plan.yaml
```

运行 `CLiD` dry-run：

```powershell
python -m diffaudit dry-run-clid --config configs/attacks/clid_plan.yaml --repo-root external/CLiD
```

运行 `CLiD` dry-run smoke：

```powershell
python -m diffaudit run-clid-dry-run-smoke --workspace experiments/clid-dry-run-smoke --repo-root external/CLiD
```

汇总 `CLiD` 上游 `inter_output` 结果：

```powershell
python -m diffaudit summarize-clid-artifacts --artifact-dir external/CLiD/inter_output/CLID --workspace experiments/clid-artifact-summary
```

生成 `recon` 计划：

```powershell
python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml
```

探测 `recon` 资产是否齐全：

```powershell
python -m diffaudit probe-recon-assets --config configs/attacks/recon_plan.yaml
```

运行 `recon` dry-run：

```powershell
python -m diffaudit dry-run-recon --config configs/attacks/recon_plan.yaml --repo-root external/Reconstruction-based-Attack
```

运行 `recon` eval smoke：

```powershell
python -m diffaudit run-recon-eval-smoke --workspace experiments/recon-eval-smoke
```

运行 `recon` 统一主线 smoke：

```powershell
python -m diffaudit run-recon-mainline-smoke --workspace experiments/recon-mainline-smoke --repo-root external/Reconstruction-based-Attack --method threshold
```

汇总 `recon` 分数 artifact：

```powershell
python -m diffaudit summarize-recon-artifacts --artifact-dir path/to/recon-scores --workspace experiments/recon-artifact-summary
```

运行 `recon` 上游 `threshold` 评估 smoke：

```powershell
python -m diffaudit run-recon-upstream-eval-smoke --workspace experiments/recon-upstream-eval-smoke --repo-root external/Reconstruction-based-Attack --method threshold
```

生成 `variation` 计划：

```powershell
python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
```

探测 `variation` 资产是否齐全：

```powershell
python -m diffaudit probe-variation-assets --config configs/attacks/variation_plan.yaml
```

运行 `variation` dry-run：

```powershell
python -m diffaudit dry-run-variation --config configs/attacks/variation_plan.yaml
```

运行 `variation` synthetic smoke：

```powershell
python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke
```

## 参考资料

仓库已经镜像了一批调研期 PDF，位置在 `references/materials/`。

说明与索引见：

- [references/README.md](references/README.md)
- [references/materials/README.md](references/materials/README.md)

这些文件当前被纳入 git，是因为团队还处在调研期，需要共享同一套阅读上下文。

## 协作与分支管理

仓库级规范见：

- [AGENTS.md](AGENTS.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/github-collaboration.md](docs/github-collaboration.md)
- [docs/getting-started.md](docs/getting-started.md)

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
