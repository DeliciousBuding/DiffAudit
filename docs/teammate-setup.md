# Teammate Setup

这份文档面向刚拿到 `Project` 仓库的队友。

目标不是解释算法，而是保证新机器能尽快进入“可跑命令、可补路径、可继续三线复现”的状态。

## 1. 拉仓后的第一步

在 `Project` 根目录执行：

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
python scripts/verify_env.py
python -m diffaudit --help
```

如果环境已存在，改用：

```powershell
conda env update -f environment.yml --prune
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
```

## 2. 本地资产模板

不要直接改共享配置里的占位符。

先复制：

```powershell
Copy-Item configs/assets/team.local.template.yaml configs/assets/team.local.yaml
```

然后只在 `team.local.yaml` 里填写你自己机器上的路径。

注意：

- `configs/assets/team.local.yaml` 会被 `.gitignore` 忽略
- `configs/assets/example.local.yaml` 和 `team.local.template.yaml` 只是模板

## 3. 三条线最小接仓入口

### 黑盒

```powershell
conda run -n diffaudit-research python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml
conda run -n diffaudit-research python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
conda run -n diffaudit-research python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke-local
```

### 灰盒

```powershell
conda run -n diffaudit-research python -m diffaudit plan-pia --config configs/attacks/pia_plan.yaml
conda run -n diffaudit-research python -m diffaudit probe-pia-assets --config configs/attacks/pia_plan.yaml --member-split-root external/PIA/DDPM
```

### 白盒

```powershell
conda run -n diffaudit-research python -m diffaudit probe-gsa-assets --repo-root workspaces/white-box/external/GSA --assets-root workspaces/white-box/assets/gsa
```

## 4. 当前主线优先级

当前仓库按这个顺序推进：

1. `PIA`
2. `recon`
3. `variation/Towards`
4. `GSA`
5. `W-1 / DPDM`

如果只是刚接仓，不要一开始就扩展新数据集或新论文。

## 5. 当前最重要的入口文档

- 综合进度：[comprehensive-progress.md](comprehensive-progress.md)
- 严格三线计划：[mentor-strict-reproduction-plan.md](mentor-strict-reproduction-plan.md)
- 黑盒计划：[../workspaces/black-box/plan.md](../workspaces/black-box/plan.md)
- 灰盒计划：[../workspaces/gray-box/plan.md](../workspaces/gray-box/plan.md)
- 白盒计划：[../workspaces/white-box/plan.md](../workspaces/white-box/plan.md)

## 6. 当前常见坑

1. 环境创建后没有重新执行 `python scripts/bootstrap_research_env.py --install`，导致 `python -m diffaudit` 不可用。
2. 把个人机器真实路径写进共享配置。
3. 把 `smoke / preview / toy` 写成复现成功。
4. 黑盒入口里误用 `SecMI`，它属于灰盒 baseline。
