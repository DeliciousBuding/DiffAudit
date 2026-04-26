# Teammate Setup

这份文档面向刚拿到 `Research` 仓库的队友。

目标不是解释算法，而是保证新机器能尽快进入“可跑命令、可补路径、可继续三线复现”的状态。

## 1. 拉仓后的第一步

在 `Research` 根目录执行：

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

上面这套是默认接仓入口，也是 CI 使用的入口。

如果你用的是较新的 NVIDIA GPU，并且在默认环境里已经遇到 `no kernel image is available for execution on the device`，再把上面的 `environment.yml` 换成 `environment.gpu-cu128.yml` 重试；普通接仓不要默认改共享入口。

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

填完后，推荐直接生成各线 local config：

```powershell
python scripts/render_team_local_configs.py
```

默认输出到：

- `tmp/configs/rendered/secmi.local.yaml`
- `tmp/configs/rendered/pia.local.yaml`
- `tmp/configs/rendered/recon.local.yaml`
- `tmp/configs/rendered/variation.local.yaml`
- `tmp/configs/rendered/clid.local.yaml`

## 3. 三条线最小接仓入口

### 黑盒

```powershell
conda run -n diffaudit-research python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml
conda run -n diffaudit-research python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
conda run -n diffaudit-research python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke-local
conda run -n diffaudit-research python -m diffaudit audit-recon-public-bundle --bundle-root D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models
python scripts/init_variation_query_set.py --root D:/path/to/variation-query-set
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
5. `TMIA-DM` 已归档但属于灰盒候选，不要把它写成黑盒主线。
6. `variation` 的真实恢复首先缺 query image set，不是先缺 API 代码。



