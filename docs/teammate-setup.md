# Contributor Setup

这份文档面向刚拿到 `Research` 仓库的新成员或外部贡献者。

目标不是解释算法细节，而是帮助新机器尽快进入“可运行命令、可绑定资产、可继续复现”的状态。

如果只看一份数据/权重说明，先看 [data-and-assets-handoff.md](data-and-assets-handoff.md)。它记录了如何拿到和当前项目一致的 `Download\` 资产布局。

如果只想找命令清单，看 [command-reference.md](command-reference.md)。根目录 `README.md` 只保留入口和导航。

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

上面这套是默认 setup 入口，也是 CI 使用的入口。

如果你用的是较新的 NVIDIA GPU，并且在默认环境里已经遇到 `no kernel image is available for execution on the device`，再把上面的 `environment.yml` 换成 `environment.gpu-cu128.yml` 重试；大多数贡献者应先使用默认共享入口。

如果你想直接复用下面文档里的 PowerShell 示例，也可以先在当前 shell 里设置：

```powershell
$env:DIFFAUDIT_ROOT = Split-Path -Parent (Resolve-Path .)
```

## 2. 本地资产模板

默认目录布局是：

```text
<DIFFAUDIT_ROOT>\
  Research\        # this git repo
  Download\        # raw datasets / weights / supplementary bundles
```

`<DIFFAUDIT_ROOT>` 是本地项目根目录，比如 `C:\Users\<you>\DiffAudit`、`D:\Projects\DiffAudit` 或 Linux/macOS 上的任意工作目录。`Download\` 不属于 git 仓库。新机器要么从项目资产镜像复制整个 `Download\`，要么按 [research-download-master-list.md](research-download-master-list.md) 的 first-wave 顺序下载。不要把这些大文件放回 `Research\external\`。

不要直接改共享配置里的占位符。

先复制：

```powershell
Copy-Item configs/assets/team.local.template.yaml configs/assets/team.local.yaml
```

然后只在 `team.local.yaml` 里填写当前机器上的路径。

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

## 3. Minimal Track Entry Points

### 黑盒

```powershell
conda run -n diffaudit-research python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml
conda run -n diffaudit-research python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
conda run -n diffaudit-research python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke-local
conda run -n diffaudit-research python -m diffaudit audit-recon-public-bundle --bundle-root "$env:DIFFAUDIT_ROOT\Download\black-box\supplementary\recon-assets\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models"
python scripts/init_variation_query_set.py --root /absolute/path/to/variation-query-set
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

## 4. Current Priority Sources

刚开始接手时，不要从历史日志反推优先级。当前真实优先级以这些文件为准：

1. [comprehensive-progress.md](comprehensive-progress.md)
2. [reproduction-status.md](reproduction-status.md)
3. [../ROADMAP.md](../ROADMAP.md)
4. 对应 lane 的 `workspaces/<lane>/plan.md`

如果只是刚开始接手，不要一开始就扩展新数据集或新论文。先复现环境、自检资产、确认负责的 lane，再做新分支。

## 5. 当前最重要的入口文档

- 综合进度：[comprehensive-progress.md](comprehensive-progress.md)
- 命令参考：[command-reference.md](command-reference.md)
- 严格三线计划：[mentor-strict-reproduction-plan.md](mentor-strict-reproduction-plan.md)
- 黑盒计划：[../workspaces/black-box/plan.md](../workspaces/black-box/plan.md)
- 灰盒计划：[../workspaces/gray-box/plan.md](../workspaces/gray-box/plan.md)
- 白盒计划：[../workspaces/white-box/plan.md](../workspaces/white-box/plan.md)

## 6. Common Setup Risks

1. 环境创建后没有重新执行 `python scripts/bootstrap_research_env.py --install`，导致 `python -m diffaudit` 不可用。
2. 把个人机器真实路径写进共享配置。
3. 把 `smoke / preview / toy` 写成复现成功。
4. 黑盒入口里误用 `SecMI`，它属于灰盒 baseline。
5. `TMIA-DM` 已归档但属于灰盒候选，不要把它写成黑盒主线。
6. `variation` 的真实恢复首先缺 query image set，不是先缺 API 代码。
7. `<DIFFAUDIT_ROOT>\Download\manifests\research-download-manifest.json` 是本机 manifest；真正随仓库走的是 [research-download-master-list.md](research-download-master-list.md) 和 [data-and-assets-handoff.md](data-and-assets-handoff.md)。
