# 科研环境说明

本项目使用独立 conda 环境 `diffaudit-research`。

## 环境目标

当前环境服务于调研与原型阶段，覆盖：

- Python 数值分析：`numpy`、`pandas`、`matplotlib`
- PyTorch 基础：`Tensor`、`Autograd`、训练与推理
- 扩散模型研究准备：`diffusers`、`transformers`、`accelerate`
- notebook、脚本与 smoke 实验

目前它还没有绑定到某一篇论文的完整执行资产。

## 为什么用 Python 3.11

当前 PyTorch 与扩散模型生态在 `Python 3.11` 下更稳，因此这里没有使用系统里的 3.12。

## 包管理策略

本项目采用：

- `conda`：负责隔离 Python 运行时
- `pip`：负责大部分科研包和深度学习包

这样做的原因是当前机器曾出现过 conda 科学计算包缓存损坏问题，使用混合方案更稳。

## 创建环境

默认情况下，请先使用团队共享的 `environment.yml`：

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

`environment.yml` 会把当前仓库以 editable 方式安装进环境，所以环境创建完成后可以直接运行：

```powershell
python -m diffaudit --help
diffaudit --help
```

为了避免环境更新后 editable install 丢失，推荐紧接着执行：

```powershell
python scripts/bootstrap_research_env.py --install
```

如果是队友第一次接仓，环境通过后继续执行：

```powershell
python scripts/render_team_local_configs.py
```

## 可选：较新 NVIDIA GPU 的 cu128 环境

`environment.yml` 仍然是 CI 和新同学接仓的默认入口，不要把它当作“只要能点亮本机 GPU 就直接全员替换”的文件。

如果你用的是较新的 NVIDIA GPU，并且在默认环境里已经出现下面这种真实 CUDA 算子错误：

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

可以改用仓库里的可选环境文件：

```powershell
conda env create -f environment.gpu-cu128.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

如果你已经创建过默认环境，想直接切到这套栈，改用：

```powershell
conda env update -f environment.gpu-cu128.yml --prune
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
```

这套可选环境当前用于较新的 GPU 兼容性补充，已在 `RTX 5070 Laptop GPU` 上通过真实 CUDA `matmul` smoke。

## 如果当前 shell 没有激活 conda

有些终端线程不会自动继承你本地已经激活过的环境。这时不要直接用系统 Python 跑仓库命令，优先显式前缀：

```powershell
conda run -n diffaudit-research python scripts/verify_env.py
conda run -n diffaudit-research python -m unittest
```

如果当前 shell 没有激活 conda，也可以直接显式指定环境来运行命令：

```powershell
conda run -n diffaudit-research python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml
```

建议先用 `conda env list` 确认 `diffaudit-research` 确实存在，再执行后续命令。

## 队友接仓推荐步骤

如果是新机器或新同学刚接仓，优先按 [teammate-setup.md](teammate-setup.md) 执行，不要直接手改共享配置。

## 本地资产路径如何填写

`configs/attacks/secmi_plan.yaml` 是共享模板，不应该提交个人机器上的真实路径。

推荐做法：

1. 先查看 `configs/assets/team.local.template.yaml`
2. 在本地复制成 `configs/assets/team.local.yaml`
3. 只在 `team.local.yaml` 里填写你自己的真实路径
4. 共享配置继续保持占位符或相对路径
5. 先用各线的 `probe-*` 命令确认资产 readiness，再跑后续命令

## 当前已验证的 GPU 栈

默认入口 / CI：

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

可选较新 GPU 栈：

- `torch==2.11.0+cu128`
- `torchvision==0.26.0+cu128`
- `torchaudio==2.11.0+cu128`

## 验证环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, pandas, matplotlib, diffusers, transformers"
python scripts/verify_env.py
python -m diffaudit --help
```

## 后续扩展原则

只有在某条算法方向真正确定后，才继续增加该论文专用依赖。调研期尽量保持环境稳定，不要为了单篇论文把环境改得过重。
