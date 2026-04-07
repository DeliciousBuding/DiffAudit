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

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

## 验证环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, pandas, matplotlib, diffusers, transformers"
python scripts/verify_env.py
python -m diffaudit --help
```

## 后续扩展原则

只有在某条算法方向真正确定后，才继续增加该论文专用依赖。调研期尽量保持环境稳定，不要为了单篇论文把环境改得过重。
