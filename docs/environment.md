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

## 当前已验证的 GPU 栈

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

## 验证环境

```powershell
$env:PYTHONPATH='src;.'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, pandas, matplotlib, diffusers, transformers"
python scripts/verify_env.py
```

## 后续扩展原则

只有在某条算法方向真正确定后，才继续增加该论文专用依赖。调研期尽量保持环境稳定，不要为了单篇论文把环境改得过重。
