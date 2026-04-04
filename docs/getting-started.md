# 新成员上手指南

这份文档面向第一次进入 DiffAudit 仓库的组员，目标是让你在最短时间内知道：

- 我们在做什么
- 你该在哪个方向工作
- 你本地先跑什么命令
- 你当前缺什么资产

## 一、先理解项目目标

DiffAudit 不是“让模型生成更好图片”的项目。

它的核心目标是：

- 研究扩散模型是否记住了训练样本
- 复现黑盒、白盒、灰盒成员推断方法
- 把攻击结果整理成可审计、可解释的证据

## 二、先确定你属于哪个工作区

- 做黑盒方向：看 `workspaces/black-box/`
- 做白盒方向：看 `workspaces/white-box/`
- 做灰盒方向：看 `workspaces/gray-box/`
- 做共享工程：看 `workspaces/implementation/`

## 三、本地第一步命令

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
$env:PYTHONPATH='src;.'
python scripts/verify_env.py
python -m unittest
```

## 四、如果你接黑盒方向

优先跑：

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml
python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

如果输出 `blocked`，说明不是代码没接好，而是你还缺实验资产。

## 五、当前最常见阻塞项

- 缺 `flagfile.txt`
- 缺 checkpoint
- 缺真实 `dataset_root`
- 缺论文对应的实验设定

## 六、你应该先交什么

不要一上来就说“我要复现整篇论文”。

建议先交：

- 一页论文总结
- 一份复现计划
- 一份资产清单
- 当前阻塞项
