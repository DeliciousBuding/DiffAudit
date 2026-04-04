# 黑盒方向初始计划

## 方向定位

黑盒方向关注：

- 只能输入样本并观察输出
- 不依赖完整模型参数
- 优先贴近真实闭源 API 场景

## 当前建议主论文

- 主论文：`2025-324-paper.pdf`
- 对照论文：`Towards Black-Box Membership Inference Attack for Diffusion Model.pdf`

## 第一周目标

1. 把两篇论文的攻击假设、输入输出、指标、资产要求整理成对比表
2. 跑通仓库现有 `plan-secmi / prepare-secmi / dry-run-secmi`
3. 明确黑盒方向真正需要的模型资产和数据资产
4. 输出一版黑盒复现计划

## 当前代码基础

当前仓库已经支持：

- `SecMI` 计划生成
- `SecMI` adapter 准备
- `SecMI` dry-run

当前还不支持：

- 基于真实 checkpoint 的正式攻击执行

## 当前阻塞

- 真实 `model_dir`
- 真实 `flagfile.txt`
- 真实 `dataset_root`
- 与论文设定一致的实验资产
