# 黑盒方向计划

## 主论文

- `2025-324-paper.pdf`
- `Towards Black-Box Membership Inference Attack for Diffusion Model.pdf`
- `CLiD_NeurIPS2024.pdf`

## 推荐分工

- 主复现：`Black-box Membership Inference Attacks against Fine-tuned Diffusion Models`
- 对照参考：`Towards Black-Box Membership Inference Attack for Diffusion Models`
- 场景补充：`CLiD`

## 一周行动清单

1. 整理三篇论文的问题定义、攻击假设、输入输出和指标
2. 对比三篇论文需要的资产：checkpoint、flagfile、dataset_root、prompt 或 text 条件
3. 跑通仓库现有黑盒基础命令：
   - `plan-secmi`
   - `prepare-secmi`
   - `dry-run-secmi`
4. 形成一份黑盒方向复现计划
5. 明确当前缺失的真实资产

## 当前阻塞项

- 缺真实 checkpoint 和训练期 flagfile
- 缺与论文一致的实验数据布局
- 黑盒不同论文的攻击假设并不完全相同，需要统一术语
