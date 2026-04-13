# 2026-04-11 TMIA-DM Protocol & Asset Decomposition

## 目的

这份文档旨在补齐 `TMIA-DM` 的 protocol decomposition 和 asset decomposition，为未来的 intake review 做准备。

## 基本信息

- `candidate_key`: `gray-box/tmia-dm/cifar10-ddpm`
- `track`: `gray-box`
- `method`: `tmia-dm`
- `current_status`: `protocol-and-asset decomposition intake`
- `gpu_release`: `none`
- `queue_state`: `not-requestable`

## 当前定位

`TMIA-DM` 当前只能写成：

- `gray-box temporal noise/gradient signal candidate`
- `PIA time/noise signal axis extension`

不能写成：

- `strict black-box mainline`
- `gray-box execution release`
- `next GPU question`

## Signal Surface

### 论文核心方法

`TMIA-DM` (Temporal Membership Inference Attack on Diffusion Models) 是一种利用时间维度信息的成员推断攻击：

- **核心思想**：利用扩散模型在生成过程中不同时间步的噪声状态来推断成员信息
- **信号类型**：
  - 中间时间步的噪声状态
  - 梯度信息（如果可访问）
  - 时间相关的噪声模式

### 攻击面

- **access assumption**: 需要访问模型的中间时间步状态
- **与 recon 的区别**：recon 是基于重建误差，TMIA-DM 是基于时间/噪声模式
- **与 PIA 的关系**：PIA 基于 epsilon-trajectory consistency，TMIA-DM 基于时间/噪声信号

### Access Assumption 分析

| 攻击方法 | 访问权限需求 | 强度 |
| --- | --- | --- |
| `recon` (black-box) | 只能查询模型输出 | 较弱 |
| `variation` (black-box) | API 查询 | 较弱 |
| `TMIA-DM` (gray-box) | 中间时间步状态/梯度 | 中等 |
| `PIA` (gray-box) | 完整轨迹 | 较强 |
| `GSA` (white-box) | 模型参数 | 最强 |

## Local Fit

### 与现有线的关系

如果要把 TMIA-DM 放进当前仓库，最合理的定位是：

- `PIA 时间/噪声信号轴的扩展候选`
- `gray-box 信号分析的多角度补充`

### 具体扩展方向

1. **时间维度扩展**
   - PIA 分析的是 epsilon-trajectory consistency
   - TMIA-DM 分析的是时间步特定的噪声模式

2. **信号类型扩展**
   - PIA: epsilon 轨迹一致性
   - TMIA-DM: 噪声状态 + 梯度信息

3. **评估协议扩展**
   - 使用相同的成员划分
   - 但使用不同的信号提取方法

## Asset Decomposition

### 必需资产

| 资产类型 | 必需 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| 目标模型 | 是 | 待确认 | 复用当前 DDPM 或需特定模型 |
| 时间步采样接口 | 是 | 待实现 | 需要获取中间时间步状态 |
| 梯度计算接口 | 是 | 待实现 | 如果需要梯度信息 |
| 成员划分 | 是 | 已有 | 复用 PIA 的 split 文件 |
| 评估脚本 | 是 | 待实现 | 提取时间/噪声特征并评估 |

### 最小可执行路径

```
1. 加载预训练 DDPM 模型
2. 获取中间时间步的噪声状态
3. 提取时间/噪声特征
4. 使用成员划分进行评估
5. 输出 AUC, ASR 等指标
```

### 最小 Smoke 定义

- **输入**: 100 张成员图像 + 100 张非成员图像
- **输出**: 成员推断 AUC
- **验证**: AUC > 0.5（随机）说明有信号

## Expected Artifact

如果未来进入 intake review，预期产物包括：

1. **Protocol Note**
   - 信号类型定义
   - 访问权限需求
   - 与 PIA 的差异分析

2. **Asset Checklist**
   - 目标模型需求
   - 接口需求
   - 评估脚本需求

3. **Minimal Smoke Result**
   - 最小可执行 run 的结果
   - 证明方法可行

## Stop Conditions

在以下情况下应直接判 `not-yet / no-go`：

1. **接口不可用**
   - 无法获取中间时间步状态
   - 无法实现梯度计算

2. **信号不明显**
   - 最小 smoke 结果接近随机
   - 无法区分成员和非成员

3. **与 PIA 无增量**
   - TMIA-DM 信号与 PIA 完全重叠
   - 没有提供新的信息

4. **实现复杂度高**
   - 需要大量改造才能适配当前仓库
   - 成本收益不合理

## 结论

`TMIA-DM` 当前定位为 `PIA 时间/噪声信号轴的扩展候选`，需要：

1. 明确时间步采样接口需求
2. 确认梯度信息是否必需
3. 验证最小 smoke 是否可行

才能继续推进 intake review。