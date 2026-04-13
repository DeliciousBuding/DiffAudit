# 2026-04-10 TMIA-DM Intake Decomposition

## 目的

这份文档只回答一个问题：

- `TMIA-DM` 当前是否已经足够进入更深一层 `Phase E` intake review，还是仍应停留在 `protocol / asset decomposition intake`。

它不是：

- run 授权
- GPU 申请单
- 黑盒执行主线升级
- admitted upgrade

## 当前固定前提

- `PIA = gray-box mainline`
- `PIA paper-aligned confirmation = document-layer conditional rank 1, execution-layer no-go`
- `Finding NeMo = zero-GPU hold`
- `DP-LoRA = comparability / intake hardening only`
- `SecMI = blocked baseline / not-yet`

## 当前已知事实

来自现有 intake 与信号轴材料，可稳定写下的只有这些：

1. `TMIA-DM` 已归档为 `gray-box` 候选，不是严格 black-box 主线。
2. 它强调的是：
   - temporal noise / gradient behavior
   - intermediate time / noise signal
3. 它当前对仓库的价值首先是：
   - 支撑 `PIA` 所在的时间 / 噪声 / 条件信号轴
   - 作为未来 gray-box 候选
4. 当前仍没有：
   - 本地可执行 repo intake
   - 代码路径
   - 资产清单
   - 最小 smoke 入口

## 当前正式裁决

- `decision_grade = decision-grade`
- `current_verdict = not-yet`
- `current_shape = protocol-and-asset decomposition intake`
- `execution_release = none`
- `gpu_release = none`

## 为什么当前只能是 not-yet

1. 现有材料只完成了 threat-model judgment，还没有执行级 protocol 拆解。
2. 当前仍缺最小资产清单，无法判断本地到底需要什么。
3. 当前仍缺最小 smoke 入口定义，无法判断下一步应验证什么。

## Protocol Decomposition

当前应优先写清三件事：

1. `signal surface`
   - 论文到底读取什么时间 / 噪声 / 梯度信息
2. `access assumption`
   - 它比 `recon / variation` 强多少
   - 与 `PIA / SimA` 近多少
3. `local fit`
   - 如果要放进当前仓库，最像哪条现有线的扩展：
     - `PIA` 的时间/噪声信号轴扩展
     - 而不是新的黑盒执行线

## Asset Decomposition

当前至少需要补齐：

1. 目标模型面
2. 所需时间步 / 中间信号接口
3. 成员 / 非成员划分需求
4. 最小评估脚本需求

## 最小预期产物

如果未来继续推进，当前最合理的下一步产物也仍然只能是：

- `protocol note`
- `asset checklist`
- `minimal smoke entry definition`
- `go / not-yet / no-go` intake verdict

不是：

- 当前直接开跑
- 当前直接升成 gray-box 执行主线
- 当前直接并入黑盒主证据

## 当前明确不做

- 不把 `TMIA-DM` 写成 strict black-box mainline
- 不把 `TMIA-DM` 写成当前 gray-box execution release
- 不把 `TMIA-DM` 写成下一条 GPU 题
- 不把它写成对 `PIA` 的替代

## 队列影响

当前这份文档只说明：

- `TMIA-DM` 继续保留在 intake review priority #4

它不说明：

- `TMIA-DM` 已超越 `SecMI`
- `TMIA-DM` 已经值得占用 GPU
- `TMIA-DM` 已经有本地执行入口
- `TMIA-DM` 已进入 released queue item

## 下一步

在这份 intake decomposition 固定后，下一条非 GPU 槽位应回到：

- `Research/workspaces/intake/index.json` 与候选索引结构的灰盒扩展判断

而不是继续为 `TMIA-DM` 编造执行口径。
