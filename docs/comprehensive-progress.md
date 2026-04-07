# Comprehensive Progress

这份文档是 `Project` 研究仓库的综合进度入口。

它不替代 [reproduction-status.md](reproduction-status.md) 的逐线细节，也不替代 [mia-defense-research-index.md](mia-defense-research-index.md) 的文献整理；它的职责是把“当前最能讲的攻击线、最缺的防御线、最短执行路径”放到一页里。

## 当前一句话

当前仓库已经具备三条攻击线的基本骨架，但真正最成熟、最适合打成“攻击 + 防御”主讲闭环的，是灰盒 `PIA`。黑盒 `recon` 负责提供最强风险证据，白盒 `GSA` 负责提供深度与上界，防御线整体仍明显落后于攻击线。

## 进度总览

| 维度 | 当前判断 | 备注 |
| --- | --- | --- |
| 黑盒攻击 | `较成熟` | `recon` 是当前最强证据线 |
| 灰盒攻击 | `最成熟` | `PIA` 已进入 real-asset runtime mainline |
| 白盒攻击 | `闭环已通` | `GSA` 已到 real-asset closed loop ready，但结果还不稳 |
| 黑盒防御 | `基本未落地` | `B-1 / B-2` 仍在设计层 |
| 灰盒防御 | `有原型` | 接近 `G-1`，但还没形成正式对照 |
| 白盒防御 | `候选已就位` | `DPDM` 已在本地，尚未形成 `W-1` run |
| 统一评估表 | `缺失` | 当前最需要补的一块 |

## 攻击主线

### 黑盒

- 主线：`recon`
- 次主线候选：`variation`（对应 `Towards Black-Box`）
- 当前能说的话：
  - 公开资产上的 black-box 风险已经有可引用主证据
  - `variation` 已能在本地 CPU 上重复跑 synthetic smoke
- 当前不能说的话：
  - 还不能把 black-box 防御讲成已有结果
  - 还不能把 `variation` 写成真实 API 闭环
- 当前用途：
  - 作为申报和答辩里的“风险存在”主证据
  - `variation` 适合作为第二黑盒候选线补充进申报叙事

### 灰盒

- 主线：`PIA`
- baseline：`SecMI`
- 当前能说的话：
  - `PIA` 已经不是 smoke，而是真实资产 mainline
  - 当前最适合把防御压到这条线上做正式比较
- 当前不能说的话：
  - 还不能说灰盒防御已经验证有效
- 当前用途：
  - 作为当前算法主讲线

### 白盒

- 主线：`GSA`
- 扩展：`Finding NeMo`
- 当前能说的话：
  - 白盒闭环已经打通
  - 资产根、checkpoint-*、bucket 已进入规范结构
- 当前不能说的话：
  - 还不能说白盒论文级复现成功
- 当前用途：
  - 作为技术深度补充线

## 防御主线

### 当前建议

| 轨道 | 当前最合理防御路线 | 当前判断 |
| --- | --- | --- |
| 黑盒 | `B-1 / B-2` | 设计方向成立，但还没有正式实现 |
| 灰盒 | `G-1` | 已有近似原型，最值得先打穿 |
| 白盒 | `W-1 = DPDM` | 已有仓库基础，适合先落 baseline |

### 当前不建议优先做

- `G-2` 知识蒸馏代理模型
- `W-2` 成员信号对抗训练

原因：

- 它们设计空间太大
- 当前仓库还没有稳定的 attack-defense 对比表
- 申报阶段更需要可运行、可对比、可讲清楚的路线

## 当前最重要的偏差

### 1. 文档路线不等于仓库真实状态

- `mia-defense-document.docx` 可以指导防御方向
- 但不能直接当作当前执行进度表

### 2. 黑盒优先不等于黑盒是当前最适合主讲的攻击-防御闭环

- 黑盒 `recon` 证据最强
- 但灰盒 `PIA` 更适合打成“攻击 + 防御”主讲闭环

### 3. 白盒价值在深度，不在当前申报阶段的稳定结果

- `GSA` 很重要
- 但当前它更适合作为“我们已经打通白盒闭环”的证明，而不是唯一主讲成果

## 当前最短执行顺序

1. `PIA baseline + defended`
2. `SecMI` promote / block 判定
3. `recon` 主证据口径收口
4. `GSA` 扩 bucket / 提强度
5. `DPDM` 接成 `W-1`
6. 统一 attack-defense 总表

## 申报 / PPT 应该怎么讲

当前最合理的讲法是：

1. 扩散模型存在成员泄露风险
2. 我们已经在黑盒、灰盒、白盒三种权限下建立了攻击验证能力
3. 当前最成熟的是灰盒 `PIA`
4. 我们正在把一个可运行的灰盒防御打成第一版正式闭环
5. 白盒 `GSA + W-1` 是下一步补强方向

## 关联文档

- 逐线状态：[reproduction-status.md](reproduction-status.md)
- 防御文档索引：[mia-defense-research-index.md](mia-defense-research-index.md)
- 防御执行清单：[mia-defense-execution-checklist.md](mia-defense-execution-checklist.md)
- 研究仓路线图：[../ROADMAP.md](../ROADMAP.md)
