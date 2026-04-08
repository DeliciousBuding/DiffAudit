# Mentor Strict Reproduction Plan

这份文档把师兄口头方案收口成当前 `Project` 仓库的严格执行面。

依据来源：

- [mia-defense-document.docx](../references/materials/context/mia-defense-document.docx)
- [和师兄聊天录音.txt](../../Archive/reference-materials/和师兄聊天录音.txt)
- 当前仓库真实状态与实验产物

## 核心原则

1. 黑盒、灰盒、白盒都继续复现，不停任何一条线。
2. 但三条线不平均分资源，优先级仍然分层。
3. 当前申报阶段先要有：
   - 可讲清楚的攻击复现
   - 至少一个能把指标打下来的防御改动
4. 前端不是当前第一优先级。

## 三线执行排序

### 第一优先级：灰盒 `PIA`

原因：

- 当前最成熟
- 最适合打成“攻击 + 防御”主讲闭环
- 师兄明确点名 `PIA` 论文必须自己读

当前目标：

1. 稳定 `PIA baseline`
2. 正式定义 `G-1`
3. 产出 `PIA baseline vs defended`

### 第二优先级：黑盒 `recon + variation`

黑盒内部再分层：

- `recon`
  - 当前最强证据线
- `variation / Towards`
  - 当前第二黑盒候选线
  - 已有本地 CPU synthetic smoke
  - 更贴近师兄口头里的“黑盒简单一点、先做一个能讲的”

当前目标：

1. `recon` 固化主证据口径
2. `variation` 保持 formal local black-box secondary track
3. 如果真实 API 资产到位，优先把 `variation` 从 synthetic 推向真实 black-box
4. 不把 `TMIA-DM` 误收进黑盒主线；它当前应明确归为灰盒候选论文

### 第三优先级：白盒 `GSA + W-1`

原因：

- 需要继续复现
- 但当前结果还不够稳，不适合抢主讲资源

当前目标：

1. `GSA` 扩到 `1k-3shadow` 并按更接近上游的口径重跑
2. `DPDM` 从 smoke checkpoint 推进到第一版 `W-1` baseline

## 当前按线执行清单

### 黑盒

- `recon`
  - 收口 `public-100 step10 / step30`
  - 继续收口语义映射
- `variation`
  - 保留本地 synthetic smoke
  - 补真实 API 资产模板
  - 真实 query image 集一旦到位，先跑 `probe-variation-assets`

### 灰盒

- `PIA`
  - 读懂论文
  - 稳定 baseline
  - 正式化 `G-1`
  - 输出前后对比
- `SecMI`
  - 继续探真实资产
  - 不再长期悬空
- `TMIA-DM`
  - 保持为已归档候选论文
  - 当前只做 threat-model 与方法对照
  - 在代码路径明确前不写成执行主线

### 白盒

- `GSA`
  - 从 `128` 级 bucket 升到 `1000 + 3 shadow`
  - 提高训练强度
  - 改用更接近上游的梯度提取与攻击评估配置
- `W-1`
  - 基于现有 smoke checkpoint 接入白盒攻击评估
  - 再逐步恢复更强训练设置

## 统一交付要求

每条线最终都要能回写到统一表中：

- `track`
- `attack`
- `defense`
- `dataset`
- `model`
- `AUC`
- `ASR`
- `TPR@low-FPR`
- `quality/cost`
- `evidence_level`

## 当前最短路径

1. `PIA baseline + defended`
2. `variation` 真实资产模板
3. `recon` 主证据口径收口
4. `GSA 1k-3shadow`
5. `DPDM -> W-1`
6. 统一 attack-defense 总表
