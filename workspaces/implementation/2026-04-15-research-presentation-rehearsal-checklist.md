# 2026-04-15 Research Presentation Rehearsal Checklist

## 用途

这份清单用于研究线答辩前最后一次 dry run。

目标不是再补材料，而是确认：

- 数字没有讲错
- 边界没有讲过头
- 时间被压缩时知道先保留什么
- 被追问时知道跳到哪份证据

## A. 上台前 60 秒核对

- 先记住三条数字主线：
  - `Recon 0.849`
  - `PIA 1024 / 1024 0.83863 -> 0.825966`
  - `GSA 0.998192`
- 再记住两条边界：
  - `CLiD` 说 local corroboration，不说 full paper-faithful benchmark
  - `GSA` 说 privileged upper bound，不说普通产品 KPI
- 再记住一句总收尾：
  - 扩散模型 membership leakage 跨权限层级持续存在，而当前轻量防御还不足以消除它

## B. 3 分钟版本顺序

1. Threat-model ladder
2. Black-box `Recon 0.849`
3. Gray-box `PIA 0.83863`
4. Defense `0.83863 -> 0.825966`
5. White-box `GSA 0.998192`
6. Final takeaway

如果只剩 2 分钟：

1. `Recon 0.849`
2. `PIA 0.83863 -> 0.825966`
3. `GSA 0.998192`
4. Final takeaway

## C. 绝对不要说错的地方

- 不要把 `AUC` 说成真实世界固定攻击成功率
- 不要把 `admitted` 说成绝对最高分
- 不要把 `CLiD` 说成完整 paper-faithful benchmark
- 不要把 `GSA` 说成普通用户场景默认风险

## D. 评委追问时的跳转顺序

### 如果追问“黑盒靠不靠谱”

- 先答：`Recon` 是 admitted 主讲线
- 再补：`CLiD` 在同资产家族上独立 corroboration
- 跳转文件：
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`
  - `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`

### 如果追问“灰盒是不是小样本幻觉”

- 先答：`512 / 512` 到 `1024 / 1024` 仍稳定
- 再补：`SecMI` 用不同 scorer 也成立
- 跳转文件：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`

### 如果追问“防御到底有没有用”

- 先答：有用，但只温和削弱
- 再给数字：`0.83863 -> 0.825966`
- 跳转文件：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
  - `workspaces/implementation/2026-04-15-attack-defense-matrix.md`

### 如果追问“白盒是不是不现实”

- 先答：是上界，不是默认场景
- 再补：它的作用是说明权限提升后风险会迅速放大
- 跳转文件：
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
  - `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`

## E. 最后一次数字检查

- `Recon admitted`: `0.849`
- `Recon best single-metric`: `0.866`
- `PIA 1024 baseline`: `0.83863`
- `PIA 1024 defended`: `0.825966`
- `SecMI stat`: `0.885833`
- `SecMI NNS`: `0.946286`
- `GSA`: `0.998192`

## F. 最后一次材料检查

- 长讲稿：
  - `workspaces/implementation/2026-04-15-slide-outline-and-speaker-notes.md`
- 超短中英口播：
  - `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`
- 一页速查：
  - `workspaces/implementation/2026-04-15-one-page-judge-cheat-sheet.md`
- 术语和边界：
  - `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`
- 页到证据映射：
  - `workspaces/implementation/2026-04-15-slide-to-evidence-map.md`

## G. 过线标准

- 3 分钟讲完不超时
- 7 个核心数字能一次说对
- 4 条边界提醒没有遗漏
- 每个高频追问都知道跳到哪份文件
