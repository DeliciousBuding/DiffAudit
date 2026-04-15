# 2026-04-15 Slide-To-Evidence Map

## 用途

这份文件把“答辩/PPT 每一页该讲什么”直接映射到“仓库里哪份证据支撑这句话”。

适用场景：

- 做最终 PPT
- 临场被追问时快速跳到证据文件
- 检查某一页有没有说过头

## Slide 1: Threat Model Ladder

- 要讲的结论：
  - DiffAudit 审计的是扩散模型 membership risk 在 `black-box -> gray-box -> white-box` 三层权限下如何变化
- 主支撑文件：
  - `workspaces/implementation/2026-04-15-competition-brief.md`
  - `workspaces/implementation/2026-04-15-threat-model-comparison.md`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`

## Slide 2: Black-Box Admitted Headline

- 要讲的结论：
  - admitted 黑盒主讲线是 `Recon AUC 0.849`
- 主支撑文件：
  - `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
  - `workspaces/implementation/reports/mainline-audit-20260415-final-refresh/summary.json`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-unified-evidence-snapshot.md`

## Slide 3: Black-Box Is Not Single-Method Luck

- 要讲的结论：
  - `CLiD` 在同一资产家族上完成独立 corroboration
- 主支撑文件：
  - `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`
  - `workspaces/black-box/runs/clid-recon-clip-partial-target100-20260415-r1/summary.json`
- 备用支撑文件：
  - `workspaces/black-box/2026-04-15-clid-local-crosscheck-note.md`
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`
- 口径提醒：
  - 说 `workspace-verified local corroboration`
  - 不说 `full paper-faithful benchmark`

## Slide 4: Gray-Box Signal Survives Scale-Up

- 要讲的结论：
  - `PIA` 从 `512 / 512` 到 `1024 / 1024` 仍保持稳定灰盒信号
- 主支撑文件：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive/summary.json`
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-unified-evidence-snapshot.md`
  - `workspaces/gray-box/2026-04-15-pia-scale-vs-secmi-note.md`

## Slide 5: Lightweight Defense Helps, But Does Not Solve

- 要讲的结论：
  - stochastic dropout 只温和削弱灰盒攻击，不能把风险压回随机
- 主支撑文件：
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive/summary.json`
  - `workspaces/implementation/2026-04-15-attack-defense-matrix.md`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`
  - `workspaces/implementation/artifacts/unified-attack-defense-table.json`

## Slide 6: Gray-Box Corroboration Beyond PIA

- 要讲的结论：
  - `SecMI` 证明灰盒信号不依赖单一 scorer
- 主支撑文件：
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
  - `workspaces/gray-box/2026-04-15-pia-scale-vs-secmi-note.md`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`

## Slide 7: White-Box Upper Bound

- 要讲的结论：
  - `GSA 0.998192` 给出 privileged access 下的风险上界
- 主支撑文件：
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
  - `workspaces/implementation/reports/mainline-audit-20260415-final-refresh/summary.json`
- 备用支撑文件：
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`
- 口径提醒：
  - 说 `privileged upper bound`
  - 不说 `normal product KPI`

## Slide 8: Final Takeaway

- 要讲的结论：
  - membership leakage 跨权限层级持续存在，方法变化不会抹掉信号，轻量防御还不够
- 主支撑文件：
  - `workspaces/implementation/2026-04-15-competition-brief.md`
  - `workspaces/implementation/2026-04-15-competition-answer-pack.md`
  - `workspaces/implementation/artifacts/final-evidence-manifest.json`
- 备用支撑文件：
  - `workspaces/implementation/2026-04-15-one-page-judge-cheat-sheet.md`
  - `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`

## 如果评委追问术语或边界

- 指标解释：
  - `workspaces/implementation/2026-04-15-metric-glossary-and-claim-boundary-card.md`
- 快问快答：
  - `workspaces/implementation/2026-04-15-judge-faq-short.md`
- 超短口播：
  - `workspaces/implementation/2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`

## 最后核查规则

1. 每一页至少对应一份主支撑文件
2. 每个核心数字必须能在 `summary.json` 或统一表里回查
3. 涉及 `CLiD`、`GSA` 的页必须保留边界提醒
4. 若时间被压缩，优先保留 Slide 2、4、5、7、8
