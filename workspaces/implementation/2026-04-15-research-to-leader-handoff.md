# 2026-04-15 Research To Leader Handoff

## 结论

`Research` 线当前已经完成到可交付状态，不建议继续在本目录扩实验或继续堆包装文档。

下一步最高价值工作不在 `Research`，而在：

- `Docs`：消费研究结论，压进最终比赛材料
- `Platform`：消费研究口径，保证演示页和报告页与研究叙事一致

## 已完成的研究侧交付

### 证据层

- 黑盒：`Recon` admitted 主讲线 + `CLiD` corroboration
- 灰盒：`PIA` scale-up baseline + stochastic dropout defended comparator + `SecMI` corroboration
- 白盒：`GSA` upper bound + defended comparator

### 包装层

- human index
- machine index
- final evidence manifest
- answer pack / brief / FAQ
- slide notes / bilingual pitch / cheat sheet
- metric glossary / boundary card
- slide-to-evidence map
- rehearsal checklist
- canonical numbers sheet
- defense coverage note
- signoff (`md + json`)
- presentation asset manifest
- presentation asset checksums

## Leader 下一步该做什么

1. 以 `2026-04-15-final-delivery-index.md` 为研究入口，不要再从零翻目录。
2. 以 `2026-04-15-competition-brief.md` 和 `2026-04-15-competition-answer-pack.md` 为比赛主讲口径。
3. 以 `2026-04-15-slide-outline-and-speaker-notes.md`、`2026-04-15-bilingual-elevator-pitch-and-rapid-answers.md`、`2026-04-15-one-page-judge-cheat-sheet.md` 为答辩话术层。
4. 以 `2026-04-15-slide-to-evidence-map.md` 和 `artifacts/presentation-asset-manifest.json` 做 PPT 或答辩页整合。
5. 以 `artifacts/presentation-asset-checksums.json` 做最终封包校验。

## 不要在外部主线里改坏的研究边界

- 不要把 `CLiD` 说成 paper-faithful benchmark。
- 不要把 `PIA` / `SecMI` 说成 benchmark-ready 结论。
- 不要把 `GSA` 说成普通产品默认风险值。
- 不要把 `AUC` 直接说成真实世界固定攻击成功率。

## 如果外部主线只拿 3 个研究文件

优先顺序：

1. `workspaces/implementation/2026-04-15-competition-brief.md`
2. `workspaces/implementation/2026-04-15-competition-answer-pack.md`
3. `workspaces/implementation/reports/mainline-audit-20260415-final-refresh/summary.json`

## 如果外部主线要拿完整研究封包

入口：

- 人读：`workspaces/implementation/2026-04-15-final-delivery-index.md`
- 机读：`workspaces/implementation/artifacts/final-delivery-index.json`

## 研究线在这里结束的原因

当前研究包已经完成：

- 证据闭环
- 边界闭环
- 话术闭环
- 索引闭环
- 完整性闭环
- 校验闭环

并且自治执行入口 `docs/codex-autonomous-agent-prompt.md` 也已经同步到“默认维护一致性 / 交接，而不是继续扩实验”的状态。

继续在 `Research` 内新增内容，边际收益已经低于把这套研究包交给 `Leader / Docs / Platform` 消费的收益。
