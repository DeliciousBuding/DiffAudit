# Admitted Results Summary (Admitted Side Only)

> 该表严格覆盖 admitted 方向的证据，不含 candidate / exploratory 资产；SMP-LoRA 在当前比赛主讲表之外，仍被定位为 comparator-design 目标，不进入本表。每列都直接取自最新的 `unified-attack-defense-table.json` 与 `competition-evidence-pack.md` 的 mainline 口径，可被复制进 Leader 的 `project-report.md` / `project-summary.md`。

| Track | Method | Attack | Defense | AUC | ASR | Evidence Location | Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Black-box | Recon risk proof | `recon DDIM public-100 step30` | `none` | 0.849 | 0.51 | `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` （black-box row） | 证据只在 `fine-tuned / controlled / public-subset / proxy-shadow-member` 语义下成立，用于证明最弱权限下存在成员泄露风险，定位为 “current black-box main evidence” 。 |
| Gray-box | PIA baseline | `PIA GPU512 baseline` | `none` | 0.841339 | 0.786133 | `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` （gray-box baseline row） | Workspace-verified + adaptive-reviewed 主结果；必须同时携带 `paper-aligned blocked by checkpoint/source provenance`，用于支撑 attack-defense 闭环的 baseline 口径。 |
| Gray-box | PIA defended | `PIA GPU512 baseline` | `provisional G-1 = stochastic-dropout (all_steps)` | 0.828075 | 0.767578 | `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` （gray-box defended row） | 同样是 workspace-verified + adaptive-reviewed，并强调 defended line 在 repeated-query review 后保持下降，边界语义为 “PIA defended mainline” 。 |
| White-box | GSA attack | `GSA 1k-3shadow` | `none` | 0.998192 | 0.9895 | `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` （white-box attack row） | Admitted white-box line，攻击部分用来标明“white-box admitted mainline”，但不能写成 final paper-level benchmark；以 risk upper-bound 的语义出现。 |
| White-box | W-1 defended | `GSA 1k-3shadow` | `W-1 strong-v3 full-scale` | 0.488783 | 0.4985 | `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` （white-box defended row） | Defended comparator 保持为 admitted deep line，提示 “white-box bridge closed-frozen / not a finished benchmark”，对比仍在治理决策层面。 |

每条记录都仅占 admitted 结果主值，并可被 Leader 直接引用；若需要说明开发消费字段，则继续参考 `unified-attack-defense-table.json` 中 `quality_cost`、`provenance_status`、`boundary` 等字段，保证 Local-API 与 Platform 只读 admitted 数据。
