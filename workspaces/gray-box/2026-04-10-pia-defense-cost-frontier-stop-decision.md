# 2026-04-10 PIA Defense-Cost Frontier Stop Decision

## Status Panel

- `owner`: `research_leader + reviewer_audit_leader + gpu_scheduler_leader`
- `track`: `gray-box`
- `scope`: `G1-B / PIA defense-cost frontier`
- `decision_grade`: `review-ready`
- `verdict`: `no-go`
- `gpu_release`: `none`
- `queue_state`: `not-requestable`
- `active_gpu_question`: `none`
- `admission_effect`: `none`
- `updated_at`: `2026-04-10 22:13 +08:00`

## Question

当前 `PIA defense-cost frontier` 是否还存在一个值得继续占用 GPU 的 bounded low-cost question。

## Inputs

- `workspaces/gray-box/2026-04-10-pia-8gb-supporting-frontier-note.md`
- `workspaces/gray-box/2026-04-10-pia-8gb-portability-ladder-execution-packet.md`
- `workspaces/gray-box/2026-04-09-pia-gpu512-adaptive-ablation.md`
- `workspaces/gray-box/2026-04-09-pia-signal-and-cost.md`
- `workspaces/gray-box/2026-04-09-pia-provenance-dossier.md`
- `docs/autonomous-research-director/state.json`
- `docs/autonomous-research-director/rounds/2026-04-10-2151-round-26.md`

## Evidence

### Round-26 frontier 已回答 portability 问题

- `GPU128`:
  - `adaptive AUC = 0.817444 -> 0.806274`
  - `adaptive ASR = 0.765625 -> 0.761719`
  - `wall-clock = 49.544877s -> 50.667101s`
- `GPU256`:
  - `adaptive AUC = 0.841293 -> 0.829559`
  - `adaptive ASR = 0.78125 -> 0.763672`
  - `wall-clock = 95.934721s -> 213.126361s`

当前 frontier 已固定为：

1. `GPU128 = quickest portable pair`
2. `GPU256 = decision rung with cost warning`
3. `stop before any GPU512 rerun`

### 当前没有新的低成本问题被写出来

当前不存在一份新的单独 hypothesis，能够同时回答：

1. 现有 `GPU128/GPU256` 为什么还不足以回答问题
2. 为什么新问题必须继续用 GPU
3. 为什么这不是 `GPU128/GPU256/GPU512` 已回答问题的重复
4. 失败时会留下什么新的 frontier artifact

### 既有 schedule 问题已经被答过

`GPU512 adaptive ablation` 已经给出：

- `all_steps` 更强
- `late_steps_only` 更保质量，但 privacy drop 更弱

因此当前没有一个“只差再跑一轮就能决定”的明显 schedule 分支。

### provenance 才是 strongest claim 的真正硬门槛

当前 strongest claim 仍然只能是：

- `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`

继续扩 GPU 不会改变这个 blocker。

## Decision

当前正式推荐：

- `G1-B / PIA defense-cost frontier = no-go`

具体含义：

1. 不为 `PIA defense-cost frontier` 继续保留 GPU 槽位
2. 不重开 `GPU512`
3. 不机械复跑 `GPU128/GPU256`
4. 不把 round-26 的 supporting frontier 偷换成 admitted update

当前统一治理口径应写成：

- `queue_state = not-requestable (frontier frozen at GPU128/GPU256; no new bounded low-cost GPU question)`

## CPU Pivot

当前最合理的 CPU pivot 不是再找一条勉强能跑的 GPU 小题，而是回到 `PIA provenance`。

当前优先顺序固定为：

1. `source / release identity reconciliation`
2. `external/PIA upstream identity note`
3. `split semantics mapping`

其中第一优先的理由是：

- 本地 checkpoint 与本地 zip 的字节级关系已经成立
- strongest claim 仍卡在“本地 retained source bundle 与 README/上游入口之间的可审查关系”

## Reopen Conditions

只有同时满足下面条件，才允许重审 `G1-B`：

1. 写出新的单一 hypothesis
2. 写出明确 `protocol delta`
3. 写出 `compute budget`
4. 写出 `expected artifact`
5. 证明这一步不是 `GPU128/GPU256/GPU512` 已回答问题的重复

在这些条件满足前：

- `G1-B` 不重开
- `active_gpu_question` 继续保持 `none`

## Forbidden Wording

当前禁止写：

- `paper-aligned confirmation`
- `new admitted gray-box mainline`
- `validated privacy win`
- `canonical defense refresh`
- `GPU256 still drops metrics so GPU512 should be reopened`

## Next Step

1. 把本 decision 写回 gray-box plan 与 long-horizon plan
2. 继续补 `PIA provenance` 的 CPU 侧审查链
3. 在不改 admitted 主结果的前提下，恢复 `recon explanation / boundary package` 的 writing-only 推进
