# 2026-04-21 X-138 04-H2 Bounded Packet-Scale Follow-Up Selection

## Question

After `X-137` confirms that `04-H2` is minimally contract-complete but still only has a transfer-only `1 / 1` null board, should this lane yield immediately, or does it deserve one minimal bounded packet-scale follow-up first?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x137-non-graybox-next-lane-reselection-after-x136.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-run-defense-pilot-20260421-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-review-defense-pilot-20260421-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/h2_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`

## Findings

### 1. `H2` is no longer blocked on contract construction

The canonical `probe / prepare / run / review` chain is already landed on admitted CIFAR10 assets. The next question is no longer whether `H2` can run, but whether the first `1 / 1` null board is just undersized noise or already enough to freeze the lane.

### 2. Any honest packet-scale follow-up must rerun the bounded pilot itself

`review-h2-defense-pilot` consumes the `executed_packet` recorded in the run summary. So packet enlargement cannot be answered by replaying the old review alone; it must stage a new bounded `run-h2-defense-pilot` packet and then review that exact packet.

### 3. `4 / 4` is the smallest honest enlargement

Immediate lane-yield would be too aggressive because `1 / 1` is too degenerate to tell whether target-transfer is structurally null or only packet-starved.

But jumping straight to `8 / 8` or `16 / 16` would spend extra CPU before we even know whether any non-null transfer behavior appears.

So the minimal honest next rung is:

- same assets
- same contract
- same transfer-only evaluator
- same CPU budget class
- one bounded `4 / 4` run + review pair

## Verdict

`positive`

`04-H2` deserves exactly one minimal packet-scale enlargement before yield.

The highest-value next live lane becomes:

- `X-139 04-H2 minimal 4/4 bounded packet-scale follow-up review`

with:

- `active GPU question = none`
- `next_gpu_candidate = none`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x138-04-h2-bounded-packet-scale-followup-selection.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side control decision only. No consumer contract changes.
