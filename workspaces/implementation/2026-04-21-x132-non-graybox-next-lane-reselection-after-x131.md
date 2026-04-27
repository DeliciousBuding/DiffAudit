# 2026-04-21 X-132 Non-Graybox Next-Lane Reselection After X-131

## Question

After `X-131` cleared the remaining active stale-entry layer, what is the highest-value next live lane: another abstract non-graybox expansion, or a return to the current `04-defense` active slot through the first missing `H2` contract stage?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x119-04-h2-canonical-contract-hardening.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/lora_ddpm.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/train_smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_lora_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_smp_lora_runtime_tuning.py`

## Findings

### 1. The control plane is now clean enough to return to the active `04` slot

After `X-131`:

- no active stale-entry layer remains
- no fresh `I-A` residue remains
- gray-box has already yielded honestly

So another abstract reselection loop would now be lower value.

### 2. `04-H2` is the nearest bounded blocker with real leverage

`H2 privacy-aware adapter` is already:

- `prototype-implemented`
- test-backed
- smoke-backed

But it still lacks the canonical `diffaudit` contract frozen in `X-119`.

Among the four required stages, the first missing step is the smallest honest executable blocker:

- `probe-h2-assets`

Landing that probe would:

- convert `H2` from vague prototype truth into machine-readable contract truth
- unblock the later `prepare / run / review` chain
- improve the current `04-defense` active slot without opening GPU

## Verdict

`positive`.

Sharper control truth:

1. after `X-131`, the next highest-value lane is no longer another empty non-graybox reselection pass
2. the next honest live lane becomes:
   - `X-133 04-H2 probe-h2-assets executable contract start`
3. `active_gpu_question = none`
4. `next_gpu_candidate = none`
5. the CPU sidecar remains non-graybox sync / reselection only if a fresh stale surface reappears

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x132-non-graybox-next-lane-reselection-after-x131.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side lane-ordering decision only. It selects the next contract-building task but does not yet change any runtime or platform contract.
