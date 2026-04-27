# 2026-04-21 X-135 04-H2 Run-H2-Defense-Pilot Contract Landing

## Question

After `X-133` and `X-134` land the first two canonical `H2` stages, can `04-H2` execute one honest bounded training pilot on current admitted assets without overstating it as a defense verdict?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x134-04-h2-prepare-contract-landing.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/h2_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_h2_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/train_smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-run-defense-pilot-20260421-r1/summary.json`

## Findings

### 1. Canonical run surface now exists

`src/diffaudit/defenses/h2_adapter.py` now exposes `run_h2_defense_pilot(...)`, and `src/diffaudit/cli.py` now exposes `run-h2-defense-pilot`.

This stage consumes a frozen `prepare-h2-contract` manifest, stages a bounded member/nonmember packet, executes the existing `train_smp_lora.py`, and emits a workspace-root `summary.json` with artifact paths and first-step metrics.

### 2. First real admitted-asset pilot is landed

The first real bounded run at `workspaces/implementation/runs/h2-run-defense-pilot-20260421-r1/summary.json` is `ready` on:

- baseline checkpoint = `checkpoint-9600/model.safetensors`
- full packet identity = `1000 / 1000`, `32 x 32 x 3`, `RGB`
- executed packet = `1 member / 1 nonmember`
- runtime = `rank 4 / lambda 0.5 / epochs 10 / batch_size 8 / device cpu / seed 7`
- artifacts = `config.json`, `checkpoint_meta.json`, `lora_summary.json`, `training_log.json`, `lora_weights.pt`, `proxy_weights.pt`, `stdout`, `stderr`

### 3. What this still does not prove

This is not yet an attack-side defense verdict.

The run only proves that the bounded `H2` training stage can execute under the canonical contract on current admitted assets.

It still lacks:

- same-packet baseline-vs-defended attack-side comparison
- mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` review board
- explicit `transfer-only` vs `defense-aware` attacker reading

## Verdict

`positive but bounded`

`H2` now reads:

- `probe landed`
- `prepare landed`
- `run landed`
- `review missing`

It still remains below:

- defense-positive wording
- `next_gpu_candidate`
- any claim about low-FPR improvement

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-run-defense-pilot-20260421-r1/summary.json`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-135 04-H2 run-h2-defense-pilot bounded execution contract start`
- `next live lane = X-136 04-H2 review-h2-defense-pilot same-packet attack-side review contract start`
- `CPU sidecar = X-136 04-H2 review-h2-defense-pilot same-packet attack-side review contract start`

## Handoff

- `Research/ROADMAP.md`: yes
- `Research/README.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/reproduction-status.md`: yes
- `docs/mainline-narrative.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `Platform/Runtime`: no

Reason:

This changes only research-side `H2` execution truth. It still does not require any Runtime endpoint, runner protocol, or Platform schema change.
