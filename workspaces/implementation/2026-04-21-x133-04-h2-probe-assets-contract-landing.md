# 2026-04-21 X-133 04-H2 Probe-H2-Assets Contract Landing

## Question

Can `04-H2 privacy-aware adapter` land the first canonical `diffaudit` contract stage on current admitted assets, or is it still blocked before any executable surface exists?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x119-04-h2-canonical-contract-hardening.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/h2_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_h2_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/checkpoints/target`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-member`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-nonmember`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-probe-assets-20260421-r1/summary.json`

## Findings

### 1. Canonical probe surface now exists

`src/diffaudit/defenses/h2_adapter.py` now exposes a real `probe_h2_assets(...)` helper, and `src/diffaudit/cli.py` now exposes `probe-h2-assets`.

This is no longer a research-only prototype surface: `H2` now has a dedicated `diffaudit` contract entry for checkpoint identity, member/nonmember roots, image-layout compatibility, and packet boundedness.

### 2. Current admitted assets pass the first contract stage

The first real admitted-asset probe at `workspaces/implementation/runs/h2-probe-assets-20260421-r1/summary.json` resolves:

- checkpoint source = `checkpoint-root-latest`
- resolved checkpoint = `checkpoint-9600/model.safetensors`
- member root = `1000` images
- nonmember root = `1000` images
- layout = full-scan `32 x 32 x 3`, `RGB`, consistent on both sides
- packet cap = `1000`
- effective packet size = `1000`

So the first missing canonical stage is no longer hypothetical; it is landed and machine-readable.

### 3. What is still missing

This does **not** make `H2` execution-ready.

It still lacks:

- frozen workspace manifest
- bounded training execution summary
- attack-side baseline-vs-defended review
- mandatory low-FPR comparison contract

## Verdict

`positive but bounded`

`H2` has now crossed from `prototype-only` into `probe-landed`, but it still remains below:

- `next_gpu_candidate`
- `run-ready`
- `review-ready`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-probe-assets-20260421-r1/summary.json`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-133 04-H2 probe-h2-assets executable contract start`
- `next live lane = X-134 04-H2 prepare-h2-contract minimal surface freeze`
- `CPU sidecar = X-134 04-H2 prepare-h2-contract minimal surface freeze`

## Handoff

- `Research/ROADMAP.md`: yes
- `Research/README.md`: yes
- `docs/comprehensive-progress.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `Platform/Runtime`: no

Reason:

This changes only research-side execution truth. It still does not justify any Runtime or Platform schema change.
