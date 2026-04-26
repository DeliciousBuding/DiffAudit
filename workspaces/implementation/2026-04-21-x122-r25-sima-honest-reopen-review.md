# 2026-04-21 X-122 R2-5 SimA Honest Reopen Review

## Question

Now that `R2-5 02-H1 SimA` has been promoted into the current support lane, does current repo truth justify reopening plain `SimA` scorer execution as the next honest task?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\sima_adapter.py`
- `D:\Code\DiffAudit\Research\tests\test_sima_adapter.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-sima-feasibility-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-sima-rescan-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-next-family-reselection.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\sima-cifar10-runtime-feasibility-20260416-cpu-32-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\sima-cifar10-runtime-rescan-20260416-cpu-32-r2\summary.json`
- `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results\02.md`

## Verification

- `python -m unittest D:\Code\DiffAudit\Research\tests\test_sima_adapter.py`
- result: `Ran 2 tests ... OK`
- note: `pytest` is not installed in the current shell, so the local validation used stdlib `unittest`

## Findings

### 1. SimA is not paper-only

The repo already has a real bounded `SimA` surface:

- `src/diffaudit/attacks/sima_adapter.py`
- `tests/test_sima_adapter.py`
- bounded runtime feasibility and later-timestep rescan artifacts

So `SimA` is already `execution-feasible` on the current `DDPM/CIFAR10` asset line.

### 2. But the current plain scorer path is still weak

Current bounded evidence remains below honest challenger quality:

- first bounded run:
  - `best_timestep = 120`
  - `AUC = 0.542969`
  - `TPR@1%FPR = 0.0625`
- later-timestep rescan:
  - `best_timestep = 160`
  - `AUC = 0.584961`
  - `TPR@1%FPR = 0.03125`

This is still far below current `PIA` strength and below promotion / GPU-release quality.

### 3. Therefore `R2-5` cannot honestly mean plain scorer reopen

The GPT-5.4 round-2 memo is still useful, but only under a sharper reading:

- `SimA` is a plausible second-signal family
- `PIA + SimA` can still be worth contract review
- but the current repo does **not** justify another plain `SimA` scorer rerun without a fresh bounded hypothesis

## Verdict

`negative but clarifying`.

Current honest control truth:

1. keep `SimA` as `execution-feasible but weak`
2. do **not** reopen plain `SimA` scorer execution right now
3. allow only two honest next uses:
   - `PIA + SimA` support-fusion / calibration contract review
   - a genuinely new bounded paper-faithful `SimA` hypothesis

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-122 R2-5 SimA honest reopen review`
- `next CPU-first lane = R2-5b PIA + SimA support-fusion contract review`

## Handoff

- `Research/ROADMAP.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no handoff

Reason:

This changes only research-side control truth and support-lane wording. It does not change admitted tables, Runtime endpoints, or Platform schema.
