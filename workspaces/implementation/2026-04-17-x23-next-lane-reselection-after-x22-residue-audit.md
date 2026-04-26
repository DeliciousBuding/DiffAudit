# 2026-04-17 X-23 Non-Graybox Next-Lane Reselection After X-22 Residue Audit

## Question

After `X-22` closed the remaining `I-A` presentation residue, what should now become the next honest non-graybox `CPU-first` lane?

## Inputs Reviewed

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\docs\leader-research-ready-summary.md`
- `D:\Code\DiffAudit\Research\docs\competition-evidence-pack.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x22-ia-higher-layer-truth-hardening-residue-audit.md`

## Candidate Comparison

### 1. Reopen blocked non-graybox challengers

Not selected.

Reason:

1. `XB-CH-2` is still `needs-assets`;
2. `GB-CH-2` remains `hold` with no new gating signal;
3. black-box and white-box candidate queues still do not expose a stronger executable lane.

### 2. Stay on `I-A` again immediately

Not selected.

Reason:

1. `X-22` already reduced the currently visible `I-A` residue;
2. another immediate pass would risk wording-only churn without new evidence;
3. the next honest move is to clean the remaining stale execution-order entry points before deciding the next substantive lane.

### 3. Execute one residual stale-entry cleanup pass

Selected.

Reason:

1. `mainline-narrative.md` still carried an old `X-20`-anchored execution-order section;
2. the root `ROADMAP.md` still pointed its `Now | 24h` window and `P1` item at `X-22`;
3. this remaining stale layer is small, bounded, CPU-only, and directly affects how higher layers read current repo truth.

## Selection

- `selected_next_live_lane = X-24 residual stale-entry cleanup after X-23 reselection`

## Verdict

- `x23_next_lane_reselection_verdict = positive`

More precise reading:

1. no blocked challenger became executable during `X-22`;
2. the strongest immediate non-graybox move is one bounded residual stale-entry cleanup pass;
3. `next_gpu_candidate` remains `none`.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `current_execution_lane = X-24 residual stale-entry cleanup after X-23 reselection`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance maintenance`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x23-next-lane-reselection-after-x22-residue-audit.md`

## Handoff Decision

- `D:\Code\DiffAudit\ROADMAP.md`: update required
- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/mainline-narrative.md`: update required
- `root_sync = required`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this step changes current control-plane ordering and stale-entry cleanup priority, but it does not change admitted metric tables or runtime contracts.
