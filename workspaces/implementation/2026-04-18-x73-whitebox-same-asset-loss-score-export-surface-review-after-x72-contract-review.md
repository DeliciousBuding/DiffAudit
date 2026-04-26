# 2026-04-18 X-73 White-Box Same-Asset Loss-Score Export Surface Review After X-72 Contract Review

## Question

What is the smallest honest export surface for a bounded same-asset white-box loss-feature lane on current admitted `DDPM/CIFAR10` assets?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x72-whitebox-same-asset-loss-feature-contract-review-after-x71-scoping.md`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA\DDPM\gen_l2_gradients_DDPM.py`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1\summary.json`

## Surface Review

### 1. Patching the upstream external GSA script is the wrong first export surface

The current admitted runtime mainline shells out to:

- `external/GSA/DDPM/gen_l2_gradients_DDPM.py`

That script does compute the relevant scalar loss internally, but today it only persists gradient tensors.

Adding loss export there would be possible, but it is not the most honest first move because it would:

- mutate the vendored / external mirror that current admitted runtime mainline still uses as its paper-aligned gradient path
- couple a new loss-score contract directly to the current admitted mainline summary path
- widen provenance and consumer drift before the export contract itself is frozen

So this is a technically possible route, but not the preferred bounded surface.

### 2. The repository already contains a better in-repo insertion point

`src/diffaudit/attacks/gsa.py` already contains one in-process extraction helper:

- `_extract_gsa_gradients_with_fixed_mask(...)`

This helper already:

- loads the admitted target checkpoint in-process
- iterates over dataset files deterministically
- computes the same denoising loss internally
- supports bounded extraction through `extraction_max_samples`
- emits in-repo artifacts rather than depending on external script side effects

So the smallest honest export surface is not “teach the upstream script a new artifact first”, but:

- add one in-repo bounded loss-score export helper / command on top of the existing internal extraction path

### 3. The export should be a separate bounded surface, not a schema mutation of current mainline

The current `run-gsa-runtime-mainline` command is the admitted gradient-centered headline surface.

Changing it in place would force:

- new artifact paths in existing summaries
- possible consumer drift on an admitted result surface
- a mixed interpretation where “mainline gradient packet” and “loss-feature scouting packet” share one summary contract before the loss-feature line is even proven useful

The cleaner first step is:

1. preserve current admitted `run-gsa-runtime-mainline` semantics
2. add one separate bounded loss-score export surface
3. keep the first loss-feature packet explicitly below admitted headline status

### 4. Minimal honest artifact contract

The preferred first export contract is:

- one separate in-repo command
- same admitted `DDPM/CIFAR10` target/shadow asset family
- bounded by explicit sample caps
- emit per-split scalar loss tensors or equivalent machine-readable score artifacts
- no immediate claim that this upgrades admitted white-box mainline

This gives the repository one honest stepping stone from:

- `contract legitimacy already frozen`

to:

- `artifact-safe execution packet available`

without overloading the current admitted gradient mainline.

## Verdict

- `x73_whitebox_same_asset_loss_score_export_surface_review_verdict = positive but bounded`

More precise reading:

1. one honest bounded export surface does exist
2. the preferred surface is an in-repo internal helper / CLI addition
3. the preferred surface is **not** patching the current upstream external extractor first
4. the preferred surface is **not** mutating current admitted `run-gsa-runtime-mainline` semantics first

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-74 white-box bounded internal loss-score export implementation after X-73 surface review`
- `current CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `Research/workspaces/white-box/plan.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- prompt/bootstrap docs: update required
- `Platform/Runtime`: no direct handoff yet
- future handoff trigger: if `X-74` changes exported fields or summary schema on admitted consumer paths, upgrade to note-level system handoff review
- competition/materials sync: note-level only
