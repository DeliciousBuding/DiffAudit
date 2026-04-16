# DiffAudit Research ROADMAP — Continuous Autonomous Mainline

> Last updated: 2026-04-16 07:30
> Mode: continuous autonomous research
> Owner: `Researcher`
> Rule: one active GPU task at a time, every task must end in a concrete verdict

## 0. Positioning

This roadmap is not a one-shot sprint checklist.

It is the operating backlog for a long-running `ResearcherAgent` that should keep pushing:

- model mainline
- innovation funnel
- system-consumable evidence
- GPU utilization quality
- long-horizon research depth

`2026-04-19` remains a real 4C deadline, but it is **not** a stop condition and it is **not** a reason to freeze research.

The correct reading is:

- before `2026-04-19`, the agent should bias toward work that can also improve 4C narrative, material quality, and system-consumable structure;
- after `2026-04-19`, the agent continues without reset and expands toward National Innovation work;
- if this roadmap is exhausted, the agent must expand it instead of declaring idle completion.

---

## 1. Operating Contract

### 1.1 Root alignment

This roadmap must stay aligned with root-level `D:\Code\DiffAudit\ROADMAP.md`, but it is not limited to "4C sprint mode".

For `Research`, root alignment means:

- respect near-term delivery pressure;
- keep generating research outcomes that can change project-level story;
- continue building the post-4C mainline instead of pausing;
- keep `Research -> Runtime -> Platform` consumption path healthy.

### 1.2 What the agent is optimizing

The `ResearcherAgent` should optimize for all of the following together:

1. stronger method diversity
2. stronger defense diversity
3. better research truth-to-system read path
4. higher GPU utilization quality
5. lower wasted GPU hours
6. less operator babysitting
7. more innovation density per week

### 1.3 Hard constraints

- only one active GPU task at a time
- every task must have a bounded hypothesis
- every task must have a bounded budget
- every task must end as one of:
  - `positive`
  - `negative`
  - `no-go`
  - `blocked`
  - `needs-assets`
- no repeated sweep without a new reason
- no "looks promising" without recorded evidence
- if a task stalls, skip it, document it, and move to the next highest-value task

### 1.4 Broad exploration policy

Research is explicitly allowed to try multiple directions in parallel over time.

Not only:

- black-box mainline
- gray-box mainline
- white-box mainline

Also:

- cross-box analysis
- fusion and calibration
- feature-space or caption-space signals
- transfer or portability probes
- mitigation-aware evaluation
- quality-cost analysis
- system-consumable schema and comparison work
- any new paper-backed or original idea that can be tested with bounded cost

### 1.5 Canonical startup contract

Fresh sessions should always read in this order:

1. `D:\Code\DiffAudit\ROADMAP.md`
2. `D:\Code\DiffAudit\Research\ROADMAP.md`
3. `D:\Code\DiffAudit\Research\AGENTS.md`
4. `D:\Code\DiffAudit\Research\docs\researcher-agent-architecture.md`
5. `D:\Code\DiffAudit\Research\README.md`
6. `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
7. `D:\Code\DiffAudit\Download\manifests\research-download-manifest.json`
8. lane-specific workspace docs only after the task is narrowed

---

## 2. Self-Cycling Loop

Every long-running session should execute this loop:

1. `Sense`
   - read root roadmap
   - read this roadmap
   - inspect latest runs, blockers, challenger queue, comparison artifacts
2. `Select`
   - choose the highest-value task, not merely the first unchecked box
3. `Delegate` (optional)
   - open subagents only if they create real leverage
4. `Execute`
   - probe -> smoke -> mainline, or blocker diagnosis, or CPU-side integration
5. `Review`
   - critique the result, critique the direction, critique the next step
6. `Sync`
   - update `summary.json`, roadmap, notes, and any system-consumable artifact
7. `Expand`
   - if the current roadmap is insufficient, add new tasks and continue

This loop never stops at "the listed tasks are done for now" if new high-value work is visible.

---

## 3. Optional Subagent Policy

The `ResearcherAgent` is allowed to open subagents, but should not do so mechanically every turn.

Use subagents when they create real parallel leverage:

### 3.1 Good subagent use cases

- `paper scout`
  - survey a paper family, extract the shortest feasible implementation path
- `code reviewer`
  - inspect current implementation for hidden bugs, weak assumptions, or drift from the method hypothesis
- `experiment auditor`
  - review a finished run and challenge whether the verdict is honest
- `platform handoff reviewer`
  - identify what fields, boundaries, or summaries should be exposed to Platform or Runtime
- `backlog critic`
  - challenge whether current roadmap priorities are still correct

### 3.2 Subagent defaults

- prefer background execution, not blocking the main loop
- prefer `gpt-5.4` with `high` reasoning effort
- wait less often, but when waiting is necessary, wait longer instead of busy-polling
- use subagents for focused, bounded questions
- do not spawn subagents for work the main agent can do faster directly

### 3.3 Subagent output contract

Default assumption:

- subagents are read-only unless a write scope is explicitly assigned

Every subagent should return:

- exact question answered
- evidence or files inspected
- verdict
- exact files needing changes, if any
- next action recommendation

Truth integration rule:

- subagent output is advisory until the main agent reviews it
- only the main agent can promote output into roadmap truth, artifact truth, or mainline wording

### 3.4 Required review cadence

At meaningful checkpoints, the main agent should trigger at least one self-review action:

- before starting a costly GPU run
- after a surprising result
- after a blocker diagnosis
- before promoting a challenger into a more official line
- when the roadmap begins to feel stale

---

## 4. Current Strategic Picture

### 4.1 Black-box

Current truth:

- `recon` is the strongest main evidence line
- `CLiD` provides strong local corroboration but still has boundary-quality work left
- a second truly different black-box family is still not firmly landed

Current need:

- new signal family
- mitigation-aware evaluation
- better boundary quality

### 4.2 Gray-box

Current truth:

- `PIA` is the strongest mainline
- `stochastic-dropout(all_steps)` is the current defended story
- `SecMI` exists, but diversity is not yet sufficient

Current need:

- second defense mechanism
- ranking-sensitive variable
- another truly different family or a stronger disagreement story

### 4.3 White-box

Current truth:

- `GSA` is a very strong white-box line
- `W-1 = DPDM` gives a meaningful defended comparator
- second white-box line is still unresolved

Current need:

- blocker resolution or decisive blocker classification
- second-line verdict
- stronger defense comparison breadth

### 4.4 Cross-box and system-consumable layer

Current truth:

- comparison table exists
- challenger queue exists
- evidence chain is already more structured than before

Current need:

- keep all new outcomes consumable by higher layers
- keep narrative and machine-readable structure aligned
- keep expanding cross-box reasoning, not just per-box silos

---

## 5. Task Selection Heuristic

When multiple unchecked tasks are available, choose in this order:

1. the task with the strongest blocker leverage
2. the task most likely to change a project-level story soon
3. the task most likely to create a new attack or defense verdict
4. the task most likely to improve system-consumable structure
5. the task most likely to open a new high-value branch
6. only then same-family optimization

If two tasks are similar, prefer:

- lower setup cost
- lower GPU cost
- higher interpretability
- higher reuse by future work
- lower risk of locking the machine for too long

---

## 6. Active Backlog

### 6.1 Cross-box integration and system sync

#### ✅ `X-1` Unified cross-box comparison table

- status: completed
- keep maintained, do not treat as frozen forever

#### ✅ `X-2` Live challenger queue

- status: completed
- must be updated when new verdicts or new ideas appear

#### ✅ `X-3` Keep system-consumable structure in sync

Goal:

- keep admitted/mainline/challenger boundaries readable
- keep comparison artifacts current
- keep higher-layer consumers able to interpret new outcomes

Tasks:

- [x] `X-3.1` audit whether latest verdicts are reflected in comparison artifacts
- [x] `X-3.2` audit whether latest verdicts change project narrative
- [x] `X-3.3` mark any Platform/Runtime-facing field additions or boundary changes

Status:

- completed for the current gray-box mainline and challenger state
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-16-graybox-system-sync-verdict.md`

Verdict:

- system-consumable gray-box artifacts now reflect the current truth:
  - `PIA` remains the admitted headline
  - `TMIA-DM late-window` is the strongest active challenger
  - defended gray-box remains multi-family rather than collapsing back to `PIA` only

Value: ⭐⭐⭐
Budget: CPU-only

Artifact rule for `X-*`, `INF-*`, `intake`, or other non-run tasks:

- they do not need to force a fake `runs/` directory
- but they must leave one canonical evidence anchor under the appropriate workspace lane
- the main agent must explicitly name that anchor in its update

#### ✅ `X-4` Cross-box exploration lane

Goal: allow work that spans boxes instead of forcing every idea into one bucket

Candidate directions:

- [ ] `X-4.1` cross-box agreement analysis
- [ ] `X-4.2` score calibration or fusion with bounded hypothesis
- [ ] `X-4.3` transfer or portability probe when assets permit
- [x] `X-4.4` visualization or decision-quality analysis that changes project understanding

Status:

- completed for the current cross-box handoff-sync round
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-16-crossbox-handoff-review.md`
  - `workspaces/implementation/2026-04-16-crossbox-agreement-analysis.md`

Verdict:

- admitted headline remains unchanged across all three boxes
- but the system-consumable challenger and boundary layer changed enough to require immediate summary-level sync:
  - gray-box strongest defended challenger is now `TMIA + temporal-striding(stride=2)`
  - black-box `CLiD` wording should be fixed to `evaluator-near local clip-only corroboration`
  - black-box `variation` should be fixed to `contract-ready blocked`
  - black-box `served-image-sanitization` is an explicit `no-go`
  - white-box should now be described as having no second executable defended family beyond `W-1 = DPDM`

Handoff decision:

- sync `Leader` and competition-material wording now
- no mandatory `Platform` or `Runtime` schema change is required in this round

Latest follow-up:

- `X-4.1` cross-box agreement analysis also completes positively:
  - all three boxes now agree that leakage risk is real, but they support different project roles
  - `black-box = existence proof`, `gray-box = main attack-defense story`, `white-box = depth / upper bound`
  - no new GPU admission question is justified by this analysis alone
- refreshed `X-4.1` also completes positively after the `DP-LoRA` lane-status downgrade:
  - the same layered project narrative still holds
  - and the new white-box truth actually makes it cleaner, because `DP-LoRA / SMP-LoRA` now sits explicitly below top-layer white-box wording as a `metric-split bounded exploration branch`
  - `gpu_release = none` remains unchanged
- `X-4.2` score calibration/fusion review now closes as `negative but useful`:
  - one cross-box scalar would still be dishonest because access level, metric semantics, and narrative role differ materially across boxes
  - current honest structure remains layered role-based summary, not fused ranking
  - `gpu_release = none`
- `X-4.3` transfer/portability review now closes as `negative but useful`:
  - no honest portability probe is ready to start
  - the branch remains `needs-assets` because paired dataset/model contracts are still missing
  - `gpu_release = none`

Carry-forward rule:

- keep `X-4` open for future agreement / calibration / portability work
- do not reopen this exact handoff audit unless a new verdict changes higher-layer wording again

Value: ⭐⭐⭐
Budget: bounded CPU-first, GPU only if justified

#### ✅ `X-5` White-box summary-layer resync

Goal:

- sync the repository-level summary layer after the `DP-LoRA` lane was reduced to a bounded metric-split exploration branch

Tasks:

- [x] `X-5.1` audit whether `docs/comprehensive-progress.md` still reflects old `SMP-LoRA` wording
- [x] `X-5.2` rewrite white-box and summary-layer wording around current `DP-LoRA` truth
- [x] `X-5.3` mark whether higher-layer material sync is suggested

Status:

- current summary-layer resync completed
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-16-whitebox-summary-layer-resync.md`

Verdict:

- the summary layer was stale
- it still described `SMP-LoRA` as if the next live question were `T06 optimizer/lr frontier`
- it now reflects the current bounded truth:
  - `DP-LoRA / SMP-LoRA` remains alive
  - but only as a `metric-split bounded exploration branch`
  - with `gpu_release = none`
- `competition_material_sync = suggested`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `X-6` Phase E sparse-registry refresh

Goal:

- refresh the machine-readable `Phase E` candidate registry after recent lane-status closures made multiple candidate records stale

Tasks:

- [x] `X-6.1` review whether current `phase-e-candidates.json` still matches lane truth after `WB-18`, `GB-18`, and `BB-7`
- [x] `X-6.2` remove records that are no longer intake-only candidate surface items
- [x] `X-6.3` record whether higher-layer intake docs now need follow-up sync

Status:

- current sparse-registry refresh completed
- canonical evidence anchor:
  - `workspaces/intake/2026-04-16-phase-e-sparse-registry-refresh-verdict.md`

Verdict:

- `phase-e-candidates.json` was stale again
- `DP-LoRA` no longer belongs on the intake-only candidate surface because it has already consumed a full executed white-box exploration lane and now sits at `bounded exploration branch + no-new-gpu-question`
- `SecMI unblock` no longer matches repository truth because gray-box now treats `SecMI` as an independent corroboration line rather than an asset-blocked baseline reopen
- current honest machine-readable posture is therefore a `sparse-hold` registry:
  - `Finding NeMo = the only remaining intake-only candidate`
  - `PIA paper-aligned confirmation = document-layer conditional only`
- `docs/future-phase-e-intake.md` and `docs/reproduction-status.md` now need follow-up sync

Value: ⭐⭐⭐
Budget: CPU-only

---

### 6.2 Black-box expansion

#### ✅ `BB-1` Second-signal black-box expansion

Goal: land a second black-box direction that is not just a mild variant of current image-similarity lines

Tasks:

- [x] `BB-1.1` caption-space probe
- [x] `BB-1.2` feature-space probe
- [x] `BB-1.3` timestep-selective reconstruction probe
- [x] `BB-1.4` prompt-response consistency probe

Status:

- completed for the current leading challenger branch
- canonical evidence anchor:
  - `workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
  - `workspaces/black-box/2026-04-16-blackbox-recon-timestep-probe-verdict.md`
  - `workspaces/black-box/runs/recon-timestep-probe-20260416-r1/summary.json`
  - `workspaces/black-box/2026-04-16-blackbox-prompt-response-consistency-verdict.md`
  - `workspaces/black-box/runs/prompt-response-consistency-20260416-r1/summary.json`

Verdict:

- the returned-image `semantic-auxiliary-classifier` landed as a real black-box challenger
- bounded local comparator metrics (`AUC = 0.910156`, `ASR = 0.875`) are strong enough for promotion into challenger status
- scaled follow-up run `semantic-aux-classifier-comparator-20260416-r2` stayed stable at `AUC = 0.90918`, confirming the signal survives a `32 / 32` comparator
- bounded `recon` coordinate-selective probe now also closes negatively on current artifacts:
  - current score vectors have length `768`
  - but the honest `shadow-select -> target-eval` best coordinate underperformed the existing `dim0` baseline on target (`AUC gain = -0.0425`)
  - current `recon` headline therefore does not gain a timestep/coordinate-selective upgrade on the present artifact stack
- bounded prompt-response consistency probe now closes negatively on the same local outputs:
  - `AUC = 0.481445`
  - `AUC gain vs mean_cos = -0.435547`
  - this confirms the current challenger is not being driven by prompt-to-return alignment
- this branch adds real method-family diversity, but it does not replace the frozen `Recon` headline

Carry-forward rule:

- keep this line in challenger status
- keep `BB-1.3` closed unless artifact semantics or selection logic changes materially
- keep `BB-1.4` closed unless the conditioning protocol changes materially
- only escalate with a new bounded hypothesis; do not spend budget on aimless scale-up

Promotion standard:

- must produce a verdict
- if positive, must be stronger than "interesting but noisy"

Value: ⭐⭐⭐

#### ✅ `BB-2` Scoring and calibration upgrades

Goal: improve black-box signal quality without pretending every scoring tweak is a new method family

Current read:

- inside the current `semantic-auxiliary-classifier` challenger, the multi-feature logistic score is not beating the simplest single score:
  - on `16 / 16`, logistic `AUC = 0.910156` while `mean_cos = 0.945312`
  - on `32 / 32`, logistic `AUC = 0.90625` while `mean_cos = 0.916992`
- rank agreement is also very high:
  - `Spearman(logistic, mean_cos) = 0.973607 / 0.978709`
- the current black-box scoring result is therefore `negative but useful`:
  - the present calibration stack mostly smooths or rescales the same ordering
  - it does not yet justify a new scoring-based challenger promotion

Tasks:

- [x] `BB-2.1` MSE-weighted or multi-score challenger
- [x] `BB-2.2` bounded fusion experiments
- [x] `BB-2.3` document whether calibration changes ranking or only threshold

Canonical evidence anchor:

- `workspaces/black-box/2026-04-16-blackbox-semantic-aux-scoring-verdict.md`
- `workspaces/black-box/runs/semantic-aux-fusion-20260416-r1/summary.json`
- `workspaces/black-box/2026-04-16-blackbox-semantic-aux-fusion-verdict.md`

Updated verdict:

- bounded fusion also closes as `negative but useful`
- on `16 / 16`, no tested fusion candidate beat `mean_cos`
- on `32 / 32`, the best bounded fusion (`cosine_pair_zmean`) only improved `AUC` from `0.916992` to `0.918945`
- that `+0.001953` gain is below the promotion bar and remains almost rank-identical to `mean_cos`
- current semantic-aux scoring truth is therefore:
  - `mean_cos` remains the preferred simple score reference
  - current logistic and bounded fusion variants are refinements of the same ordering, not a stronger new score family

Value: ⭐⭐

#### ✅ `BB-3` CLiD boundary-quality upgrade

Goal: improve the honesty and strength of the CLiD boundary claim

Current read:

- staged-path bridge preparation already exists, so the current blocker is no longer generic hygiene
- the executed target-side local CLiD rung now has an explicitly checked evaluator-near artifact shape:
  - both output files parse into numeric `100 x 5` matrices after skipping the first header line
- but the executed rung still records a user-cache SD1.5 `diff_path` in that header, while the staged-path-prepared run has not yet been executed as a new benchmark rung
- the released `cal_clid_th.py` still expects a shadow train/test pair in addition to the target pair
- current honest boundary therefore tightens to `evaluator-near local clip-only corroboration`, but remains `shadow-blocked` rather than `paper-aligned local benchmark`

Tasks:

- [x] `BB-3.1` review remaining boundary gaps
- [x] `BB-3.2` perform one minimum honest upgrade
- [x] `BB-3.3` update verdict wording and evidence note

Canonical evidence anchor:

- `workspaces/black-box/2026-04-15-clid-paper-alignment-audit.md`
- `workspaces/black-box/2026-04-16-clid-threshold-evaluator-compatibility-verdict.md`
- `workspaces/black-box/runs/clid-threshold-compatibility-20260416-r1/summary.json`

Value: ⭐⭐

#### ⬜ `BB-4` Mitigation-aware evaluation

Goal: test black-box methods under at least one more realistic mitigation condition

Current read:

- the first realistic deployment-side mitigation candidate is now explicit:
  - `served-image-sanitization`
  - implemented as mild `JPEG recompression + resize 512 -> 448 -> 512`
- a bounded `CLiD clip` probe has already been executed on that mitigation:
  - sanitized probe `AUC = 1.0 / ASR = 1.0 / TPR@1%FPR = 1.0`
  - frozen local baseline also `AUC = 1.0 / ASR = 1.0 / TPR@1%FPR = 1.0`
- utility remained high enough (`mean PSNR = 38.286 dB`, `mean MAE = 1.879`) that the null result is meaningful
- current black-box mitigation verdict is therefore `negative but useful no-go`

Tasks:

- [x] `BB-4.1` design protocol
- [x] `BB-4.2` select mitigation
- [x] `BB-4.3` evaluate recon + CLiD or new challenger
- [x] `BB-4.4` record quality-cost-utility trade-off

Canonical evidence anchor:

- `workspaces/black-box/2026-04-16-blackbox-served-image-mitigation-verdict.md`

Value: ⭐⭐

#### ✅ `BB-5` Variation asset-contract clarification

Goal: remove ambiguity around what exactly is needed before `variation` can leave blocked state

Current read:

- `variation` is no longer blocked in a vague or exploratory sense
- the repo already has a concrete recovery contract:
  - `query_image_root / query images`
  - real variation endpoint or equivalent proxy
  - explicit query budget
  - frozen attack parameters
- the first hard gate remains `query_image_root`, not runner code

Tasks:

- [x] `BB-5.1` review the existing blocked note and recovery template
- [x] `BB-5.2` record the explicit unblock contract

Canonical evidence anchor:

- `workspaces/black-box/2026-04-16-variation-asset-contract-verdict.md`

Value: ⭐⭐

#### ✅ `BB-6` Same-protocol cross-method score package

Goal: test whether the current black-box headline and challenger can form a bounded package that improves actionability without pretending to be a new family

Current read:

- `Recon` remains the black-box headline
- `semantic-auxiliary-classifier` remains the leading new-family challenger
- current `semantic-aux` scoring and fusion line is already closed unless:
  - a new feature family appears, or
  - a same-protocol cross-method score package appears
- the shortest honest reopen path is therefore:
  - align existing `Recon` and `semantic-aux` score artifacts under one bounded comparison contract
- current alignment check now says the existing artifacts do **not** yet share one clean same-protocol surface:
  - admitted `Recon` is frozen on `public-* / proxy-shadow-member / celeba_partial_target`
  - current `semantic-aux` challenger is frozen on the local CelebA target-family stack with `celeba_target/checkpoint-25000`
- current release posture:
  - `gpu_release = none`
  - first pass should stay CPU-only

Tasks:

- [x] `BB-6.1` align `Recon` and `semantic-aux` score artifacts under one same-protocol comparison contract
- [x] `BB-6.2` test one bounded score package against the best single method
- [x] `BB-6.3` record whether the package changes actionability or only calibration

Canonical evidence anchor:

- `workspaces/black-box/2026-04-16-blackbox-next-lane-score-package-selection-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-score-package-alignment-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-score-package-contract-selection-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-score-package-aligned-comparator-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-score-package-verdict.md`

Updated verdict:

- `BB-6.1` closes as `negative but useful`
- current blocker is protocol/asset mismatch, not missing score logic
- one aligned semantic-aux comparator on the selected contract surface is now `execution-positive`:
  - `public-50 step10 / celeba_partial_target / 16 / 16`
  - `AUC = 0.859375`, `ASR = 0.8125`, `TPR@1%FPR = 0.625`
- first same-protocol package test is now `positive but bounded` on the aligned `16 / 16` split:
  - `Recon dim0 AUC = 0.820312`
  - `semantic-aux mean_cos AUC = 0.902344`
  - `z-score sum AUC = 0.933594`
  - `z-score max ASR = 0.875`
- current honest reading is:
  - bounded same-protocol black-box packaging can improve actionability
  - but the result is still too small and too local to replace the frozen `Recon` headline package

Value: ⭐⭐⭐

#### ⬜ `BB-7` Post-second-signal black-box next-question review

Goal: decide whether black-box currently still contains any honest new GPU-worthy question, after `semantic-auxiliary-classifier` stabilized as the leading challenger and the current side branches were either bounded, blocked, or closed

Current read:

- `Recon` remains the frozen black-box headline
- `semantic-auxiliary-classifier` remains the leading second-signal challenger
- current scoring/fusion work already closed as `negative but useful`
- `CLiD` is now fixed to `evaluator-near local clip-only corroboration`
- `variation` is now `contract-ready blocked`
- the remaining question is therefore lane status and GPU release, not more same-family execution

Tasks:

- [x] `BB-7.1` review whether any current black-box branch still exposes a real new GPU-worthy question
- [x] `BB-7.2` decide whether current challenger / corroboration / blocked branches justify more bounded execution or should now be frozen
- [x] `BB-7.3` record the carry-forward rule for future black-box reopen conditions

Canonical evidence anchor:

- `workspaces/black-box/2026-04-16-post-second-signal-blackbox-next-question-review.md`

Selection verdict:

- `BB-7` now closes as `negative but stabilizing`
- black-box currently has `no-new-gpu-question`
- current stable hierarchy is:
  - `Recon = headline`
  - `semantic-auxiliary-classifier = leading challenger`
  - `CLiD = corroboration / boundary-only`
  - `variation = contract-ready blocked`
- future black-box reopen should require a genuinely new feature family or a real asset/boundary change, not more scoring-only retries or mechanical scale-up
- `gpu_release = none`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

---

### 6.3 Gray-box expansion

#### ✅ `GB-1` Second gray-box defense

Goal: avoid a single defended story dominating gray-box forever

Current read:

- `stochastic-dropout(all_steps)` remains the only defended gray-box mainline
- `epsilon-precision-throttling` is already a negative bounded candidate
- new bounded candidate `epsilon-output-noise (std = 0.1)` also landed negative on `cpu-32`, so it should not be released to GPU
- `input-gaussian-blur (sigma = 1.0)` landed even more negatively on `cpu-32`, strengthening rather than weakening the attack
- the cheap perturbation-style `GB-1` frontier is effectively exhausted for now
- new challenger-specific hypothesis `TMIA-DM late-window temporal-striding(stride=2)` is now repeat-positive on `cpu-32`:
  - `seed0`: `AUC = 0.697266` versus undefended `0.823242`
  - `seed1`: `AUC = 0.696289` versus undefended `0.760742`
- this is the first post-dropout `TMIA-DM` defense candidate that has now survived both `cpu-32` and `GPU128`
- exact-contract `GPU128` repeats landed at `AUC = 0.727234` and `0.711609`, both clearly below undefended `TMIA-DM` and the current `dropout(all_steps)` defense
- exact-contract `GPU256` repeats then landed at `AUC = 0.733322` and `0.7173`, again clearly below undefended `TMIA-DM` and the current `dropout(all_steps)` defense
- defended operating-point comparison and system-sync review are now complete:
  - `PIA` remains the defended headline
  - `TMIA + temporal-striding(stride=2)` is now the strongest defended gray-box challenger reference

Tasks:

- [x] `GB-1.1` shortlist materially different defense mechanisms
- [x] `GB-1.2` pick one bounded candidate
- [x] `GB-1.3` compare the defended challenger against current gray-box references when possible
- [x] `GB-1.4` record defended verdict

Canonical evidence anchors:

- `workspaces/gray-box/2026-04-15-graybox-defense-precision-throttling-note.md`
- `workspaces/gray-box/2026-04-15-graybox-epsilon-output-noise-defense-verdict.md`
- `workspaces/gray-box/2026-04-15-graybox-input-blur-defense-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-second-defense-shortlist-review.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu128-verdict.md`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu256-verdict.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`
- `workspaces/gray-box/2026-04-16-graybox-second-defense-closure-verdict.md`

Verdict:

- current `GB-1` round closes as `positive`
- gray-box defended story is now explicitly multi-family:
  - `PIA` remains the defended headline
  - `TMIA + temporal-striding(stride=2)` is the defended challenger reference
- future gray-box work should move to new-family diversity or disagreement, not another broad defense shortlist

Value: ⭐⭐⭐

#### ✅ `GB-2` Ranking-sensitive variable

Goal: find something that changes ranking, not just score scale

Tasks:

- [x] `GB-2.1` analyze disagreement patterns
- [x] `GB-2.2` propose bounded variable
- [x] `GB-2.3` probe with minimal budget
- [x] `GB-2.4` record verdict

Status:

- completed on the current `PIA vs SecMI` branch
- canonical evidence anchor:
  - `workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`

Verdict:

- same-split `PIA` and `SecMI` scores are highly correlated (`Spearman = 0.907588`)
- disagreement exists (`12.2559%`), but simple score averaging does not beat the better single method
- current gray-box ranking-sensitive branch is therefore `negative but useful`, not promotion-worthy for naive fusion

Reopen rule:

- only reopen with a new bounded hypothesis such as class-conditional disagreement or confidence-gated switching
- do not spend more budget on another naive score ensemble

Value: ⭐⭐⭐

#### ✅ `GB-3` New gray-box family

Goal: test whether another family is worth joining the long-term mainline

Candidate directions:

- SIMA
- MoFit
- noise-as-probe
- SIDe
- another paper-backed gray-box mechanism

Current read:

- after repeated negative `GB-1` perturbation candidates, gray-box expansion should favor new-family diversity
- `SimA` is the current best next family because it is materially different from `PIA` yet still fits the current CIFAR-10 DDPM asset line
- `TMIA-DM` and `MoFit` remain valid future branches, but not the shortest next execution path
- local `structural memorization` has now been reviewed as a separate gray-box family candidate on the current CelebA target-family approximation:
  - bounded smoke landed `AUC = 0.375 / ASR = 0.53125`
  - member mean structure score stayed below non-member mean (`0.730527 < 0.75017`)
  - the branch is therefore `negative but useful` under the current local threat model and should not receive more GPU in its present form
- bounded local `SimA` feasibility is now execution-positive but strength-negative:
  - it runs
  - it does not currently merit challenger promotion or GPU release
- bounded later-timestep `SimA` rescan also completes:
  - best timestep shifts from `120` to `160`
  - `AUC` improves from `0.542969` to `0.584961`
  - but the family still remains below challenger quality and below GPU-release threshold
- after the weak `SimA` result, the next active gray-box family branch should pivot to `TMIA-DM protocol / asset decomposition`, not immediate `SimA` reopen and not a large `MoFit` jump
- `TMIA-DM` is now protocol-ready in the repo sense:
  - signal surface, access assumption, local-fit path, and minimal smoke entry are now concrete
  - execution release and GPU release remain `none`
- bounded local `TMIA-DM protocol probe` is now execution-positive but family-mixed:
  - `long_window` shows the only credible positive branch so far (`AUC = 0.702148`, `ASR = 0.703125`)
  - `short_window` is negative and naive fusion adds no value
  - GPU release and challenger promotion remain `none`
- bounded `TMIA-DM long_window` repeat is now directionally stable across two `cpu-32` seeds:
  - repeat `seed1` dropped from `AUC = 0.702148` to `AUC = 0.663086`, so stability is real but only moderate
  - `long_window` remains the only active refinement branch
  - GPU release still remains `none`
- bounded `TMIA-DM late-window long_window` refine changed the release decision:
  - late-window `[80,100,120]` reached `AUC = 0.823242` on `seed0` and `AUC = 0.760742` on `seed1`
  - this is the first `TMIA-DM` local branch that looks competitive with `PIA`
  - minimal GPU pilot is now justified, but headline replacement is still not
- attempted `TMIA-DM` GPU pilot is currently execution-blocked locally:
  - the default `Research` interpreter is `torch 2.11.0+cpu`
  - `torch.cuda.is_available()` is false
  - this is an environment blocker, not a negative method verdict
- the local GPU blocker has now been resolved:
  - `conda` env `diffaudit-research` is CUDA-capable and is the admitted runtime
  - repairing the editable package target restored `diffaudit` imports inside that env
- first `TMIA-DM late-window GPU128` rung landed positive:
  - `AUC = 0.825317`, `ASR = 0.769531`, `TPR@1%FPR = 0.085938`
  - this is enough for challenger status, but not yet for headline promotion
- second matched `TMIA-DM late-window GPU128` rung confirmed the signal:
  - `AUC = 0.836975`, `ASR = 0.769531`, `TPR@1%FPR = 0.078125`
  - the line is now repeat-confirmed as a real GPU challenger
- first `TMIA-DM late-window GPU256` rung kept the line alive:
  - `AUC = 0.839554`, `ASR = 0.765625`
  - it is near-parity with `PIA GPU256` on headline metrics and stronger on low-FPR behavior
- second `TMIA-DM late-window GPU256` rung confirmed scale stability:
  - `AUC = 0.837814`, `ASR = 0.787109`
  - the line is now repeat-confirmed at both `GPU128` and `GPU256`
- first `TMIA-DM` defense interaction is now explicit:
  - `stochastic-dropout(all_steps)` weakened the line from `AUC = 0.825317` to `0.809326`
  - but the challenger remained stronger than current defended `PIA` on `AUC` and low-FPR behavior
- second defended `TMIA-DM` rung confirmed that result:
  - defended repeat landed at `AUC = 0.819397`, `ASR = 0.757812`
  - `TMIA-DM` is now repeat-confirmed even under the current dropout defense
- first defended `TMIA-DM GPU256` rung kept the line alive at scale:
  - `AUC = 0.825867`, `ASR = 0.746094`
  - the line is slightly behind defended `PIA` on headline metrics but remains far stronger on low-FPR behavior
- second defended `TMIA-DM GPU256` rung confirmed scale stability:
  - `AUC = 0.82164`, `ASR = 0.765625`
  - the defended challenger is now repeat-confirmed at both `GPU128` and `GPU256`
- challenger-specific `late_steps_only` dropout ablation is now classified:
  - on `TMIA-DM late-window GPU128`, it only reduced `AUC` from `0.825317` to `0.820984`
  - it is weaker than `all_steps` and should remain an ablation, not a new defended mainline
- challenger-specific `timestep-jitter(radius=10)` defense is now rejected:
  - on `TMIA-DM late-window GPU128`, it raised `AUC` from `0.825317` to `0.850098`
  - it strengthens the challenger and should not receive more budget in its current form
- challenger-specific `temporal-striding(stride=2)` defense is now the next credible gate:
  - on `TMIA-DM late-window CPU32`, it reduced `AUC` from `0.823242` to `0.697266` on `seed0`
  - on the paired `seed1` repeat, it again reduced `AUC` from `0.760742` to `0.696289`
  - on `GPU128`, it then reduced `AUC` from `0.825317 / 0.836975` to `0.727234 / 0.711609`
  - on `GPU256`, it then held at `0.733322 / 0.7173`, remaining far below the paired undefended and dropout-defended challenger
  - it is now repeat-confirmed through `GPU256` and becomes a defended comparison / packaging candidate rather than a mere next gate

Tasks:

- [x] `GB-3.1` choose one family
- [x] `GB-3.2` write feasibility note
- [x] `GB-3.3` implement probe or smoke
- [x] `GB-3.4` record verdict

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-graybox-sima-feasibility-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-sima-rescan-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-structural-memorization-verdict.md`
- follow-up branch selector:
  - `workspaces/gray-box/2026-04-16-graybox-next-family-reselection.md`
- protocol-ready follow-up note:
  - `workspaces/gray-box/2026-04-16-tmiadm-protocol-and-asset-note.md`
- protocol-probe verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-protocol-probe-verdict.md`
- long-window repeat verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-long-window-repeat-verdict.md`
- late-window refine verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-late-window-refine-verdict.md`
- gpu pilot blocker:
  - `workspaces/gray-box/2026-04-16-tmiadm-gpu-pilot-blocker.md`
- gpu pilot verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-gpu-pilot-verdict.md`
- gpu repeat verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-gpu-repeat-verdict.md`
- gpu256 rung verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-gpu256-rung-verdict.md`
- gpu256 repeat verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-gpu256-repeat-verdict.md`
- dropout-defense verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-verdict.md`
- dropout-defense repeat verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-repeat-verdict.md`
- dropout-defense gpu256 rung verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-gpu256-rung-verdict.md`
- dropout-defense gpu256 repeat verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-dropout-defense-gpu256-repeat-verdict.md`
- late-steps dropout defense verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-latesteps-dropout-defense-verdict.md`
- timestep-jitter defense verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-timestep-jitter-defense-verdict.md`
- temporal-striding defense verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- temporal-striding gpu128 verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu128-verdict.md`
- temporal-striding gpu256 verdict:
  - `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-gpu256-verdict.md`

Value: ⭐⭐

#### ✅ `GB-4` Disagreement exploitation

Goal: if a second family lands, determine whether disagreement is useful or redundant

Current read:

- run-level same-split comparison is now explicit:
  - `PIA cpu-32` remains clearly ahead of `TMIA-DM long_window` across both bounded repeats
  - current `TMIA-DM` therefore does not justify role swap or immediate disagreement-exploitation work
- GPU-level operating-point comparison is now also explicit:
  - `PIA` remains the safest single headline on global-ranking metrics
  - `TMIA-DM late-window` is now the strongest challenger line, with credible low-FPR advantages
- defended operating-point comparison is now explicit too:
  - `stochastic-dropout(all_steps)` weakens both families
  - `TMIA-DM late-window` remains the strongest defended challenger family, not a collapsed side branch
- defended challenger ordering has now changed inside the `TMIA-DM` family:
  - `TMIA + temporal-striding(stride=2)` is materially stronger than `TMIA + dropout(all_steps)` on the `GPU256` defended rung
  - summary-layer gray-box artifacts should therefore prefer `temporal-striding` as the defended challenger reference
- `GB-4` should stay narrow until either `TMIA-DM` strengthens further or a different second family lands
  - current best reading is `headline plus challenger plus TMIA-specific defended branch`, not `single-family gray-box story`

Tasks:

- [x] `GB-4.1` compare on aligned split
- [x] `GB-4.2` measure correlation and disagreement
- [x] `GB-4.3` test whether disagreement improves actionability

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-long-window-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-operating-point-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-disagreement-exploitation-verdict.md`

Verdict:

- current `GB-4` round is `positive but bounded`
- `PIA` and `TMIA-DM` are not rank-identical:
  - undefended `Spearman = 0.604766 / 0.651955`
  - defended `GPU256 Spearman = 0.302501`
- top-rank overlap remains partial rather than collapsed:
  - undefended `top-32 overlap = 0.3125 / 0.21875`
  - defended `GPU256 top-32 overlap = 0.0`
- bounded same-split actionability is now positive:
  - z-scored sum beats the best single method on both undefended aligned checks
  - it also slightly beats defended `PIA` on the defended `GPU256` check
- keep the result at `offline same-split actionability gain`, not immediate headline replacement and not GPU-release evidence

Value: ⭐⭐

#### ⬜ `GB-5` Genuinely-new-family selector

Goal: reopen gray-box innovation without wasting GPU on already-packaged `TMIA-DM` branches

Current read:

- current gray-box headline and defended challenger packaging are already strong for this round
- the gray-box plan explicitly says the next shortest step should be another lane or a truly new mechanism
- current black-box plan still says `no immediate black-box rerun`
- white-box just completed one bounded candidate-generation round on `DP-LoRA`

Tasks:

- [x] `GB-5.1` shortlist the next real gray-box family candidates
- [x] `GB-5.2` reject the near-miss alternatives for now
- [x] `GB-5.3` define one first bounded smoke and future `gpu_release` conditions

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-graybox-new-family-selector-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-new-family-shortlist-refresh-verdict.md`

Selection verdict:

- current reselection round is `positive`
- `Noise as a Probe` is now the selected next genuinely-new family
- `MoFit / SIDe / SimA reopen / structural memorization reopen` are rejected for now
- the first bounded next step is a CPU-first `Noise as a Probe protocol / asset contract`
- `gpu_release = none`
- current `GB-5` lane closes as `positive`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-6` Noise-as-a-Probe protocol / asset contract

Goal: decide whether the current repo can support one honest first smoke for `Noise as a Probe` without faking asset readiness

Current read:

- the family is selected because it opens a genuinely new gray-box interface around controllable initial noise
- it is still not execution-ready on current repo truth
- the next honest step is to lock target family, pretrain base, custom-noise path, and calibration split before any smoke request

Tasks:

- [x] `GB-6.1` select one honest local target family for the first contract
- [x] `GB-6.2` write the minimum asset/interface checklist
- [x] `GB-6.3` define the first bounded smoke and `gpu_release` gate

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-protocol-asset-contract.md`

Selection verdict:

- `GB-6` now closes as `positive but bounded`
- first honest local target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- first bounded smoke:
  - `one member + one non-member interface canary`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-7` Noise-as-a-Probe implementation-surface review

Goal: decide whether the current repo already has enough primitives for the first interface canary, or whether a real implementation block still remains

Current read:

- target-side SD1.5 + LoRA loading already exists
- target-side prompt-conditioned generation already exists
- VAE encode/decode plus `DDIMScheduler` stepping already exists
- image-distance scoring already exists
- but the repo still lacks an explicit reusable `DDIM inversion + custom-noise target-generation` path

Tasks:

- [x] `GB-7.1` audit current code for target generation primitives
- [x] `GB-7.2` audit current code for inversion / custom-noise gaps
- [x] `GB-7.3` record whether the interface canary is implementation-ready or still blocked by glue code

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-implementation-surface-review.md`

Selection verdict:

- `GB-7` now closes as `positive but bounded`
- current repo is not yet end-to-end canary-ready
- missing block:
  - reusable `DDIM inversion + custom-noise target-generation` glue
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-8` Noise-as-a-Probe canary scaffold

Goal: build the smallest reusable local harness that can power the one-member/one-non-member interface canary

Current read:

- the contract is frozen
- the implementation surface is partially ready
- the remaining work is now concrete code, not more abstract candidate review

Tasks:

- [x] `GB-8.1` choose whether to extend an existing script or add a dedicated canary script
- [x] `GB-8.2` implement inversion + custom-noise generation glue
- [x] `GB-8.3` define the first canary output schema

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-canary-scaffold-selection.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-canary-scaffold-implementation-verdict.md`

Selection verdict:

- `GB-8` is the new live CPU-first lane
- `GB-8.1` now closes as `positive`
  - selected option:
    - dedicated canary script
- `GB-8.2` now closes as `positive`
  - dedicated script added:
    - `scripts/run_noise_as_probe_interface_canary.py`
- `GB-8.3` now closes as `positive`
  - canary output schema is now explicit in the new dedicated scaffold
- current `GB-8` lane closes as `positive but bounded`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-9` Noise-as-a-Probe calibration / expansion policy

Goal: decide how the first successful interface canary should expand without pretending immediate benchmark readiness

Current read:

- the first one-member/one-non-member interface canary already completed end to end
- prompt source, custom-noise path, and canary schema are now frozen in practice
- the remaining missing piece before any larger ask is calibration / expansion policy

Tasks:

- [x] `GB-9.1` define the smallest honest calibration set policy
- [x] `GB-9.2` define the first expansion rung beyond `1 + 1`
- [x] `GB-9.3` keep `gpu_release` and narrative boundary explicit

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-calibration-expansion-policy.md`

Selection verdict:

- `GB-9` now closes as `positive`
- smallest honest calibration set:
  - `8` prior non-members
- first expansion rung:
  - `8 members + 8 eval non-members + 8 calibration non-members`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-10` Noise-as-a-Probe first expansion rung

Goal: check whether the first local `Noise as a Probe` signal survives a bounded expansion beyond the `1 + 1` canary

Current read:

- the first interface canary already completed end to end
- calibration and expansion policy are now explicit
- the next honest question is whether the signal survives the first `8 / 8 / 8` rung without pretending benchmark quality

Tasks:

- [x] `GB-10.1` run the first bounded `8 / 8 / 8` rung
- [x] `GB-10.2` record calibration-threshold behavior
- [x] `GB-10.3` decide whether the branch stays alive or collapses immediately

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-expansion-rung-verdict.md`

Selection verdict:

- `GB-10` now closes as `positive but bounded`
- member-vs-nonmember mean `MSE` separation remains alive on the first bounded rung
- simple percentile-style calibration also remains directionally usable
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: single bounded GPU rung

#### ⬜ `GB-11` Noise-as-a-Probe expansion repeat

Goal: decide whether the first successful `8 / 8 / 8` rung survives a disjoint repeat, or whether it was only a lucky split

Current read:

- the first `8 / 8 / 8` rung was positive but bounded
- the highest-value uncertainty now is split sensitivity, not canary plumbing

Tasks:

- [x] `GB-11.1` run one disjoint repeat of the `8 / 8 / 8` rung
- [x] `GB-11.2` compare direction, not just absolute threshold values
- [x] `GB-11.3` decide whether the branch is now repeat-positive or still one-off

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-expansion-repeat-verdict.md`

Selection verdict:

- `GB-11` now closes as `positive but bounded`
- the branch is now `repeat-positive` on bounded `8 / 8 / 8` rungs
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: single bounded GPU rung

#### ⬜ `GB-12` Noise-as-a-Probe threshold hardening

Goal: decide whether the current calibration-only percentile threshold is stable enough to keep as a bounded local contract, or whether split sensitivity is still too high to trust even locally

Current read:

- the branch is now `repeat-positive` across two disjoint bounded `8 / 8 / 8` rungs
- the highest-value uncertainty is no longer basic signal existence
- the next honest question is whether thresholding survives cross-run reuse, not whether the mean-gap still exists once more

Tasks:

- [x] `GB-12.1` verify the exact implemented threshold rule against the current canary script
- [x] `GB-12.2` compare self-threshold and cross-run threshold behavior on `r1` and `r2`
- [x] `GB-12.3` decide whether a frozen conservative local threshold band is honest enough for one larger bounded rung

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-threshold-hardening-verdict.md`

Selection verdict:

- `GB-12` now closes as `positive but bounded`
- calibration-only low-percentile thresholding remains locally coherent across the first two bounded repeats
- a frozen conservative local threshold band now exists:
  - `1304.8905 .. 1308.7131`
- release-grade thresholding is still `no-go`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-13` Noise-as-a-Probe larger bounded rung

Goal: test whether the current local `Noise as a Probe` line stays alive on a larger disjoint bounded rung, and whether the conservative threshold band from `GB-12` still works on that larger split

Current read:

- the branch is already `repeat-positive` on bounded `8 / 8 / 8`
- `GB-12` fixed a conservative threshold band and removed the biggest threshold-stability uncertainty
- the next honest GPU question is scale within bounds, not another tiny rung

Tasks:

- [x] `GB-13.1` run one larger disjoint bounded rung
- [x] `GB-13.2` compare self-threshold and frozen-threshold behavior on the larger split
- [x] `GB-13.3` decide whether the line is now strong enough to justify one same-scale repeat instead of immediate promotion

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-larger-rung-verdict.md`

Selection verdict:

- `GB-13` now closes as `positive`
- member-vs-nonmember separation remains strong on the larger disjoint rung
- the `GB-12` frozen conservative threshold band transfers cleanly onto the larger split
- the branch is now `strengthened but still bounded`
- `gpu_release = none`

Value: ⭐⭐⭐⭐
Budget: single bounded GPU rung

#### ⬜ `GB-14` Noise-as-a-Probe larger-rung repeat

Goal: decide whether the first strong `16 / 16 / 16` rung survives one disjoint same-scale repeat, and whether the frozen conservative threshold story still stays cleaner than newly re-fit thresholds

Current read:

- `GB-13` materially strengthened the branch
- the remaining highest-value uncertainty is now same-scale repeatability, not simple threshold viability
- a second same-scale rung is cheaper and more honest than jumping straight to packaging promotion

Tasks:

- [x] `GB-14.1` run one disjoint repeat at the same `16 / 16 / 16` scale
- [x] `GB-14.2` compare self-threshold versus frozen-threshold behavior
- [x] `GB-14.3` decide whether the branch is now a strengthened bounded challenger candidate or still just one strong rung

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-larger-rung-repeat-verdict.md`

Selection verdict:

- `GB-14` now closes as `positive`
- the branch is now same-scale `repeat-positive` at `16 / 16 / 16`
- the frozen conservative threshold story remains stronger than the new self-fit `r4` threshold
- the line is now a `strengthened bounded challenger candidate`
- `gpu_release = none`

Value: ⭐⭐⭐⭐
Budget: single bounded GPU rung

#### ⬜ `GB-15` Noise-as-a-Probe challenger-boundary review

Goal: decide the honest narrative and packaging boundary for `Noise as a Probe` after the same-scale `16 / 16 / 16` repeat-positive result

Current read:

- `GB-14` moved the branch beyond a one-off stronger rung
- the top remaining uncertainty is packaging level, not raw execution reality
- the next honest task is CPU-side review, not another blind GPU expansion

Tasks:

- [x] `GB-15.1` compare `Noise as a Probe` packaging pressure against current `PIA/TMIA-DM` gray-box hierarchy
- [x] `GB-15.2` decide whether it is ready to replace the current active challenger
- [x] `GB-15.3` record exact wording boundary for higher-layer summaries

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-challenger-boundary-review.md`

Selection verdict:

- `GB-15` now closes as `positive`
- `PIA` remains admitted headline
- `TMIA-DM` remains strongest packaged active challenger
- `Noise as a Probe` is now a `strengthened bounded challenger candidate`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-16` Noise-as-a-Probe summary-layer sync

Goal: synchronize higher-layer research summaries to the current gray-box truth after `GB-15`, without prematurely promoting `Noise as a Probe` above its honest boundary

Current read:

- the local branch is now strong enough to be mentionable in higher-layer summaries
- `docs/comprehensive-progress.md` still carries stale gray-box wording in several places
- the highest-value next CPU action is summary truth sync, not more execution

Tasks:

- [x] `GB-16.1` identify stale gray-box summary wording
- [x] `GB-16.2` update summary-layer wording for `SecMI`, `TMIA-DM`, and `Noise as a Probe`
- [x] `GB-16.3` keep headline and active challenger boundaries explicit

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-summary-layer-sync-verdict.md`

Selection verdict:

- `GB-16` now closes as `positive`
- higher-layer gray-box summary now reflects:
  - `PIA = headline`
  - `TMIA-DM = strongest packaged challenger`
  - `Noise as a Probe = strengthened bounded challenger candidate`
- `gpu_release = none`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-17` Noise-as-a-Probe defended-extension feasibility review

Goal: decide whether `Noise as a Probe` now has one honest minimal defended-extension task, or whether any defended branch is still below release on the current local contract

Current read:

- the attack-side branch is now strong enough for bounded challenger-candidate wording
- but there is still no defended line for this family
- the highest-value next question is therefore feasibility, not blind defense execution

Tasks:

- [x] `GB-17.1` review whether `PIA/TMIA-DM` defense hooks transfer directly to this family
- [x] `GB-17.2` review whether the current local `SD1.5` execution surface exposes a real minimal dropout path
- [x] `GB-17.3` decide whether any defended-extension GPU gate should be opened now

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-noise-as-probe-defended-extension-feasibility-review.md`

Selection verdict:

- `GB-17` now closes as `negative but useful`
- there is no honest minimal defended-extension task on the current local contract
- direct `stochastic-dropout` port is `no-go`
- direct `temporal-striding` port is also `no-go`
- `gpu_release = none`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-18` Post-temporal-striding gray-box next-question review

Goal: decide whether gray-box currently still contains any honest new GPU-worthy question, after `TMIA-DM + temporal-striding(stride=2)` closed as the strongest defended challenger reference and `Noise as a Probe` finished its boundary review

Current read:

- defended gray-box packaging is now stable:
  - `PIA + stochastic-dropout(all_steps)` remains the admitted defended headline
  - `TMIA-DM late-window + temporal-striding(stride=2)` is the strongest defended challenger reference
- `Noise as a Probe` is now a strengthened bounded challenger candidate
- but `GB-17` already showed that the current local `SD1.5` contract does not expose an honest minimal defended-extension path for `Noise as a Probe`
- the remaining question is therefore lane status and GPU release, not more same-family execution

Tasks:

- [x] `GB-18.1` review whether any current gray-box branch still exposes a real new GPU-worthy question
- [x] `GB-18.2` decide whether current TMIA / Noise-as-a-Probe branches justify more bounded rungs or should now be frozen
- [x] `GB-18.3` record the carry-forward rule for future gray-box reopen conditions

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-post-temporal-striding-graybox-next-question-review.md`

Selection verdict:

- `GB-18` now closes as `negative but stabilizing`
- gray-box currently has `no-new-gpu-question`
- current stable hierarchy is:
  - `PIA = admitted headline`
  - `TMIA + temporal-striding(stride=2) = strongest defended challenger reference`
  - `Noise as a Probe = strengthened bounded challenger candidate`
- future gray-box reopen should require a genuinely new mechanism or real contract change, not more same-family rungs or fusion retries
- `gpu_release = none`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

---

### 6.4 White-box expansion

#### ✅ `WB-1` Gradient extraction blocker resolution

Goal: decide whether the current white-box blocker is solvable or should become a hard no-go for now

Tasks:

- [x] `WB-1.1` run gradient extraction directly
- [x] `WB-1.2` capture root cause
- [x] `WB-1.3` fix or classify blocker
- [x] `WB-1.4` verify with one minimal job

Status:

- completed with a direct upstream replay on the admitted target-member asset line
- canonical evidence anchor:
  - `workspaces/white-box/2026-04-15-whitebox-gradient-extraction-direct-verdict.md`

Verdict:

- white-box direct gradient extraction is runnable on the current admitted asset line
- `attack_method = 2` is not the blocker
- the real execution-layer root causes were:
  - wrong dataset mode when `--dataset_name cifar10` is forced onto imagefolder-style assets
  - missing output parent directory before `torch.save(...)`

Carry-forward rule:

- stop treating gradient extraction as a fundamental white-box blocker
- move `WB-2` to a narrower follow-up question instead of a blind blocker hunt

Value: ⭐⭐⭐

#### ✅ `WB-2` Second white-box line verdict

Goal: land or reject a second white-box line with evidence

Possible branches:

- GSA2 comparator
- NeMo if blocker removed
- another non-GSA family

Current live path:

- `GSA2 comparator`
- canonical evidence anchor:
  - `workspaces/white-box/2026-04-16-whitebox-gsa2-bounded-comparator-verdict.md`

Current read:

- `Finding NeMo` is already closed negatively for the current wave
- `Local Mirror` still collapses back into the admitted `GSA` family
- the bounded `GSA2` comparator now completes on the admitted target pair plus the first shadow pair
- `GSA2` is strong (`AUC = 0.922498`) but still below admitted `GSA1 1k-3shadow` (`AUC = 0.998192`)
- the correct promotion level is `positive secondary line / corroboration`, not new headline and not new family

Tasks:

- [x] `WB-2.1` choose path based on `WB-1`
- [x] `WB-2.2` implement and test
- [x] `WB-2.3` record verdict

Carry-forward rule:

- stop spending white-box budget on more blind `GSA2` canaries
- only reopen for a specific bounded reason such as cost-vs-strength comparison or multi-shadow expansion
- keep the next white-box budget on `WB-3` or system-consumable packaging

Value: ⭐⭐

#### ✅ `WB-3` White-box defense breadth

Goal: broaden white-box defense comparison beyond the current single defended comparator

Current read:

- the current repo still has only one executable defended white-box family:
  - `DPDM / W-1`
- `Finding NeMo` remains a mechanism / observability extension on `zero-GPU hold`, not a released defense comparator
- `Local Mirror` is not a defense candidate and does not create breadth
- `DPDM` strong-v2 / strong-v3 / same-protocol variants widen comparator depth, not family diversity
- current `WB-3.1` verdict is therefore `negative but useful`: there is no second distinct executable white-box defense family in the repo yet

Tasks:

- [x] `WB-3.1` survey alternatives beyond DPDM
- [x] `WB-3.2` pick one bounded defense
- [x] `WB-3.3` keep attack-vs-defense execution closed until a second family exists
- [x] `WB-3.4` record attack degradation boundary as `not-requestable`

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-whitebox-defense-breadth-shortlist-verdict.md`
- `workspaces/white-box/2026-04-16-whitebox-bounded-defense-selection-verdict.md`
- `workspaces/white-box/2026-04-16-whitebox-defense-breadth-closure-verdict.md`

Updated verdict:

- `WB-3` now closes as `negative but useful`
- selected bounded defense:
  - `none`
- current white-box breadth truth is:
  - there is still no second distinct executable defended family on the admitted asset line
  - `WB-3.3` and `WB-3.4` are therefore `not-requestable` until a genuinely new family appears
  - current white-box budget should stay off breadth execution and move to candidate generation, import, or another lane

Value: ⭐⭐

#### ✅ `WB-4` White-box feature/trajectory upgrade

Goal: revisit deeper white-box analysis if and only if blocker cost becomes reasonable

Tasks:

- [x] `WB-4.1` write hypothesis first
- [x] `WB-4.2` run lightweight probe
- [x] `WB-4.3` record whether it changes story

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-whitebox-feature-trajectory-verdict.md`

Verdict:

- the migrated DDPM observability contract still resolves cleanly on current admitted assets
- but that readiness still does not change white-box story:
  - `gpu_release = none`
  - `validation-smoke release = none`
  - `admitted_change = none`
- current feature/trajectory branch is therefore `negative but useful`:
  - entry plumbing exists
  - story-level upgrade still does not

Value: ⭐

#### ⬜ `WB-5` DP-LoRA comparability dossier

Goal: open the next honest white-box successor lane without pretending execution readiness or benchmark comparability too early

Current read:

- `WB-3` closed with `none selected`, so the next white-box budget should move to candidate generation rather than fake breadth execution
- `Finding NeMo` remains under a stricter `zero-GPU hold` and still requires a separate hypothesis/budget review before any reopen
- `DP-LoRA` already has a defined CPU-first dossier shape in the long-horizon plan:
  - `protocol overlap note`
  - `minimal config candidate`
  - `no-go triggers`
- `phase-e-candidates.json` had become stale because `TMIA-DM` was still listed as an intake-only candidate after promotion into an executed gray-box challenger branch

Tasks:

- [x] `WB-5.1` write the `DP-LoRA` protocol-overlap note against the current admitted `DDPM/CIFAR-10 + GSA/W-1` line
- [x] `WB-5.2` define one minimal local config candidate instead of a vague future defense idea
- [x] `WB-5.3` define explicit `no-go` and future `gpu_release` triggers

Canonical evidence anchor:

- `workspaces/intake/2026-04-16-phase-e-registry-refresh-and-dplora-selection-verdict.md`
- `workspaces/white-box/2026-04-16-dplora-protocol-overlap-note.md`
- `workspaces/white-box/2026-04-16-dplora-minimal-local-config-candidate.md`
- `workspaces/white-box/2026-04-16-dplora-no-go-and-gpu-release-triggers.md`

Selection verdict:

- current reselection round is `positive`
- `DP-LoRA` becomes the next live `CPU-first` lane
- `Finding NeMo` remains `zero-GPU hold`
- `TMIA-DM` should no longer remain on the `Phase E` intake candidate surface
- `gpu_release = none`
- `WB-5.1` now closes as `positive but bounded`:
  - `DP-LoRA` has real defense-family overlap with the current white-box story
  - but only at `partial-overlap` level, not same-protocol comparability
- `WB-5.2` now closes as `positive but bounded`:
  - the frozen local translation candidate is `lambda=0.1 / rank=4 / epochs=10`
  - it is a local `DDPM/CIFAR10` bridge candidate, not a paper-faithful `DP-LoRA` release packet
- `WB-5.3` now closes as `positive`
  - no-go and future gpu-release triggers are explicit
  - current `gpu_release` still remains `none`
- `WB-5` current lane closes as `positive but bounded`
  - the white-box successor candidate is now more honest and more execution-disciplined
  - but it still remains below release and below admitted upgrade

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-6` DP-LoRA comparator release review

Goal: decide whether the `DP-LoRA` successor lane should now move to immediate GPU validation, or whether the next honest live question is still the frozen `baseline vs SMP-LoRA vs W-1` comparator review

Current read:

- `WB-5` already froze protocol overlap, minimal local candidate, and no-go triggers
- local historical evidence is strong enough to keep the lane alive
- but there is still no fresh comparator verdict against `W-1`

Tasks:

- [x] `WB-6.1` review whether current local evidence is enough to justify keeping the lane alive
- [x] `WB-6.2` review whether any new GPU question exists beyond the historical best point
- [x] `WB-6.3` decide the next live white-box question for this lane

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-comparator-release-review.md`

Selection verdict:

- `WB-6` now closes as `positive`
- the next honest white-box question is still:
  - `baseline vs SMP-LoRA vs W-1 comparator`
- immediate GPU release remains `none`
- current `DP-LoRA` lane stays `positive but bounded`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-7` DP-LoRA comparator contract reconciliation

Goal: reconcile the future `baseline vs SMP-LoRA vs W-1` comparator review against the newly frozen minimal local candidate, so later release review does not drift back to stale optimizer/lr frontier artifacts

Current read:

- `WB-6` already says comparator-first is the next honest question
- but the currently written comparator packet still points at older `T06 batch14 throughput` artifacts
- the frozen local candidate is now the simpler `lambda=0.1 / rank=4 / epochs=10` board

Tasks:

- [x] `WB-7.1` compare the current comparator packet against the frozen local candidate
- [x] `WB-7.2` decide which board should remain canonical for future release review
- [x] `WB-7.3` record whether this changes `gpu_release`

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-comparator-contract-reconciliation.md`

Selection verdict:

- `WB-7` now closes as `positive`
- the current comparator packet is directionally useful but contract-stale
- the only honest future comparator board is now:
  - `baseline vs frozen SMP-LoRA local candidate vs W-1`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-8` DP-LoRA comparator artifact-board preflight

Goal: decide whether the reconciled `baseline vs frozen SMP-LoRA vs W-1` board is already supported by honest local artifacts, or whether a schema/eval-surface mismatch still blocks future release review

Current read:

- `WB-7` fixed the right comparator board
- but a board can still be conceptually right while artifact-incomplete
- the next honest CPU check is therefore artifact-board preflight, not GPU escalation

Tasks:

- [x] `WB-8.1` verify baseline and frozen SMP-LoRA local artifacts
- [x] `WB-8.2` verify current `W-1` reference artifact and metric surface
- [x] `WB-8.3` decide whether the board is already release-review-ready

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-comparator-artifact-board-preflight.md`

Selection verdict:

- `WB-8` now closes as `negative but useful`
- the reconciled board is still blocked by schema/eval-surface mismatch
- `baseline` and frozen `SMP-LoRA` align locally
- `W-1` remains on a different and larger evaluation surface
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-9` DP-LoRA comparator schema-alignment contract

Goal: lock the minimum honest release-review schema for `baseline vs frozen SMP-LoRA vs W-1`, and decide whether one bounded alignment refresh is now mandatory

Current read:

- `WB-8` proved the board is conceptually right but not yet release-review-valid
- the remaining blocker is no longer row identity, but comparator schema
- the next honest CPU task is therefore to lock the release board contract, not to improvise a GPU ask

Tasks:

- [x] `WB-9.1` define the shared release-review fields that all three rows must expose
- [x] `WB-9.2` decide whether the current board is valid only for queue truth or also for release review
- [x] `WB-9.3` decide whether one bounded alignment refresh is now required, and on which side

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-comparator-schema-alignment-contract.md`

Selection verdict:

- `WB-9` now closes as `negative but useful`
- the current board is valid for queue truth only, not for release review
- a release-review board must lock one shared primary metric and one shared evaluation surface
- one bounded alignment refresh is now required
- the cleanest next move is a bounded `W-1` local-surface refresh
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-10` DP-LoRA W-1 local-surface refresh feasibility

Goal: decide whether the newly required bounded `W-1` alignment refresh can already be executed as an evaluation-only refresh on frozen `strong-v3` artifacts, or whether retraining is still required first

Current read:

- `WB-9` already fixed the next alignment side to `W-1`
- the next honest question is now execution feasibility, not schema design
- if this step is positive, the following task can shrink to one bounded execution packet instead of another abstract review

Tasks:

- [x] `WB-10.1` verify whether baseline and frozen SMP-LoRA local outputs point to a reproducible shared asset surface
- [x] `WB-10.2` verify whether frozen `W-1 strong-v3` artifacts already expose the checkpoint and runtime inputs needed for a smaller-surface rerender
- [x] `WB-10.3` decide whether the next step is evaluation-only or still requires retraining

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-w1-local-surface-refresh-feasibility.md`

Selection verdict:

- `WB-10` now closes as `positive`
- the local `baseline / SMP-LoRA` board and frozen `W-1 strong-v3` row already share the same legacy asset family
- the next bounded `W-1` alignment step is evaluation-only, not retraining
- the clean next move is to reuse frozen `strong-v3` checkpoints with `max_samples=63`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-11` DP-LoRA local comparator board refresh verdict

Goal: decide what the first completed same-asset local `baseline vs frozen SMP-LoRA vs W-1` board actually says, now that the bounded `W-1 local63` refresh has been executed

Current read:

- `WB-10` already proved the refresh was executable
- the next honest question is now result interpretation, not feasibility
- this step decides whether the successor lane gained a real local-board win or merely a new artifact

Tasks:

- [x] `WB-11.1` read the completed `W-1 local63` summary
- [x] `WB-11.2` compare it against the frozen `baseline` and `SMP-LoRA` local rows on the shared primary metric
- [x] `WB-11.3` decide what this changes, and what it still does not change

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`

Selection verdict:

- `WB-11` now closes as `positive but bounded`
- the first honest same-asset local comparator board now exists
- on the shared local `AUC` board, frozen `SMP-LoRA` beats refreshed `W-1`, and refreshed `W-1` still beats baseline
- this upgrades successor-lane truth, but does not change admitted white-box claims
- `gpu_release = none`

Value: ⭐⭐⭐⭐
Budget: one bounded GPU evaluation completed

#### ⬜ `WB-12` DP-LoRA comparator release-review refresh

Goal: refresh the old `WB-6` release-review logic around the completed same-asset local comparator board, and decide whether the new local-board win changes the current GPU release gate

Current read:

- `WB-11` already created the first honest same-asset local comparator board
- the next honest question is now release-review interpretation, not comparator existence
- this task decides whether the successor lane is still merely `bridge-positive`, or has upgraded to a stronger bounded status

Tasks:

- [x] `WB-12.1` compare the old `WB-6` release-review assumptions against the new local-board evidence
- [x] `WB-12.2` decide what the completed local board upgrades, and what it still does not upgrade
- [x] `WB-12.3` decide whether the current GPU release gate changes

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-comparator-release-review-refresh.md`

Selection verdict:

- `WB-12` now closes as `positive but bounded`
- the successor lane is no longer just `bridge-positive`; it is now `local-comparator positive`
- the new local-board win strengthens white-box queue truth
- but it still does not justify immediate new GPU release
- the clean next move is CPU-side refresh of the stale comparator admission packet
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-13` DP-LoRA comparator admission packet refresh

Goal: rewrite the stale `baseline vs SMP-LoRA vs W-1` admission packet so it matches the completed local comparator board rather than the old pre-board framing

Current read:

- `WB-12` already refreshed the release-review logic
- but the intake packet itself still points at stale comparator framing and stale stop conditions
- the next honest CPU step is therefore to refresh the packet object, not to invent a new experiment

Tasks:

- [x] `WB-13.1` identify which parts of the old admission packet are now stale
- [x] `WB-13.2` rewrite the packet around the completed local board
- [x] `WB-13.3` freeze whether the refreshed packet releases any new GPU question

Canonical evidence anchor:

- `workspaces/intake/2026-04-16-dplora-comparator-admission-packet-refresh.md`

Selection verdict:

- `WB-13` now closes as `positive`
- the old packet is now explicitly stale
- the refreshed packet now centers the completed local comparator board
- it also freezes `gpu_release = none`
- no new GPU question is released by the packet itself

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-14` DP-LoRA next-question review

Goal: decide whether the lane still contains a real bounded next question after the completed local comparator board and packet refresh, or whether it should now explicitly enter `no-new-gpu-question` hold

Current read:

- `WB-13` has already frozen current packet truth
- the next honest move is no longer to repeat the local board
- the only remaining candidate question is whether the local board should be harmonized on defended-style secondary metrics

Tasks:

- [x] `WB-14.1` identify which earlier questions are already fully answered
- [x] `WB-14.2` decide whether any bounded question still remains
- [x] `WB-14.3` decide whether that remaining question is GPU or CPU-first

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-next-question-review.md`

Selection verdict:

- `WB-14` now closes as `positive but narrow`
- there is still one bounded next question
- that question is `local-board secondary-metric harmonization`
- it is CPU-first evaluation-layer hardening, not a new GPU question
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-15` DP-LoRA secondary-metric harmonization audit

Goal: decide whether the remaining `local-board secondary-metric harmonization` question can be answered safely from current frozen artifacts, or whether the evaluator must be hardened first

Current read:

- `WB-14` already selected harmonization as the only remaining bounded question
- the current evaluator can likely produce richer metrics
- but the frozen local outputs may not preserve enough information to upgrade the board without drift

Tasks:

- [x] `WB-15.1` audit what the current local evaluator already computes internally
- [x] `WB-15.2` audit what the frozen local outputs do and do not preserve
- [x] `WB-15.3` decide whether harmonization is artifact-safe today

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-secondary-metric-harmonization-audit.md`

Selection verdict:

- `WB-15` now closes as `negative but useful`
- post-hoc harmonization is not artifact-safe today
- the blocker is missing score artifacts plus unseeded evaluation splitting
- the next honest move is evaluator hardening, not direct rerender
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-16` DP-LoRA local evaluator hardening

Goal: remove the evaluator-side blocker identified by `WB-15`, so any future local-board rerender can become deterministic, auditable, and capable of emitting defended-style secondary metrics

Current read:

- `WB-15` already showed that direct post-hoc harmonization is unsafe
- the blocker is no longer conceptual
- the next honest move is to harden the local evaluator itself before deciding whether a rerender is worthwhile

Tasks:

- [x] `WB-16.1` make evaluation split control explicit and deterministic
- [x] `WB-16.2` add defended-style secondary metrics to the evaluator output
- [x] `WB-16.3` add a way to persist score/probability artifacts for later audit

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-local-evaluator-hardening.md`

Selection verdict:

- `WB-16` now closes as `positive`
- the evaluator-side blocker from `WB-15` is removed
- future local rerenders can now be deterministic and audit-friendlier
- the local board itself is not yet rerendered
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `WB-17` DP-LoRA harmonized local board verdict

Goal: decide what the first hardened local comparator board actually says, now that `baseline` and frozen `SMP-LoRA` have been rerendered under the deterministic metric-harmonized evaluator

Current read:

- `WB-16` removed the evaluator-side blocker
- the next honest step was one harmonized rerender of the local `baseline / SMP-LoRA` rows
- this task decides whether the old `WB-11` ordering survives under the hardened evaluator

Tasks:

- [x] `WB-17.1` read the new harmonized `baseline` and frozen `SMP-LoRA` local outputs
- [x] `WB-17.2` compare them against `W-1 local63` on all shared available metrics
- [x] `WB-17.3` decide what old local-board claims survive and what must now be superseded

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-harmonized-local-board-verdict.md`

Selection verdict:

- `WB-17` now closes as `mixed but useful`
- frozen `SMP-LoRA` still beats `W-1` on local `AUC` and `ASR`
- but the old `WB-11` one-line ordering does not survive the hardened evaluator
- the harmonized local board is now a metric-split board, not a clean dominance story
- `gpu_release = none`

Value: ⭐⭐⭐⭐
Budget: one bounded GPU rerender pair

#### ⬜ `WB-18` DP-LoRA post-harmonized lane-status review

Goal: decide the honest current status of the `DP-LoRA` successor lane after the harmonized local board, and freeze whether it currently contains any new GPU-worthy question

Current read:

- `WB-17` already replaced the old clean local-win story with a `metric-split local board`
- the next honest question is now lane status, not another rerender
- this task decides whether `DP-LoRA` remains an active GPU branch or should explicitly enter `no-new-gpu-question`

Tasks:

- [x] `WB-18.1` decide which parts of the old local-win story still survive
- [x] `WB-18.2` decide whether the lane still contains any new GPU-worthy question
- [x] `WB-18.3` sync the white-box lane plan to the new truth

Canonical evidence anchor:

- `workspaces/white-box/2026-04-16-dplora-post-harmonized-lane-status-review.md`

Selection verdict:

- `WB-18` now closes as `mixed but stabilizing`
- the successor lane remains alive as a bounded exploration branch
- but the clean local-win story is gone
- the lane now explicitly enters `no-new-gpu-question`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

---

### 6.5 Infrastructure, automation, and agent leverage

#### ✅ `INF-1` CLIP/BLIP loading fixes

- status: completed
- keep documented and reusable

#### ⬜ `INF-2` Research automation health

Goal: make the repository easier for a long-running autonomous agent to operate

Tasks:

- [x] `INF-2.1` identify friction points in current run/update workflow
- [x] `INF-2.2` add bounded automation where it reduces repeated human babysitting
- [x] `INF-2.3` improve run artifact consistency or summary templates if needed

Status:

- first automation-health round completed
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-16-research-automation-health-verdict.md`
  - `workspaces/implementation/2026-04-16-run-anchor-consistency-verdict.md`
  - `workspaces/implementation/artifacts/research-automation-health-20260416.json`

Verdict:

- the new automation round is `positive`, but audited repo health is still `friction detected`
- current autonomous workflow had one repeated-friction gap:
  - priority ladder, next GPU candidate, GPU-idle reason, and run-summary anchor hygiene were discoverable only by repeated manual scanning
- bounded automation now exists to audit exactly those signals in one pass
- `INF-2.3` now also completes positively:
  - template placeholders are no longer conflated with active broken references
  - remaining friction is now reported more honestly as active ignored/untracked/missing anchor issues
- later hardening from `INF-3.2` also closed two review-found gaps:
  - active untracked anchors now count as actionable friction
  - git path checks no longer assume Windows-only separator normalization

Carry-forward rule:

- use the automation-health audit as a CPU-side preflight, not as a substitute for research judgment

Value: ⭐⭐⭐

#### ✅ `INF-3` Subagent leverage experiments

Goal: determine when optional subagents actually improve research throughput

Tasks:

- [x] `INF-3.1` test paper-scout subagent workflow
- [x] `INF-3.2` test code-review subagent workflow
- [x] `INF-3.3` test backlog-critic or experiment-auditor workflow
- [x] `INF-3.4` record what should become standard and what should stay optional

Status:

- three bounded read-only subagent experiments completed
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-16-subagent-paper-scout-verdict.md`
  - `workspaces/implementation/2026-04-16-subagent-code-review-verdict.md`
  - `workspaces/implementation/2026-04-16-subagent-backlog-critic-verdict.md`
  - `workspaces/implementation/2026-04-16-subagent-standardization-verdict.md`

Verdict:

- the read-only paper-scout workflow is also `positive`
- it identified `SimA cpu-32 timestep rescan` as the shortest credible next gray-box candidate-generation branch, while explicitly rejecting a premature jump to `MoFit / noise-as-a-probe / SIDe`
- the read-only code-review workflow is also `positive`
- it surfaced two real hardening gaps in the automation-health patch:
  - active untracked anchors were not yet counted as friction
  - git path classification still assumed Windows-style separator rewriting
- the read-only backlog-critic workflow is `positive` in the current repo state
- it created real leverage precisely because many recent box-local branches had just closed
- it independently selected `X-4.1 cross-box agreement analysis` as the highest-value next live task
- `INF-3.4` now also closes as `positive`
- current standardization rule:
  - use paper-scout when multiple literature candidates exist but local implementation distance differs sharply
  - use code-review subagents for automation/tooling commits that affect audit truth or canonical evidence generation
  - use backlog-critic after multiple closures or when the priority ladder may have drifted
  - keep it optional for trivial or obviously single-path loops

Value: ⭐⭐

---

## 7. Near-Term Priority Ladder

This is a preference order, not a prison.

### Top now

`WB-18` DP-LoRA post-harmonized lane-status review is now closed.

No immediate GPU lane should be opened until a new bounded comparison or defended-extension question is selected.

Current release posture:

- `gpu_release = none`
- `next_gpu_candidate = none`

### Next

1. ✅ `WB-18` DP-LoRA post-harmonized lane-status review
2. ✅ `WB-17` DP-LoRA harmonized local board verdict
3. ✅ `WB-16` DP-LoRA local evaluator hardening
4. ✅ `WB-15` DP-LoRA secondary-metric harmonization audit
5. ✅ `WB-14` DP-LoRA next-question review
6. ✅ `WB-13` DP-LoRA comparator admission packet refresh
7. ✅ `WB-12` DP-LoRA comparator release-review refresh
8. ✅ `WB-11` DP-LoRA local comparator board refresh verdict
9. ✅ `WB-10` DP-LoRA W-1 local-surface refresh feasibility
10. ✅ `WB-9` DP-LoRA comparator schema-alignment contract
11. ✅ `WB-8` DP-LoRA comparator artifact-board preflight
12. ✅ `WB-7` DP-LoRA comparator contract reconciliation
13. ✅ `WB-6` DP-LoRA comparator release review
14. ✅ `GB-17` Noise-as-a-Probe defended-extension feasibility review
15. ✅ `GB-16` Noise-as-a-Probe summary-layer sync
16. ✅ `GB-15` Noise-as-a-Probe challenger-boundary review
17. ✅ `GB-14` Noise-as-a-Probe larger-rung repeat
18. ✅ `GB-13` Noise-as-a-Probe larger bounded rung
19. ✅ `GB-12` Noise-as-a-Probe threshold hardening
20. ✅ `GB-11` Noise-as-a-Probe expansion repeat
21. ✅ `GB-10` Noise-as-a-Probe first expansion rung
22. ✅ `GB-9` Noise-as-a-Probe calibration / expansion policy
23. ✅ `GB-8` Noise-as-a-Probe canary scaffold
24. ✅ `GB-7` Noise-as-a-Probe implementation-surface review
25. ✅ `GB-6` Noise-as-a-Probe protocol / asset contract
26. ✅ `GB-5` genuinely-new-family selector
27. ✅ `WB-5` DP-LoRA comparability dossier
28. ✅ `BB-6` same-protocol cross-method score package
29. ✅ `WB-3` white-box defense breadth
30. ✅ `GB-1` second gray-box defense
31. ✅ `BB-1` second-signal black-box expansion
32. ✅ `INF-2` research automation health
33. ✅ `INF-3` subagent leverage experiments
34. ✅ `WB-4` white-box feature/trajectory upgrade
35. ✅ `X-3` system-consumable sync
36. ✅ `BB-3` CLiD boundary-quality upgrade
37. ✅ `X-4` cross-box exploration lane

### Then

16. ✅ `WB-2` second white-box verdict
17. ✅ `GB-3` new gray-box family
18. ✅ `BB-4` mitigation-aware black-box evaluation

---

## 8. Success Conditions

### Near-term health

- at least one more high-value verdict lands
- GPU is not idle without a stated reason
- GPU is not wasted on low-value repeats
- latest outcomes remain consumable by higher layers

### Long-term health

- each threat model has at least two meaningfully different lines
- each threat model has more than one defense story or explicit reason why not
- blocker-heavy areas are either unblocked or honestly frozen
- roadmap keeps expanding rather than stalling

### Never-call-it-done rule

This roadmap is not "done" when the current boxes are checked.

It is only in a temporary resting state if:

- the current backlog has no high-value unchecked work;
- the challenger queue has been mined;
- blockers are honestly documented;
- and no obvious next branch is available.

If that happens, the agent must add new branches and continue.

---

## 9. Changelog

| Date | Change |
|------|--------|
| 2026-04-15 19:30 | Archived earlier P0-P3 roadmap and opened long-horizon mainline phase |
| 2026-04-15 22:30 | Added root alignment and broad exploration rules |
| 2026-04-15 23:10 | Reorganized roadmap into continuous autonomous research system with self-cycling loop, optional subagent policy, and long-running backlog expansion rules |
| 2026-04-15 23:55 | Closed `GB-2` with the `PIA vs SecMI` disagreement verdict; promoted `SecMI` to corroboration line and rejected naive gray-box fusion |
| 2026-04-16 00:15 | Closed the current `BB-1` branch as a positive `semantic-auxiliary-classifier` challenger and kept `Recon` as the black-box headline |
| 2026-04-16 00:45 | Re-ran the semantic auxiliary challenger at `32 / 32`; metrics stayed stable and the challenger remained promotion-worthy |
| 2026-04-16 01:05 | Rejected `GB-1` bounded candidate `epsilon-output-noise (std=0.1)` on `cpu-32`; it did not beat baseline and should not be released to GPU |
| 2026-04-16 01:20 | Rejected `GB-1` bounded candidate `input-gaussian-blur (sigma=1.0)` on `cpu-32`; it strengthened the attack and should not be released to GPU |
| 2026-04-16 01:30 | Reviewed the gray-box second-defense shortlist: cheap perturbation candidates are exhausted for now, so the next active gray-box step should pivot to `G-2` unblock/design or `GB-3` new-family exploration |
| 2026-04-16 01:35 | Closed `WB-1` positively: direct GSA gradient extraction works, and the blocker reduced to dataset-mode mismatch plus missing output-directory hygiene |
| 2026-04-16 01:40 | Selected `SimA` as the next gray-box family (`GB-3`), with a bounded CPU feasibility path on the current CIFAR-10 DDPM asset line |
| 2026-04-16 02:00 | Closed the current `GB-3 / SimA` branch as `negative but useful`: local CPU feasibility is real, but strength is too weak for challenger promotion or GPU release |
| 2026-04-16 02:10 | Reselected the next gray-box family branch onto `TMIA-DM protocol / asset decomposition`, preferring a DDPM-local time/noise path over immediate `SimA` reopen or a larger `MoFit` jump |
| 2026-04-16 02:20 | Upgraded `TMIA-DM` from generic intake to `protocol-ready but not execution-released`: local signal surface, access assumption, asset fit, and minimal smoke entry are now explicit |
| 2026-04-16 03:05 | Ran the first bounded `TMIA-DM` CPU-32 protocol probe: `long_window` is locally positive (`AUC = 0.702148`) while `short_window` is negative and naive fusion is not useful, so the family is promising but still not GPU-released |
| 2026-04-16 03:20 | Repeated the bounded `TMIA-DM long_window` probe at `cpu-32` with `seed1`: the branch stayed positive (`AUC = 0.663086`) but softened, so it is repeat-positive yet still below GPU release threshold |
| 2026-04-16 03:35 | Compared `PIA` against `TMIA-DM long_window` on the same local `cpu-32` split: `PIA` stayed clearly ahead, so `TMIA-DM` remains a secondary refinement/corroboration branch rather than the gray-box headline |
| 2026-04-16 03:50 | Refined `TMIA-DM long_window` onto late timesteps `[80,100,120]`: two bounded `cpu-32` runs landed at `AUC = 0.823242` and `0.760742`, upgrading the branch from CPU-only refinement to GPU-eligible challenger candidate |
| 2026-04-16 04:05 | Attempted the first `TMIA-DM late-window GPU128` pilot and hit an execution blocker: the current `Research` Python runtime is `torch 2.11.0+cpu`, so `cuda:0` is unavailable locally despite the machine having a visible NVIDIA GPU |
| 2026-04-16 04:20 | Resolved the local `Research` CUDA blocker by switching back to `conda` env `diffaudit-research` and repairing the editable install; the first `TMIA-DM late-window GPU128` rung then landed positive at `AUC = 0.825317`, strong enough to keep the challenger line active |
| 2026-04-16 04:30 | Repeated `TMIA-DM late-window GPU128` with `seed1`; the line stayed strong at `AUC = 0.836975`, upgrading it from a first positive GPU rung to a repeat-confirmed gray-box challenger |
| 2026-04-16 04:40 | Scaled `TMIA-DM late-window` to `GPU256`: the line stayed strong at `AUC = 0.839554 / ASR = 0.765625`, reaching near-parity with `PIA GPU256` while improving low-FPR detection |
| 2026-04-16 04:50 | Repeated `TMIA-DM late-window GPU256` with `seed1`; the line held at `AUC = 0.837814 / ASR = 0.787109`, confirming that the challenger remains stable at the higher rung |
| 2026-04-16 05:00 | Wrote the gray-box operating-point comparison: `PIA` remains the safest headline on global metrics, while `TMIA-DM late-window` has become the strongest low-FPR challenger line |
| 2026-04-16 05:15 | Ran the first `TMIA-DM late-window` defense interaction on `GPU128`: `stochastic-dropout(all_steps)` weakened the line but did not neutralize it, leaving `TMIA-DM` stronger than current defended `PIA` on `AUC` and `TPR@1%FPR` |
| 2026-04-16 05:25 | Repeated the defended `TMIA-DM late-window GPU128` rung with `seed1`; the line held at `AUC = 0.819397`, confirming that the current dropout defense weakens but does not eliminate the challenger |
| 2026-04-16 05:35 | Wrote the defended gray-box operating-point comparison: `PIA` stays the defended headline by continuity, but `TMIA-DM late-window` remains the strongest defended challenger rather than being neutralized by dropout |
| 2026-04-16 05:45 | Scaled defended `TMIA-DM late-window` to `GPU256`: the line stayed positive at `AUC = 0.825867`, slightly trailing defended `PIA` on headline metrics but remaining much stronger on low-FPR behavior |
| 2026-04-16 05:55 | Repeated defended `TMIA-DM late-window GPU256` with `seed1`; the line held at `AUC = 0.82164 / ASR = 0.765625`, confirming that the defended challenger also remains stable at the higher rung |
| 2026-04-16 06:00 | Synchronized the unified attack-defense artifacts to the new gray-box reality: `PIA` remains headline, `TMIA-DM late-window` is now the strongest challenger, and defended gray-box stays multi-family |
| 2026-04-16 06:10 | Tested a `late_steps_only` dropout ablation targeted at `TMIA-DM late-window`; it was weaker than `all_steps`, so the defended headline stays on `all_steps` and the new result is recorded as a narrow ablation only |
| 2026-04-16 06:20 | Tested a `timestep-jitter(radius=10)` defense targeted at `TMIA-DM late-window`; it increased the challenger signal (`AUC = 0.850098`), so the hypothesis is rejected as a counterproductive defense |
| 2026-04-16 06:35 | Added `TMIA-DM temporal-striding(stride=2)` as a new defense family, verified it with a failing test first, and got two repeat-positive `cpu-32` rungs (`AUC = 0.697266 / 0.696289`), so the next gate is one minimal `GPU128` rung rather than another wide defense search |
| 2026-04-16 06:55 | Scaled `TMIA-DM temporal-striding(stride=2)` to `GPU128` and repeated it with `seed1`; the defense held at `AUC = 0.727234 / 0.711609`, making it the strongest TMIA-specific defended candidate and moving the next gate to one `GPU256` rung |
| 2026-04-16 07:10 | Scaled `TMIA-DM temporal-striding(stride=2)` to `GPU256` and repeated it with `seed1`; the defense held at `AUC = 0.733322 / 0.7173`, so the branch is now repeat-confirmed through scale and should move to defended operating-point comparison plus system-sync review |
| 2026-04-16 07:20 | Completed the defended comparison and system-layer sync: `TMIA + temporal-striding(stride=2)` now supersedes `TMIA + dropout` as the strongest defended gray-box challenger in comparison artifacts and the unified attack-defense table |
| 2026-04-16 07:30 | Tightened the `CLiD` black-box boundary from generic local bridge wording to `evaluator-near local clip-only corroboration`: current target-side outputs are `100 x 5` numeric matrices after header-skip, but full `cal_clid_th.py` alignment remains blocked on missing shadow-side files and cache-root leakage in the executed rung header |
| 2026-04-16 07:40 | Added a reusable zero-GPU `CLiD` threshold-compatibility audit tool and ran it on the current target100 rung; the result is now machine-readable (`target_pair.ready=true`, `shadow_pair.ready=false`) and closes the current `BB-3` boundary-tightening step without pretending to have a paper-aligned benchmark |
| 2026-04-16 14:45 | Closed `X-6` positively: `phase-e-candidates.json` had become stale again after recent lane-status closures, so the registry is now refreshed into a sparse-hold posture; `DP-LoRA` is removed because it is no longer intake-only, `SecMI unblock` is removed because it no longer matches repo truth, and `Finding NeMo` remains the only intake-only candidate while `PIA paper-aligned confirmation` stays document-layer conditional only |
| 2026-04-16 14:25 | Closed `BB-7` as `negative but stabilizing`: after the second-signal challenger, scoring review, `CLiD` boundary tightening, mitigation no-go, and `variation` asset-contract clarification, black-box currently has no honest new GPU-worthy question; keep `Recon` as headline, `semantic-auxiliary-classifier` as leading challenger, `CLiD` as corroboration-only, and `variation` as contract-ready blocked until a genuinely new feature family or real asset change appears |
| 2026-04-16 08:05 | Refreshed the `Phase E` candidate registry after recent lane promotions and selected `WB-5 DP-LoRA comparability dossier` as the next live CPU-first lane; `Finding NeMo` remains `zero-GPU hold`, `TMIA-DM` is removed from intake-only candidate ordering, and `gpu_release` stays `none` |
| 2026-04-16 08:20 | Closed `WB-5.1` as `positive but bounded`: `DP-LoRA` has real white-box defense-family overlap and a local `SMP-LoRA under DDPM/CIFAR10` bridge hint, but the current relation to admitted `GSA/W-1` remains `partial-overlap only`, so `gpu_release` still stays `none` and the next gate is the minimal local config candidate |
| 2026-04-16 08:35 | Closed `WB-5.2` as `positive but bounded`: the minimal local translation candidate is now frozen to `lambda=0.1 / rank=4 / epochs=10` on the local `DDPM/CIFAR10 + GSA` bridge, while `gpu_release` still stays `none` and the next gate becomes explicit no-go / future gpu-release triggers |
| 2026-04-16 08:45 | Closed `WB-5.3` and the current `WB-5` lane as `positive but bounded`: `DP-LoRA` now has explicit no-go and future GPU-release triggers, so the successor lane is governance-ready but still below release and below admitted upgrade |
| 2026-04-16 08:55 | Re-opened the next live CPU-first lane as `GB-5 genuinely-new-family selector`: current gray-box packaging is strong enough for now, black-box remains in `no immediate rerun`, and the next research value lies in selecting one truly new gray-box family plus its first bounded smoke |
| 2026-04-16 09:10 | Closed `GB-5` positively: selected `Noise as a Probe` as the next genuinely new gray-box family, rejected `MoFit / SIDe / SimA reopen / structural memorization reopen` for now, and opened `GB-6` as the CPU-first protocol / asset contract lane with `gpu_release = none` |
| 2026-04-16 09:20 | Closed `GB-6` as `positive but bounded`: the first honest local `Noise as a Probe` target family is `SD1.5 + celeba_partial_target/checkpoint-25000`, the first bounded smoke is a one-member/one-non-member interface canary, and `gpu_release` remains `none` until prompt source, custom-noise path, and canary schema are frozen |
| 2026-04-16 09:35 | Closed `GB-7` as `positive but bounded`: the repo already has target-side SD1.5+LoRA loading, latent-diffusion stepping, and distance scoring pieces, but still lacks reusable `DDIM inversion + custom-noise target-generation` glue; the next live lane is now `GB-8 Noise-as-a-Probe canary scaffold` |
| 2026-04-16 09:45 | Closed `GB-8.1` positively: the first `Noise as a Probe` canary should use a dedicated script rather than overloading the current semantic-aux or structural-memorization scripts; helper reuse is allowed, but the surface itself should stay family-specific |
| 2026-04-16 10:00 | Closed `GB-8.2/GB-8.3` as `positive`: added a dedicated `Noise as a Probe` interface-canary scaffold script with prompt-source freeze, latent inversion, custom-noise target generation, and explicit canary artifact schema; the lane remains below release and `gpu_release` stays `none` until real runtime verification lands |
| 2026-04-16 10:15 | Closed the first `Noise as a Probe` interface canary as `positive but bounded`: the end-to-end local path now runs on `SD1.5 + celeba_partial_target/checkpoint-25000` for `1 member + 1 non-member`, but the next live lane shifts to calibration / expansion policy rather than premature benchmark or release claims |
| 2026-04-16 10:25 | Closed `GB-9` positively: fixed the smallest honest `Noise as a Probe` calibration / expansion policy to `8` prior non-members for calibration and a first bounded rung of `8 members + 8 eval non-members + 8 calibration non-members`, while keeping `gpu_release = none` |
| 2026-04-16 10:40 | Closed `GB-10` as `positive but bounded`: the first `8 / 8 / 8` `Noise as a Probe` expansion rung kept member-vs-nonmember mean `MSE` separation alive and yielded a simple calibration threshold with `accuracy = 0.75 / TPR = 0.75 / FPR = 0.25`, but the branch still remains below any release-grade claim |
| 2026-04-16 10:55 | Closed `GB-11` as `positive but bounded`: a disjoint repeat of the `8 / 8 / 8` `Noise as a Probe` rung remained same-directional with member mean `MSE` still below non-member mean `MSE`, so the branch is now `repeat-positive` rather than a one-split fluke, while `gpu_release` still remains `none` |
| 2026-04-16 11:15 | Closed `GB-12` as `positive but bounded`: the implemented calibration-only `15th percentile` rule remains locally coherent across `r1/r2`, a conservative frozen threshold band now sits at `1304.8905 .. 1308.7131`, but release-grade thresholding is still `no-go` and the next gate should be one larger bounded rung rather than promotion |
| 2026-04-16 11:35 | Closed `GB-13` positively: a disjoint `16 / 16 / 16` `Noise as a Probe` rung landed at `accuracy = 0.90625 / TPR = 0.8125 / FPR = 0.0`, and the frozen `GB-12` thresholds transfer cleanly onto the larger split, so the branch is materially strengthened but still needs one same-scale repeat before any promotion |
| 2026-04-16 11:55 | Closed `GB-14` positively: the disjoint same-scale `16 / 16 / 16` repeat also stayed strong, and the frozen `GB-12/GB-13` threshold story (`1308.7131` / `1326.5686`) remained cleaner than the new `r4` self-threshold, so `Noise as a Probe` is now a strengthened bounded challenger candidate and the next gate should move to CPU-side challenger-boundary review |
| 2026-04-16 12:10 | Closed `GB-15` positively: `Noise as a Probe` is now strong enough to count as a strengthened bounded challenger candidate, but it still should not replace `TMIA-DM` as the strongest packaged gray-box challenger because it lacks same-protocol comparability, defended evidence, and higher-layer operating-point comparison |
| 2026-04-16 12:20 | Closed `GB-16` positively: the higher-layer gray-box summary is now synchronized to current truth, replacing stale `SecMI blocked / TMIA intake-only` wording with `SecMI corroboration / TMIA packaged challenger / Noise as a Probe strengthened bounded challenger candidate` while preserving `PIA` as the headline |
| 2026-04-16 12:35 | Closed `GB-17` as `negative but useful`: `Noise as a Probe` still has no honest minimal defended-extension gate on the current local `SD1.5` contract, because direct `stochastic-dropout` port is not real on a U-Net whose dropout modules all sit at `p=0.0`, and direct `temporal-striding` would redefine the attack rather than add a bounded defense |
| 2026-04-16 14:05 | Closed `GB-18` as `negative but stabilizing`: after the `TMIA + temporal-striding` challenger closure and the `Noise as a Probe` boundary review, gray-box currently has no honest new GPU-worthy question; keep `PIA` as headline, `TMIA + temporal-striding` as defended challenger reference, `Noise as a Probe` as strengthened bounded challenger candidate, and require a genuinely new mechanism or contract shift before reopening GPU |
| 2026-04-16 12:50 | Closed `WB-6` positively: the `DP-LoRA` successor lane should stay alive, but the next honest white-box question is still the frozen `baseline vs SMP-LoRA vs W-1` comparator review rather than immediate GPU validation, because current local SMP-LoRA evidence is bridge-positive yet still lacks a fresh comparator verdict against `W-1` |
| 2026-04-16 13:00 | Closed `WB-7` positively: the existing comparator packet is still directionally right but contract-stale, because it points at older `T06 batch14 throughput` artifacts instead of the newly frozen `lambda=0.1 / rank=4 / epochs=10` local candidate; future release review should therefore use only the reconciled `baseline vs frozen SMP-LoRA vs W-1` board |
| 2026-04-16 13:10 | Closed `WB-8` as `negative but useful`: the reconciled `baseline vs frozen SMP-LoRA vs W-1` board is conceptually right but not yet release-review-ready, because baseline/SMP-LoRA live on a local `63 / 63` evaluation surface while `W-1 strong-v3 full-scale` is still reported on a larger `target_eval_size = 2000` surface with a different metric schema |
| 2026-04-16 13:20 | Closed `WB-9` as `negative but useful`: the current `baseline vs frozen SMP-LoRA vs W-1` board is valid only for queue truth, not release review, because a release-review board must lock one shared primary metric and one shared evaluation surface; one bounded `W-1` local-surface refresh is now the cleanest next alignment step |
| 2026-04-16 13:35 | Closed `WB-10` positively: the required `W-1` local-surface alignment step is already operationally feasible as an evaluation-only refresh, because baseline, frozen `SMP-LoRA`, and frozen `W-1 strong-v3` all point to the same legacy `gsa-cifar10-1k-3shadow` asset family and the existing comparator entrypoint already supports checkpoint reuse plus `max_samples` control |
| 2026-04-16 09:03 | Closed `WB-11` as `positive but bounded`: the first honest same-asset local comparator board now exists, and on the shared local `AUC` metric it orders `frozen SMP-LoRA (0.34375) < refreshed W-1 local63 (0.474175) < baseline (0.5565217391304348)`; this upgrades successor-lane truth but still does not change admitted white-box claims |
| 2026-04-16 09:20 | Closed `WB-12` as `positive but bounded`: the old `WB-6` release-review gate is now superseded by a stronger bounded reading, because the successor lane is no longer merely `bridge-positive` but now has one completed same-asset local comparator win over refreshed `W-1`; this strengthens white-box queue truth but still does not justify immediate new GPU release |
| 2026-04-16 09:30 | Closed `WB-13` positively: the stale `baseline vs SMP-LoRA vs W-1` admission packet has now been refreshed around the completed local comparator board, replacing old `batch14 throughput` framing and pre-board stop conditions with the current bounded truth and explicitly freezing `gpu_release = none` |
| 2026-04-16 09:45 | Closed `WB-14` as `positive but narrow`: the lane still contains one bounded next question, but it is now only `local-board secondary-metric harmonization`; the completed local board does not justify another GPU question, training sweep, or admitted-upgrade ask |
| 2026-04-16 10:00 | Closed `WB-15` as `negative but useful`: the remaining harmonization question is real, but current frozen `baseline / SMP-LoRA` local outputs are not artifact-safe for post-hoc upgrade because the evaluator does not persist score artifacts and still uses an unseeded random split; the next honest move is evaluator hardening before any rerender |
| 2026-04-16 10:10 | Closed `WB-16` positively: the local evaluator now records an explicit evaluation seed, emits defended-style secondary metrics, and supports persisted score/probability artifacts, removing the evaluator-side blocker that made harmonization rerenders unsafe |
| 2026-04-16 10:25 | Closed `WB-17` as `mixed but useful`: under the hardened evaluator, frozen `SMP-LoRA` still beats `W-1` on local `AUC` and `ASR`, but the old `WB-11` one-line local ordering no longer survives because `baseline` now beats frozen `SMP-LoRA` on `AUC`; the harmonized local board is therefore a metric-split board rather than a clean dominance story |
| 2026-04-16 10:40 | Closed `WB-18` as `mixed but stabilizing`: after the harmonized local board, the `DP-LoRA` successor lane remains alive only as a bounded exploration branch; the clean local-win story is gone, and the lane now explicitly enters `no-new-gpu-question` until a genuinely new bounded hypothesis appears |
| 2026-04-16 01:55 | Fixed `WB-2` path selection on `GSA2 comparator`; target-side `attack_method=2` canaries succeeded on both member and non-member splits |
| 2026-04-16 02:05 | Extended `WB-2` canary truth onto shadow-side: `shadow-01-member` succeeded under the same direct `GSA2` extraction contract, narrowing the next gate to `shadow-01-nonmember` |
| 2026-04-16 02:12 | Completed the first `WB-2` shadow pair: `shadow-01-nonmember` succeeded, so `WB-2.2` is done and the next gate is a bounded `GSA2` comparator verdict |
| 2026-04-16 02:28 | Closed `WB-2` as `positive secondary line`: bounded `GSA2` comparator completed with `AUC = 0.922498`, strong enough for corroboration but still below admitted `GSA1` mainline |

---

## 10. Archived Roadmaps

- `legacy/2026-04-15-P0-P3-completed-roadmap.md`
- `legacy/2026-04-15-competition-sprint-roadmap-archived.md`
