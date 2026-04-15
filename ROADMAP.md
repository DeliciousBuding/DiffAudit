# DiffAudit Research ROADMAP — Continuous Autonomous Mainline

> Last updated: 2026-04-16 03:05
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

#### ⬜ `X-3` Keep system-consumable structure in sync

Goal:

- keep admitted/mainline/challenger boundaries readable
- keep comparison artifacts current
- keep higher-layer consumers able to interpret new outcomes

Tasks:

- [ ] `X-3.1` audit whether latest verdicts are reflected in comparison artifacts
- [ ] `X-3.2` audit whether latest verdicts change project narrative
- [ ] `X-3.3` mark any Platform/Runtime-facing field additions or boundary changes

Value: ⭐⭐⭐
Budget: CPU-only

Artifact rule for `X-*`, `INF-*`, `intake`, or other non-run tasks:

- they do not need to force a fake `runs/` directory
- but they must leave one canonical evidence anchor under the appropriate workspace lane
- the main agent must explicitly name that anchor in its update

#### ⬜ `X-4` Cross-box exploration lane

Goal: allow work that spans boxes instead of forcing every idea into one bucket

Candidate directions:

- [ ] `X-4.1` cross-box agreement analysis
- [ ] `X-4.2` score calibration or fusion with bounded hypothesis
- [ ] `X-4.3` transfer or portability probe when assets permit
- [ ] `X-4.4` visualization or decision-quality analysis that changes project understanding

Value: ⭐⭐⭐
Budget: bounded CPU-first, GPU only if justified

---

### 6.2 Black-box expansion

#### ✅ `BB-1` Second-signal black-box expansion

Goal: land a second black-box direction that is not just a mild variant of current image-similarity lines

Tasks:

- [x] `BB-1.1` caption-space probe
- [x] `BB-1.2` feature-space probe
- [ ] `BB-1.3` timestep-selective reconstruction probe
- [ ] `BB-1.4` prompt-response consistency probe

Status:

- completed for the current leading challenger branch
- canonical evidence anchor:
  - `workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`

Verdict:

- the returned-image `semantic-auxiliary-classifier` landed as a real black-box challenger
- bounded local comparator metrics (`AUC = 0.910156`, `ASR = 0.875`) are strong enough for promotion into challenger status
- scaled follow-up run `semantic-aux-classifier-comparator-20260416-r2` stayed stable at `AUC = 0.90918`, confirming the signal survives a `32 / 32` comparator
- this branch adds real method-family diversity, but it does not replace the frozen `Recon` headline

Carry-forward rule:

- keep this line in challenger status
- only escalate with a new bounded hypothesis; do not spend budget on aimless scale-up

Promotion standard:

- must produce a verdict
- if positive, must be stronger than "interesting but noisy"

Value: ⭐⭐⭐

#### ⬜ `BB-2` Scoring and calibration upgrades

Goal: improve black-box signal quality without pretending every scoring tweak is a new method family

Tasks:

- [ ] `BB-2.1` MSE-weighted or multi-score challenger
- [ ] `BB-2.2` bounded fusion experiments
- [ ] `BB-2.3` document whether calibration changes ranking or only threshold

Value: ⭐⭐

#### ⬜ `BB-3` CLiD boundary-quality upgrade

Goal: improve the honesty and strength of the CLiD boundary claim

Tasks:

- [ ] `BB-3.1` review remaining boundary gaps
- [ ] `BB-3.2` perform one minimum honest upgrade
- [ ] `BB-3.3` update verdict wording and evidence note

Value: ⭐⭐

#### ⬜ `BB-4` Mitigation-aware evaluation

Goal: test black-box methods under at least one more realistic mitigation condition

Tasks:

- [ ] `BB-4.1` design protocol
- [ ] `BB-4.2` select mitigation
- [ ] `BB-4.3` evaluate recon + CLiD or new challenger
- [ ] `BB-4.4` record quality-cost-utility trade-off

Value: ⭐⭐

---

### 6.3 Gray-box expansion

#### ⬜ `GB-1` Second gray-box defense

Goal: avoid a single defended story dominating gray-box forever

Current read:

- `stochastic-dropout(all_steps)` remains the only defended gray-box mainline
- `epsilon-precision-throttling` is already a negative bounded candidate
- new bounded candidate `epsilon-output-noise (std = 0.1)` also landed negative on `cpu-32`, so it should not be released to GPU
- `input-gaussian-blur (sigma = 1.0)` landed even more negatively on `cpu-32`, strengthening rather than weakening the attack
- the cheap perturbation-style `GB-1` frontier is effectively exhausted for now
- the next `GB-1` step should be either a `G-2 distillation` unblock/design review or a pivot to `GB-3` new-family exploration

Tasks:

- [x] `GB-1.1` shortlist materially different defense mechanisms
- [x] `GB-1.2` pick one bounded candidate
- [ ] `GB-1.3` test against PIA + SecMI when possible
- [ ] `GB-1.4` record defended verdict

Canonical evidence anchors:

- `workspaces/gray-box/2026-04-15-graybox-defense-precision-throttling-note.md`
- `workspaces/gray-box/2026-04-15-graybox-epsilon-output-noise-defense-verdict.md`
- `workspaces/gray-box/2026-04-15-graybox-input-blur-defense-verdict.md`
- `workspaces/gray-box/2026-04-16-graybox-second-defense-shortlist-review.md`

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
- bounded local `SimA` feasibility is now execution-positive but strength-negative:
  - it runs
  - it does not currently merit challenger promotion or GPU release
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

Tasks:

- [x] `GB-3.1` choose one family
- [x] `GB-3.2` write feasibility note
- [x] `GB-3.3` implement probe or smoke
- [x] `GB-3.4` record verdict

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-graybox-sima-feasibility-verdict.md`
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

Value: ⭐⭐

#### ⬜ `GB-4` Disagreement exploitation

Goal: if a second family lands, determine whether disagreement is useful or redundant

Current read:

- run-level same-split comparison is now explicit:
  - `PIA cpu-32` remains clearly ahead of `TMIA-DM long_window` across both bounded repeats
  - current `TMIA-DM` therefore does not justify role swap or immediate disagreement-exploitation work
- `GB-4` should stay narrow until either `TMIA-DM` strengthens further or a different second family lands
  - current best reading is `secondary corroboration candidate`, not `mainline challenger`

Tasks:

- [x] `GB-4.1` compare on aligned split
- [ ] `GB-4.2` measure correlation and disagreement
- [ ] `GB-4.3` test whether disagreement improves actionability

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-long-window-comparison.md`

Value: ⭐⭐

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

#### ⬜ `WB-3` White-box defense breadth

Goal: broaden white-box defense comparison beyond the current single defended comparator

Tasks:

- [ ] `WB-3.1` survey alternatives beyond DPDM
- [ ] `WB-3.2` pick one bounded defense
- [ ] `WB-3.3` test against GSA
- [ ] `WB-3.4` record attack degradation verdict

Value: ⭐⭐

#### ⬜ `WB-4` White-box feature/trajectory upgrade

Goal: revisit deeper white-box analysis if and only if blocker cost becomes reasonable

Tasks:

- [ ] `WB-4.1` write hypothesis first
- [ ] `WB-4.2` run lightweight probe
- [ ] `WB-4.3` record whether it changes story

Value: ⭐

---

### 6.5 Infrastructure, automation, and agent leverage

#### ✅ `INF-1` CLIP/BLIP loading fixes

- status: completed
- keep documented and reusable

#### ⬜ `INF-2` Research automation health

Goal: make the repository easier for a long-running autonomous agent to operate

Tasks:

- [ ] `INF-2.1` identify friction points in current run/update workflow
- [ ] `INF-2.2` add bounded automation where it reduces repeated human babysitting
- [ ] `INF-2.3` improve run artifact consistency or summary templates if needed

Value: ⭐⭐⭐

#### ⬜ `INF-3` Subagent leverage experiments

Goal: determine when optional subagents actually improve research throughput

Tasks:

- [ ] `INF-3.1` test paper-scout subagent workflow
- [ ] `INF-3.2` test code-review subagent workflow
- [ ] `INF-3.3` test backlog-critic or experiment-auditor workflow
- [ ] `INF-3.4` record what should become standard and what should stay optional

Value: ⭐⭐

---

## 7. Near-Term Priority Ladder

This is a preference order, not a prison.

### Top now

1. ⬜ `GB-1` second gray-box defense
2. ⬜ `X-3` system-consumable sync
3. ⬜ `BB-3` CLiD boundary-quality upgrade
4. ⬜ `X-4` cross-box exploration lane
5. ⬜ `WB-2` second white-box verdict

### Next

6. ⬜ `INF-2` research automation health
7. ⬜ `GB-3` new gray-box family
8. ⬜ `BB-4` mitigation-aware black-box evaluation
9. ⬜ `WB-3` white-box defense breadth

### Then

10. ⬜ `INF-3` subagent leverage experiments

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
| 2026-04-16 01:55 | Fixed `WB-2` path selection on `GSA2 comparator`; target-side `attack_method=2` canaries succeeded on both member and non-member splits |
| 2026-04-16 02:05 | Extended `WB-2` canary truth onto shadow-side: `shadow-01-member` succeeded under the same direct `GSA2` extraction contract, narrowing the next gate to `shadow-01-nonmember` |
| 2026-04-16 02:12 | Completed the first `WB-2` shadow pair: `shadow-01-nonmember` succeeded, so `WB-2.2` is done and the next gate is a bounded `GSA2` comparator verdict |
| 2026-04-16 02:28 | Closed `WB-2` as `positive secondary line`: bounded `GSA2` comparator completed with `AUC = 0.922498`, strong enough for corroboration but still below admitted `GSA1` mainline |

---

## 10. Archived Roadmaps

- `legacy/2026-04-15-P0-P3-completed-roadmap.md`
- `legacy/2026-04-15-competition-sprint-roadmap-archived.md`
