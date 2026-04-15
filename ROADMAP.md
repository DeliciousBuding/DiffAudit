# DiffAudit Research ROADMAP — Continuous Autonomous Mainline

> Last updated: 2026-04-15 23:10
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

#### ⬜ `BB-1` Second-signal black-box expansion

Goal: land a second black-box direction that is not just a mild variant of current image-similarity lines

Tasks:

- [ ] `BB-1.1` caption-space probe
- [ ] `BB-1.2` feature-space probe
- [ ] `BB-1.3` timestep-selective reconstruction probe
- [ ] `BB-1.4` prompt-response consistency probe

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

Tasks:

- [ ] `GB-1.1` shortlist materially different defense mechanisms
- [ ] `GB-1.2` pick one bounded candidate
- [ ] `GB-1.3` test against PIA + SecMI when possible
- [ ] `GB-1.4` record defended verdict

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

#### ⬜ `GB-3` New gray-box family

Goal: test whether another family is worth joining the long-term mainline

Candidate directions:

- SIMA
- MoFit
- noise-as-probe
- SIDe
- another paper-backed gray-box mechanism

Tasks:

- [ ] `GB-3.1` choose one family
- [ ] `GB-3.2` write feasibility note
- [ ] `GB-3.3` implement probe or smoke
- [ ] `GB-3.4` record verdict

Value: ⭐⭐

#### ⬜ `GB-4` Disagreement exploitation

Goal: if a second family lands, determine whether disagreement is useful or redundant

Tasks:

- [ ] `GB-4.1` compare on aligned split
- [ ] `GB-4.2` measure correlation and disagreement
- [ ] `GB-4.3` test whether disagreement improves actionability

Value: ⭐⭐

---

### 6.4 White-box expansion

#### ⬜ `WB-1` Gradient extraction blocker resolution

Goal: decide whether the current white-box blocker is solvable or should become a hard no-go for now

Tasks:

- [ ] `WB-1.1` run gradient extraction directly
- [ ] `WB-1.2` capture root cause
- [ ] `WB-1.3` fix or classify blocker
- [ ] `WB-1.4` verify with one minimal job

Value: ⭐⭐⭐

#### ⬜ `WB-2` Second white-box line verdict

Goal: land or reject a second white-box line with evidence

Possible branches:

- GSA2 comparator
- NeMo if blocker removed
- another non-GSA family

Tasks:

- [ ] `WB-2.1` choose path based on `WB-1`
- [ ] `WB-2.2` implement and test
- [ ] `WB-2.3` record verdict

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

1. ⬜ `WB-1` gradient extraction blocker resolution
2. ⬜ `BB-1` second-signal black-box expansion
3. ⬜ `GB-1` second gray-box defense
4. ⬜ `X-3` system-consumable sync
5. ⬜ `BB-3` CLiD boundary-quality upgrade

### Next

6. ⬜ `X-4` cross-box exploration lane
7. ⬜ `WB-2` second white-box verdict
8. ⬜ `INF-2` research automation health
9. ⬜ `GB-3` new gray-box family

### Then

10. ⬜ `BB-4` mitigation-aware black-box evaluation
11. ⬜ `WB-3` white-box defense breadth
12. ⬜ `INF-3` subagent leverage experiments

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

---

## 10. Archived Roadmaps

- `legacy/2026-04-15-P0-P3-completed-roadmap.md`
- `legacy/2026-04-15-competition-sprint-roadmap-archived.md`
