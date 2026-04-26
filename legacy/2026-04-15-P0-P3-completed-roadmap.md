# DiffAudit Research ROADMAP — P0-P3 Reopened Queue + Long-Horizon Mainlines

> Last updated: 2026-04-15
> Mode: reopened research after frozen competition package
> Rule: one GPU task at a time, every task must produce a concrete verdict

---

## 0. How To Use This Roadmap

- Checked items are already done.
- Unchecked items are active backlog.
- Always pick the highest-priority unchecked item unless the user explicitly reprioritizes.
- New experiments must compare against the frozen baseline package instead of silently replacing it.
- When a task finishes, update this file and the relevant `workspaces/<track>/runs/` artifacts immediately.
- This roadmap has two layers:
  - `P0-P3` for immediate execution priority.
  - `Long-horizon mainlines` for continuous exploration after the first-wave reopen queue is absorbed.
- If all current `P0` items are blocked, switch to the highest-value CPU-side preparation task for the next `P1/P2/P3` candidate instead of idling.
- Always maintain:
  - one active GPU run or one clearly justified reason why no GPU run should be active;
  - one queued next GPU candidate with dependencies checked;
  - one CPU-side implementation or analysis task that can proceed while GPU is busy.

Frozen reference package:

- human index: `workspaces/implementation/2026-04-15-final-delivery-index.md`
- machine index: `workspaces/implementation/artifacts/final-delivery-index.json`
- archived competition sprint roadmap:
  - `legacy/2026-04-15-competition-sprint-roadmap-archived.md`
- download master list:
  - `docs/research-download-master-list.md`
- download manifest:
  - `D:\Code\DiffAudit\Download\manifests\research-download-manifest.json`

---

## 0.5 Long-Horizon Program Intent

This roadmap is not only a reopen checklist. It is the long-horizon execution backbone for the research line.

The long-horizon objective is to turn the frozen competition package into a broader and more credible research program with:

1. more than one serious method family per box;
2. at least one meaningful defense story beyond the frozen comparators;
3. at least one paper-faithful upgrade path;
4. at least one genuinely novel or recombined attack idea discovered locally;
5. an execution cadence that keeps CPU/GPU work fed without wasting GPU on low-information reruns.

Working principles:

- Prefer mechanism diversity over same-family count inflation.
- Prefer probe -> verdict -> promote loops over large blind sweeps.
- Prefer honest negative results over inflated stories.
- Prefer maintaining a live challenger queue over repeatedly polishing the current champion.

---

## 1. Frozen Baseline Completed

- [x] Black-box admitted headline exists: `Recon DDIM public-100 step30`, `AUC 0.849`
- [x] Black-box best single-metric rung exists: `Recon public-50 step10`, `AUC 0.866`
- [x] Black-box corroboration exists: `CLiD` local `100 / 100` target-family rungs
- [x] Gray-box runtime mainline exists: `PIA 512 / 512` and `1024 / 1024`
- [x] Gray-box defended comparator exists: stochastic dropout `all_steps`
- [x] Gray-box corroboration exists: `SecMI stat 0.885833`, `NNS 0.946286`
- [x] White-box upper bound exists: `GSA 0.998192`
- [x] White-box defended comparator exists: `DPDM W-1 strong-v3 0.488783`
- [x] Unified evidence table exists
- [x] Attack-defense matrix exists
- [x] Threat-model comparison exists
- [x] Competition answer pack / brief / FAQ / speaker notes exist
- [x] Presentation manifest / checksums / signoff / handoff exist

---

## 2. P0 — Highest-Value Reopen Tasks

### Download and staging

- [x] `P0-DL-1` Verify all first-wave assets from `docs/research-download-master-list.md` are present in `D:\Code\DiffAudit\Download\...`
- [x] `P0-DL-2` Ingest any newly downloaded assets into repo-consumable locations or config pointers
- [x] `P0-DL-3` Record exact asset provenance / path mapping for anything newly staged

### Gray-box defense diversification

Dependencies:

- `SH-DS-01`
- `GB-WT-01` preferred if a SecMI-linked defense route is used

- [x] `P0-GD-1` Choose one gray-box defense mechanism that is materially different from stochastic dropout
- [x] `P0-GD-2` Write a short hypothesis note explaining why this defense might change ranking structure, not just threshold
- [x] `P0-GD-3` Implement a smoke/probe run on existing CIFAR-10 gray-box assets
- [x] `P0-GD-4` If smoke is promising, run a mainline comparator and write `summary.json`
- [x] `P0-GD-5` Update gray-box comparison note and attack-defense matrix with the verdict

### Black-box mitigation line

Dependencies:

- `SH-DS-02`
- `SH-DS-03`
- `SH-WT-01`
- `SH-WT-02`

- [x] `P0-BM-1` Choose one realistic black-box mitigation direction
- [x] `P0-BM-2` Define the black-box threat model and utility constraints before implementation
- [x] `P0-BM-3` Implement a minimal probe against the current `Recon / CLiD` asset stack
- [x] `P0-BM-4` If the mitigation is viable, run a comparator and record attack degradation
- [x] `P0-BM-5` Write a short note explaining whether this is competition-usable or only exploratory

### CLiD paper-faithful upgrade

Dependencies:

- `SH-WT-01`
- `SH-WT-02`
- `BB-SUP-02` if obtainable

- [x] `P0-CL-1` Audit the current local `CLiD` bridge against the paper protocol
- [x] `P0-CL-2` List the minimum missing pieces needed to honestly tighten the boundary
- [x] `P0-CL-3` Execute one paper-alignment upgrade step instead of another same-family target rung
- [x] `P0-CL-4` Decide whether `CLiD` stays “local corroboration” or can be promoted toward “paper-aligned local benchmark”

---

## 3. P1 — New Attack Families

### Gray-box new family

Dependencies:

- relevant `GB-PAP-*` asset for the chosen family

- [x] `P1-GA-1` Pick exactly one of `SIMA / MoFit / Noise-as-a-probe / SIDe / structural memorization`
- [x] `P1-GA-2` Produce a paper-to-code feasibility note
- [x] `P1-GA-3` Implement a minimal prototype
- [x] `P1-GA-4` Run a smoke/probe and record either signal or negative result
- [x] `P1-GA-5` Decide whether the line deserves a mainline GPU run

### Black-box new family

Dependencies:

- `SH-DS-02`
- `SH-WT-01`
- `SH-WT-02`
- `SH-WT-03` if caption-side black-box probing is used

- [x] `P1-BA-1` Pick one black-box direction beyond `Recon + CLiD`
- [x] `P1-BA-2` Define assets, access assumptions, and scoring rule
- [x] `P1-BA-3` Implement a small-sample probe
- [x] `P1-BA-4` If signal exists, run a proper comparator against current black-box baselines

### White-box second line

Dependencies:

- `WB-SUP-01` preferred for `NeMo`
- `WB-PAP-01` if falling through to `Local Mirror`

- [x] `P1-WA-1` Turn `NeMo` from adapter-ready into a real executed verdict
- [x] `P1-WA-2` If `NeMo` blocks, document the blocker precisely and move to `Local Mirror`
- [x] `P1-WA-3` Produce one real “2nd white-box line” verdict, positive or negative

---

## 4. P2 — Deepening Existing Lines

### PIA / SecMI deepening

- [x] `P2-GS-1` Extend `SecMI` beyond corroboration by testing defended or disagreement-analysis variants — SecMI stat vs PIA disagreement analysis on 1024/1024: Spearman 0.908, 12.3% disagreement, simple ensemble AUC 0.869 (midpoint, not synergistic); verdict: negative for naive fusion
- [ ] `P2-GS-2` Reopen `PIA` only if the new variable is genuinely different from scale-only reruns
- [ ] `P2-GS-3` If reopened, document the hypothesis before the run

### GSA deepening

- [ ] `P2-WG-1` Test one meaningful `GSA` feature upgrade
- [ ] `P2-WG-2` Compare the upgraded feature against current `GSA` baseline
- [ ] `P2-WG-3` Keep or reject the upgrade with a written verdict

### Recon optimization

- [x] `P2-BR-1` Reopen `Recon` only with a concrete hypothesis (`eta`, fusion, loss, or timestep logic) — Hypothesis prepared: MSE + LPIPS + CLIP multi-loss fusion
- [x] `P2-BR-2` Run one bounded probe — MSE probe positive: 0.659 AUC vs 0.576 baseline (+0.083), exceeds significance threshold
- [x] `P2-BR-3` Promote only if it changes the black-box story materially — Verdict: no-promotion; fusion AUC 0.659 below 0.70 threshold, LPIPS/CLIP failed; MSE documented as alternative scoring method

---

## 5. P3 — Cross-Cutting / Stretch

- [x] `P3-X-1` Try one cross-method ensemble that still preserves an honest threat model — Gray-box ensemble (PIA + SecMI stat): no improvement over best single method (0.884); verdict: negative, SecMI dominates
- [x] `P3-X-2` Try one transfer/generalization question across model families or datasets — Verdict: deferred; true transfer experiments require assets not available (no paired models across datasets/architectures); documented limitation
- [x] `P3-X-3` Try one temporal or training-trajectory analysis question — Verdict: deferred; requires 1+ GPU hour for gradient extraction across multiple checkpoints; blocked by same infrastructure issues as P2-WG (GSA gradient extraction failures); high cost + low expected value
- [x] `P3-X-4` Produce at least one judge-friendly visualization from any successful reopened line — 5 visualizations generated: ROC curves, score distributions, disagreement heatmap, agreement scatter, summary table

---

## 5.5 Long-Horizon Model Mainlines

These are persistent mainlines. They do not replace `P0-P3`; they guide what to do next when multiple unchecked tasks are available or when new ideas are discovered.

### Black-box mainline

- [ ] `LH-BB-1` Maintain one strong reconstruction-style line (`Recon` family) and one non-reconstruction line at all times
- [ ] `LH-BB-2` Upgrade `CLiD` or its successor until the boundary is either honestly paper-aligned or clearly documented as local-only
- [ ] `LH-BB-3` Add at least one mitigation-aware black-box evaluation track, not just attacker-side scoring
- [ ] `LH-BB-4` Search for one novel black-box signal source beyond current image-space similarity routines

### Gray-box mainline

- [ ] `LH-GB-1` Maintain at least two materially different gray-box attack families with real verdicts
- [ ] `LH-GB-2` Maintain at least two materially different gray-box defense comparators with real verdicts
- [ ] `LH-GB-3` Perform disagreement analysis between `PIA`, `SecMI`, and any new gray-box family
- [ ] `LH-GB-4` Search for one ranking-sensitive variable beyond threshold tuning or batch scaling

### White-box mainline

- [ ] `LH-WB-1` Turn `GSA` from upper-bound singleton into one line among multiple white-box verdicts
- [ ] `LH-WB-2` Land one second white-box line (`NeMo`, `Local Mirror`, or better)
- [ ] `LH-WB-3` Add one white-box defense or regularization-sensitive comparison that is not merely inherited from frozen DPDM
- [ ] `LH-WB-4` Search for one feature-space or trajectory-space upgrade that could change the white-box ranking story

### Cross-box mainline

- [ ] `LH-X-1` Build one honest cross-box comparison table that can survive new additions without manual rewriting
- [ ] `LH-X-2` Test at least one transfer question across threat models, model families, or datasets
- [ ] `LH-X-3` Keep one live challenger list of ideas that are not yet worth GPU but are worth implementation or paper study

---

## 6. Reopen Guardrails

- [ ] Do not run two GPU jobs at once
- [ ] Do not rerun another same-family `PIA` sweep without a new hypothesis
- [ ] Do not rerun another target-side `CLiD` rung just to increase count
- [ ] Do not spend GPU on cosmetic scale-up that does not change the narrative
- [ ] Do not overwrite frozen baseline package semantics with exploratory results

These are process rules, not deliverables. Keep them checked unless you explicitly violate one and explain why.

---

## 6.5 GPU Utilization Policy

The goal is not maximum raw GPU occupancy. The goal is maximum useful evidence per GPU hour.

- [ ] `GPU-POL-1` Before every GPU run, write the hypothesis and promotion criterion
- [ ] `GPU-POL-2` Keep one queued next GPU candidate so the device does not go idle after a run ends
- [ ] `GPU-POL-3` While GPU is busy, spend CPU time on one of: paper audit, data staging, implementation, analysis, or visualization
- [ ] `GPU-POL-4` Do not start a long GPU run until its input assets, config, and output path are already staged
- [ ] `GPU-POL-5` If a run shows no signal in smoke/probe, kill escalation early instead of consuming the full GPU budget
- [ ] `GPU-POL-6` Prefer bounded comparators and champion-vs-challenger tests over unstructured parameter sweeps

Suggested GPU queue order when several tasks are ready:

1. tasks that can change the box-level verdict story;
2. tasks that can create a new method family verdict;
3. tasks that improve paper-faithful alignment;
4. tasks that improve robustness or defense interpretation;
5. same-family optimizations.

---

## 7. Promotion Criteria

A reopened result is worth promoting only if it changes at least one of:

1. attack strength
2. defense story
3. boundary quality
4. paper-faithful alignment
5. cross-method diversity

If it changes none of the above, log it as:

- `failed`
- `no-go`
- or `negative but useful`

---

## 7.5 Innovation Funnel

Innovation is a required output, not a side effect.

- [ ] `INNO-1` Keep a rolling list of new ideas discovered from failures, paper gaps, or cross-family disagreement
- [ ] `INNO-2` Before implementing a new idea, classify it as one of:
  - new attack surface;
  - new score fusion;
  - new defense mechanism;
  - new calibration / ranking analysis;
  - new transfer / generalization question
- [ ] `INNO-3` Every new idea must first pass a feasibility note or CPU-side prototype before taking GPU time
- [ ] `INNO-4` At least one local idea should be tested even if no paper directly proposes it
- [ ] `INNO-5` Keep negative-but-useful ideas visible instead of deleting them from the queue

High-value innovation templates:

- disagreement-driven scoring between existing methods;
- timestep- or trajectory-selective membership signals;
- feature-space or caption-space black-box probes;
- defense-aware calibration rather than plain attack reruns;
- cross-family ensemble under an honest threat model.

---

## 8. Completion Rule For This Reopened Roadmap

This reopened roadmap is complete only when all of the following are true:

- [ ] At least one materially different new gray-box defense line has a verdict
- [ ] At least one credible black-box mitigation line has a verdict
- [ ] At least one paper-faithful upgrade attempt has a verdict
- [ ] At least one newly implemented attack family beyond the frozen package has a verdict
- [ ] At least one second white-box line (`NeMo` or equivalent) has a real verdict
- [ ] Cross-cutting section has at least one useful result or negative result

---

## 8.5 Continuous Execution Rule

This roadmap should be treated as continuously executable.

Do not stop after the first wave of `P0-P3` items if all of the following are still true:

- there are unchecked `LH-*`, `GPU-POL-*`, or `INNO-*` items;
- there are credible new ideas or challengers not yet tested;
- there are existing lines whose boundary quality or defense story is still obviously weak.

When the current highest-priority task finishes, the next action should be:

1. update artifacts and checkbox state;
2. pick the next highest-value unchecked item;
3. if nothing is immediately GPU-ready, do CPU-side prep for the next GPU-worthy line;
4. if the roadmap structure is no longer sufficient, extend it before drifting into ad hoc work.

---

## 9. Changelog

| Date | Change |
|------|--------|
| 2026-04-15 | Completed `P2-GS-1`: PIA vs SecMI disagreement analysis — high correlation (Spearman 0.908), simple ensemble not synergistic |
| 2026-04-15 | Archived the previous competition-sprint roadmap to `legacy/2026-04-15-competition-sprint-roadmap-archived.md` |
| 2026-04-15 | Replaced the root roadmap with a P0-P3 checkbox-based reopened full-exploration roadmap |
| 2026-04-15 | Added a download master list, machine-readable download manifest, and explicit asset dependencies for reopened tasks |
| 2026-04-15 | Expanded the roadmap into a long-horizon execution program with persistent box mainlines, innovation funnel, and GPU utilization policy |
