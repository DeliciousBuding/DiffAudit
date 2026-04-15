# DiffAudit Research ROADMAP — P0-P3 Reopened Queue

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

Frozen reference package:

- human index: `workspaces/implementation/2026-04-15-final-delivery-index.md`
- machine index: `workspaces/implementation/artifacts/final-delivery-index.json`
- archived competition sprint roadmap:
  - `legacy/2026-04-15-competition-sprint-roadmap-archived.md`

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

### Gray-box defense diversification

- [ ] `P0-GD-1` Choose one gray-box defense mechanism that is materially different from stochastic dropout
- [ ] `P0-GD-2` Write a short hypothesis note explaining why this defense might change ranking structure, not just threshold
- [ ] `P0-GD-3` Implement a smoke/probe run on existing CIFAR-10 gray-box assets
- [ ] `P0-GD-4` If smoke is promising, run a mainline comparator and write `summary.json`
- [ ] `P0-GD-5` Update gray-box comparison note and attack-defense matrix with the verdict

### Black-box mitigation line

- [ ] `P0-BM-1` Choose one realistic black-box mitigation direction
- [ ] `P0-BM-2` Define the black-box threat model and utility constraints before implementation
- [ ] `P0-BM-3` Implement a minimal probe against the current `Recon / CLiD` asset stack
- [ ] `P0-BM-4` If the mitigation is viable, run a comparator and record attack degradation
- [ ] `P0-BM-5` Write a short note explaining whether this is competition-usable or only exploratory

### CLiD paper-faithful upgrade

- [ ] `P0-CL-1` Audit the current local `CLiD` bridge against the paper protocol
- [ ] `P0-CL-2` List the minimum missing pieces needed to honestly tighten the boundary
- [ ] `P0-CL-3` Execute one paper-alignment upgrade step instead of another same-family target rung
- [ ] `P0-CL-4` Decide whether `CLiD` stays “local corroboration” or can be promoted toward “paper-aligned local benchmark”

---

## 3. P1 — New Attack Families

### Gray-box new family

- [ ] `P1-GA-1` Pick exactly one of `SIMA / MoFit / Noise-as-a-probe / SIDe / structural memorization`
- [ ] `P1-GA-2` Produce a paper-to-code feasibility note
- [ ] `P1-GA-3` Implement a minimal prototype
- [ ] `P1-GA-4` Run a smoke/probe and record either signal or negative result
- [ ] `P1-GA-5` Decide whether the line deserves a mainline GPU run

### Black-box new family

- [ ] `P1-BA-1` Pick one black-box direction beyond `Recon + CLiD`
- [ ] `P1-BA-2` Define assets, access assumptions, and scoring rule
- [ ] `P1-BA-3` Implement a small-sample probe
- [ ] `P1-BA-4` If signal exists, run a proper comparator against current black-box baselines

### White-box second line

- [ ] `P1-WA-1` Turn `NeMo` from adapter-ready into a real executed verdict
- [ ] `P1-WA-2` If `NeMo` blocks, document the blocker precisely and move to `Local Mirror`
- [ ] `P1-WA-3` Produce one real “2nd white-box line” verdict, positive or negative

---

## 4. P2 — Deepening Existing Lines

### PIA / SecMI deepening

- [ ] `P2-GS-1` Extend `SecMI` beyond corroboration by testing defended or disagreement-analysis variants
- [ ] `P2-GS-2` Reopen `PIA` only if the new variable is genuinely different from scale-only reruns
- [ ] `P2-GS-3` If reopened, document the hypothesis before the run

### GSA deepening

- [ ] `P2-WG-1` Test one meaningful `GSA` feature upgrade
- [ ] `P2-WG-2` Compare the upgraded feature against current `GSA` baseline
- [ ] `P2-WG-3` Keep or reject the upgrade with a written verdict

### Recon optimization

- [ ] `P2-BR-1` Reopen `Recon` only with a concrete hypothesis (`eta`, fusion, loss, or timestep logic)
- [ ] `P2-BR-2` Run one bounded probe
- [ ] `P2-BR-3` Promote only if it changes the black-box story materially

---

## 5. P3 — Cross-Cutting / Stretch

- [ ] `P3-X-1` Try one cross-method ensemble that still preserves an honest threat model
- [ ] `P3-X-2` Try one transfer/generalization question across model families or datasets
- [ ] `P3-X-3` Try one temporal or training-trajectory analysis question
- [ ] `P3-X-4` Produce at least one judge-friendly visualization from any successful reopened line

---

## 6. Reopen Guardrails

- [ ] Do not run two GPU jobs at once
- [ ] Do not rerun another same-family `PIA` sweep without a new hypothesis
- [ ] Do not rerun another target-side `CLiD` rung just to increase count
- [ ] Do not spend GPU on cosmetic scale-up that does not change the narrative
- [ ] Do not overwrite frozen baseline package semantics with exploratory results

These are process rules, not deliverables. Keep them checked unless you explicitly violate one and explain why.

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

## 8. Completion Rule For This Reopened Roadmap

This reopened roadmap is complete only when all of the following are true:

- [ ] At least one materially different new gray-box defense line has a verdict
- [ ] At least one credible black-box mitigation line has a verdict
- [ ] At least one paper-faithful upgrade attempt has a verdict
- [ ] At least one newly implemented attack family beyond the frozen package has a verdict
- [ ] At least one second white-box line (`NeMo` or equivalent) has a real verdict
- [ ] Cross-cutting section has at least one useful result or negative result

---

## 9. Changelog

| Date | Change |
|------|--------|
| 2026-04-15 | Archived the previous competition-sprint roadmap to `legacy/2026-04-15-competition-sprint-roadmap-archived.md` |
| 2026-04-15 | Replaced the root roadmap with a P0-P3 checkbox-based reopened full-exploration roadmap |
