# DiffAudit Research ROADMAP — Continuous Autonomous Mainline

> Last updated: 2026-04-21 18:08
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

### 4.0 GPT-5.4 Pro Research Package (X-92, 2026-04-18)

**Status**: Delivered, reports received and synthesized

**Package contents**:
- 6 research directions × 20 files (120 material files total)
- Prompts optimized for divergent thinking and deep literature survey
- Material files: 800-2800 words, with math foundations, pseudocode, cost analysis
- Explorer hypotheses: 9 new bounded CPU-first candidates (3 per box)

**Report reception**:
- `docs/report-bundles/gpt54/round1-results/` first-round raw results preserved
- `docs/report-bundles/gpt54/round2-results/` second-round raw results preserved

**Canonical convergence anchor**:
- round-2 planning convergence is now maintained directly in `ROADMAP.md`
- `docs/comprehensive-progress.md` carries the synced higher-layer summary

**Planning convergence**:
- active execution is no longer six-way parallel
- current priority ladder is:
  1. `04-defense`
  2. `05-cross-box`
  3. `02-gray-box`
  4. `03-white-box`
  5. `01-black-box`
  6. `06-g1a`

**Next**:
- promote report consensus into roadmap-level priorities
- keep `04` as the current active slot after `05-H4` auxiliary-only landing
- treat `02/03/01` as support lanes rather than equal-priority slots
- keep `06` as governance fallback rather than a near-term main slot

### 4.1 Black-box

Current truth:

- `recon` is the strongest main evidence line
- issue #10 is now closed as `positive hardening`: strict `Attack-I` Stage 0 starts must use `check-recon-stage0-paper-gate`, and the current public bundle correctly returns `blocked` because it is still `proxy-shadow-member / local-semantic-chain-ready`, not paper-aligned
- `CLiD` provides strong local corroboration but still has boundary-quality work left
- a second truly different black-box family is still not firmly landed
- current round-2 readout candidates are converging on `response-cloud geometry` vs `strength-response curve`
- current black-box value is more about preserving a candidate pool than taking the next main slot

Current need:

- freeze one honest `recon` comparator surface before new black-box promotion
- preserve `response-cloud geometry / strength-response / micro-bag statistical audit` as parked candidate pool
- do not let black-box preempt `05 -> 04 -> 06` unless it directly solves a project-level blocker

### 4.2 Gray-box

Current truth:

- `PIA` is the strongest mainline
- `stochastic-dropout(all_steps)` is the current defended story
- `SecMI` exists, but diversity is not yet sufficient
- report synthesis still converges on `SimA` as the best next second-signal candidate
- but current repo truth is sharper: plain `SimA` scorer execution is already `execution-feasible but weak`, so the fastest honest gray-box follow-up is now `PIA + SimA` support-fusion / calibration review rather than another plain scorer rerun

Current need:

- do not reopen plain `SimA` scorer execution without a genuinely new bounded hypothesis
- reuse `PIA + SimA` for support-fusion / low-FPR calibration review rather than opening a separate gray-box mainline
- test `suspicion-gated late-step perturbation` only after the `SimA` sidecar is real

### 4.3 White-box

Current truth:

- `GSA` is a very strong white-box line
- `W-1 = DPDM` gives a meaningful defended comparator
- second white-box line is still unresolved
- report synthesis now converges on `activation-subspace fingerprint` as the cleanest distinct second family
- the most promising post-training defense is `risky-subspace pruning / projection`

Current need:

- preserve `H1 activation-subspace fingerprint` as the medium-horizon second-family default
- preserve `H4 risky-subspace pruning` as the medium-horizon post-training defense default
- do not let white-box same-family auxiliary lines reopen the near-term main slot before `05 -> 04 -> 06`

### 4.4 Cross-box and system-consumable layer

Current truth:

- comparison table exists
- challenger queue exists
- evidence chain is already more structured than before
- `agreement-first` is no longer the honest next `I-C` interpretation
- report synthesis now converges on `GSA + PIA` shared-score validation as the next executable cross-box packet

Current need:

- build one canonical `GSA + PIA` shared score table
- validate `calibrated late fusion` and `support/disconfirm/neutral` before any heavier portability branch
- keep all cross-box claims tied to low-FPR gain on shared surfaces, not broad agreement wording

### 4.5 Current execution posture

Current truth:

- `active GPU question = none`
- `next_gpu_candidate = none`
- issue #10 is now closed as `positive hardening`: a dedicated strict `recon` Stage 0 paper gate now blocks paper-faithful `Attack-I` when the bundle only proves `proxy-shadow-member`; this changes Research-side start-gate semantics only and does not release any GPU work or consumer schema change
- the first real `PIA vs TMIA-DM confidence-gated switching` packet already closed as `negative but useful`
- `X-88` is now closed as `bounded`: `G1-A = gray-box tri-evidence audit scorer` won post-`B-M0` long-horizon scoping as the next honest distinct-family candidate
- `X-89` is now closed as `positive`: the `X-88` CPU gate landed honestly on frozen `gpu256_undefended / gpu256_defended` surfaces with one real internal tri-score canary while keeping current consumer schema unchanged
- no direct GPU fire is authorized yet; the latest `k16 alpha-up` follow-up has already been executed and closed negative, so the next honest move is back to CPU-side parameter-selection review rather than another immediate GPU rerun
- `X-13` is now closed as `positive`: higher-layer / system-consumable entry points now carry the sharper `I-C` boundary, including `translated-contract-only + negative falsifier`, without changing any admitted mainline table or opening a new consumer contract
- `I-D.1` is now closed as `positive but bounded`: the first honest conditional target contract is frozen to `Stable Diffusion v1.5 base + celeba_partial_target/checkpoint-25000` on a local latent-diffusion canary surface, while current `recon DDIM public` evidence remains only a family-level runtime support surface
- `I-D.2` is now closed as `positive but bounded`: on one frozen `8 / 8 / 8` packet, `generation_guidance_scale = 7.5` widens raw member-vs-nonmember `MSE` separation over `3.5`, but fixed thresholds are not portable across scales
- `I-D.3` is now closed as `positive but bounded`: hidden-guidance jitter is a plausible conditional randomization defense idea, but current mixed-packet evidence is still seed-sensitive and below low-FPR / adaptive release
- `X-14` is now closed as `positive`: the first bounded `I-D` attack/defense packet pair is synchronized into higher-layer truth without changing admitted tables, consumer schema, or competition-facing claims
- `I-D.4` is now closed as `negative but useful`: the first actual runner-level deterministic hidden-guidance-jitter rerun is real and reproducible, but on the frozen packet it collapses to `accuracy = 0.5 / TPR = 0.625 / FPR = 0.625`, so the candidate falls below low-FPR and adaptive-release honesty
- `I-A` refresh is now closed as `positive but stabilizing`: the failed actual `I-D.4` rerun narrows the conditional future-surface but does not weaken the current `PIA + stochastic-dropout` mechanistic packet
- `X-15` is now closed as `positive`: after the `I-A` refresh, the next honest non-graybox slot is no longer maintenance or future-surface cleanup but `I-B` execution-side reopening
- `I-B.5` is now closed as `positive`: the first executable packet should stay white-box-only and CPU-only on the native `I-B` pair rather than importing `I-C` bridge semantics
- `I-B.6` is now closed as `positive but bounded`: the first packet executes successfully on current admitted assets, but still only proves execution and local drift readability, not defense effect
- `I-B.7` is now closed as `positive`: the first attack-side review should stay on admitted `GSA` assets with one bounded evaluation-size override rather than full-scale rerender or subset-asset duplication
- `I-B.8` is now closed as `positive but bounded`: the admitted `GSA` runtime surface now exposes a real bounded attack-side evaluation control via `--max-samples`, and one CPU-only `max_samples = 64` review packet on reused admitted gradients preserves strong attack signal while sharply shrinking evaluation size and low-FPR stability
- `I-B.9` is now closed as `positive`: the first honest intervention-on/off review must be a target-anchored fixed-mask dual-run bounded board on the same `max_samples = 64` packet, with one frozen mask object replayed across target and shadow extractions rather than per-model reselection
- `I-B.10` is now closed as `positive but bounded`: the repository now exposes `run-gsa-runtime-intervention-review`, which reads one frozen target-anchored mask summary and emits bounded baseline/intervened boards plus delta fields on the same review surface, but admitted execution is still pending
- `I-B.11` is now closed as `blocked but useful`: the first admitted target-anchored fixed-mask packet is still not honest to release because the current dual-run surface bounds only evaluation, not extraction; it would still traverse the full admitted `8000`-image board twice
- `I-B.12` is now closed as `positive but bounded`: the dual-run review surface now supports extraction-side boundedness via `extraction_max_samples`, with fallback from `max_samples`, so the first packet can be honestly bounded at both extraction and evaluation layers
- `I-B.13` is now closed as `positive`: the first truly bounded admitted packet is frozen to the existing `max_samples = extraction_max_samples = 64` contract on admitted `GSA epoch300 rerun1` assets, with the native `965 / 467` frozen mask summary replayed on `cuda` under the paper-aligned runtime schedule; the packet is still cost-warning rather than cheap, but it is now honest to release on the current host
- `I-B.14` is now closed as `negative but useful`: the first actual bounded admitted packet executes cleanly on real assets, but the frozen fixed mask pushes the bounded attack-side metrics slightly upward (`AUC +0.00354 / ASR +0.015624 / TPR@1%FPR +0.03125 / TPR@0.1%FPR +0.0`) while the locality anchor remains clean, so the branch now carries one real falsifier rather than a defense-positive result
- `I-B.15` is now closed as `negative but clarifying`: the first actual `Finding NeMo / I-B` packet should now be frozen as an `actual bounded falsifier` rather than either a `zero-GPU hold` or a defense-positive result, so same-family GPU rescue reruns are below release until a genuinely new bounded hypothesis appears
- `X-16` is now closed as `positive`: after that sharper `I-B` boundary, the next honest non-graybox `CPU-first` lane is not another box-local reopen or an asset-blocked portability probe, but a cross-box / system-consumable sync pass
- `X-17` is now closed as `positive`: higher-layer entry points now carry the updated `Finding NeMo` boundary that one actual bounded packet exists and is `negative but useful`, while admitted tables and consumer schema remain unchanged
- `I-A` refresh after the first actual negative `I-B` packet is now closed as `positive but stabilizing`: the new white-box falsifier narrows a competing branch but does not weaken the mechanistic `PIA + stochastic-dropout` packet, so `I-A` remains the strongest near-term innovation track
- `X-18` is now closed as `positive`: after that `I-A` refresh, the next honest non-graybox `CPU-first` lane is `XB-CH-2 transfer / portability blocker refresh review`, not another same-family reopen
- `XB-CH-2` transfer / portability blocker refresh is now closed as `negative but useful`: recent `I-B` falsifiers, `I-A` refresh, and higher-layer sync did not create the missing paired model/split contracts, so the branch remains `needs-assets` rather than execution-ready
- `X-19` is now closed as `positive`: after the refreshed transfer blocker, the next honest non-graybox move is one bounded higher-layer stale-entry sync pass rather than another empty reselection loop or a blocked-branch reopen
- `X-20` is now closed as `positive`: Research-side entry docs and the root control board are now aligned to the current `Finding NeMo = actual bounded falsifier / XB-CH-2 = needs-assets / no active GPU` truth
- `X-21` is now closed as `positive`: after `X-20`, there is still no honest executable reopen in `XB-CH-2` or other blocked queue items, so the strongest next non-graybox lane is a bounded return to `I-A`
- `X-22` is now closed as `positive`: the remaining `I-A` residue was higher-layer presentation strength rather than mechanism truth, and leader/material entry docs now carry the stronger four-metric plus bounded-adaptive reading
- `X-23` is now closed as `positive`: after `X-22`, blocked challengers still do not expose a stronger executable branch, so the next honest move is one residual stale-entry cleanup pass before reopening substantive lane selection
- `X-24` is now closed as `positive`: the last visible stale execution-order layer in higher docs and the root control board is now aligned to current truth, so the next honest move is another non-graybox reselection pass
- `X-25` is now closed as `positive`: after `X-24`, no blocked or hold non-graybox branch gained an honest execution release, so the strongest executable next lane is to promote `PIA provenance maintenance` from carry-forward sidecar into the main CPU-first slot
- `X-26` is now closed as `positive`: the `PIA` provenance blocker no longer needs new consumer fields or a new execution release, so the correct maintenance action is to freeze its carry-forward boundary, intake reading, and reopen trigger explicitly
- `X-27` is now closed as `positive`: after `X-26`, the strongest remaining unresolved non-graybox branch is `XB-CH-2`, but only as a CPU-side blocker/contract review rather than an execution reopen
- `X-28` is now closed as `positive`: the current repo still cannot freeze one honest cross-box shared execution surface, but it can now freeze the exact reopen contract for `XB-CH-2`, so the branch remains `needs-assets` with sharper paired-model / paired-split / shared-metric requirements
- `X-29` is now closed as `positive`: after `X-28`, `XB-CH-2` remains blocker-frozen rather than execution-ready, so the strongest executable non-graybox lane becomes a bounded return to `I-A`
- `X-30` is now closed as `positive but stabilizing`: the current mechanistic / low-FPR / bounded-adaptive `I-A` reading is now stable across higher-layer docs, so `I-A` can return to sidecar status unless new drift appears
- `X-31` is now closed as `positive`: the remaining post-`X-30` drift was only stale control-plane wording, and higher-layer entry docs are now aligned again
- `X-32` is now closed as `positive`: once `I-A` is back to sidecar and all blocked/hold non-graybox branches remain below release, the strongest honest main-slot choice is to promote `cross-box / system-consumable wording maintenance` back into the main lane because active intake/system surfaces still drift on `Finding NeMo` and `Phase E` truth
- `X-33` is now closed as `positive`: the remaining stale intake/system surfaces are now aligned again, and `Finding NeMo` is no longer encoded as the current `Phase E` intake-only `zero-GPU hold` candidate in active higher-layer or machine-readable registry docs
- `X-34` is now closed as `positive`: after both control-plane and intake-plane stale surfaces were cleared, the visible non-graybox candidate pool still contains no honest ready main-slot lane above blocked/hold status or stable sidecar maintenance, so the next move is candidate-surface expansion rather than another fake reopen
- `X-35` is now closed as `positive`: the first honest way to expand the stale non-graybox pool is to restore `I-D` as an active candidate surface, because black-box still lacks a new family contract, white-box still lacks a distinct defended-family import lane, and `I-C` still lacks a genuinely new bounded cross-box hypothesis beyond its frozen translated-contract falsifier
- `X-36` is now closed as `positive`: the restored `I-D` surface does not currently contain one genuinely new bounded successor lane beyond its frozen contract, bounded `CFG` packet, and negative actual runner-level hidden-guidance-jitter rerun, so the honest outcome is to freeze `I-D` back to future-surface support only and yield to non-graybox reselection
- `X-37` is now closed as `positive`: after `X-36`, no blocked/hold non-graybox branch honestly reopened and no new `I-A` successor question appeared, so the strongest immediate main-slot move is one bounded cross-box / system-consumable stale-surface sync pass on the remaining active `I-D` material wording drift
- `X-38` is now closed as `positive`: the remaining active material-facing `I-D` wording surface is now aligned to `X-36`, so higher-layer readers no longer see `bounded hidden-guidance defense idea` without the stronger `negative actual runner-level rerun + no honest bounded successor lane` boundary

### 4.6 Post-report priority stack (2026-04-18)

Near-term mainline order:

1. `04-H1` — risk-targeted SISS unlearning bounded pilot
2. `05-H1/H2` — stable cross-box late-fusion evidence line with promoted `logistic_2feature`
3. `02-H1/H3` — `SimA` scorer and `PIA + SimA` low-FPR sidecar
4. `03-H1/H4` — white-box activation-subspace / risky-subspace medium-horizon preparation
5. `01-H1/H2` — black-box parked candidate pool after comparator freeze
6. `06-H5` — governance fallback preserved after per-sample blocker-resolution miss

Current `06` execution truth (2026-04-18):

- one reusable `06-H1` surface now exists in-repo via `export-temporal-surrogate-feature-packet` plus `evaluate-temporal-surrogate-packets`
- one real `64 -> 128 -> 256` teacher-calibrated packet was executed on current `PIA/DDPM/CIFAR10` assets against fresh `TMIA-DM long_window` teacher boards on the same fixed `12`-point timestep grid
- the first fixed `H1` instantiation stays `stable but insufficient`: at `256`, it reaches `Spearman = 0.748677 / Pearson = 0.790525 / AUC = 0.687477` against `teacher AUC = 0.850357`, with `threshold_cv = 0.009821`
- the first fixed `H2 RMIA / BASE temporal LR` packet is now also executed on the same `256` calibration packet and lands weak/unstable (`AUC = 0.644142`, `TPR@1%FPR = 0.007812`, `threshold_cv = 0.806137`)
- the existing `H5` fallback is real via internal `CDI` canary plus paired machine-readable contract, but it remains internal-only and set-level, not a per-sample `X-90` resolution
- current honest `06` state is therefore `governance-fallback-preserved but lane-yielded`

Current `05` execution truth (2026-04-18):

- the repository now exposes one reusable `analyze-crossbox-pairboard` surface for `best_single / weighted_average / logistic_2feature / support_disconfirm_neutral`
- `GSA` is no longer blocked on scalar export absence; `loss-score-export` plus `records_path` recovery now permit real shared-index intersection attempts
- `GSA loss-score export` now also supports targeted sample-ID filtering, which turns the old “larger matched packet” blocker into an executable path
- one exact-index `PIA` packet export surface now also exists for explicit member/nonmember ID lists and emits pairboard-ready `scores.json`
- the real same-label target overlap is now frozen to `461 member / 474 nonmember`
- one enlarged targeted `GSA` export now exists on the corresponding `935`-ID union allowlist and lands `target member = 522 / target non-member = 523`
- one enlarged actual pairboard now exists with `shared member = 461 / shared nonmember = 474`
- on `5x` repeated holdout, `best_single` stays `pia` in `5/5` repeats
- `weighted_average` is still tail-only auxiliary evidence: it wins `TPR@1%FPR` and `TPR@0.1%FPR` in `5/5` repeats, but loses `AUC` in `3/5`
- `logistic_2feature` is now the honest promoted candidate: it wins `AUC` in `4/5`, `ASR` in `3/5` with `1` tie, and both low-FPR tails in `5/5`
- mean held-out metrics for `logistic_2feature` are `AUC = 0.815292`, `TPR@1%FPR = 0.148918`, `TPR@0.1%FPR = 0.046753`, versus `best_single = 0.810430 / 0.096104 / 0.007792`
- current honest `05` state is therefore `stable low-FPR tail-lift confirmed on enlarged matched packet`
- the first bounded `H4` packet is now also executed and lands `negative but useful`: around `8%` relative overhead can recover some low-FPR tail, but both routed `logistic` and routed `weighted` variants collapse `AUC / ASR` too hard to count as promoted next-stage lines
- current promoted `05` result therefore remains `H1/H2 logistic_2feature`, while the immediate active slot can now yield to `04-defense`

Current `04` execution truth (2026-04-18):

- `04` no longer sits only at family-selection wording; `H1 risk-targeted SISS / retain-forget mixture` is now the selected bounded pilot family and `H2 privacy-aware adapter` is demoted to fallback
- the repo now exposes one reusable CPU-first `prepare-risk-targeted-unlearning-pilot` surface that aligns `GSA + PIA` scores on shared indices, orients scorer polarity, converts both into within-split percentile ranks, and exports bounded forget/control lists for `k = 16 / 32 / 64`
- one real prep run now exists on the enlarged full-overlap board (`461 member / 474 nonmember`) using current `PIA` exact packet export plus targeted `GSA` loss-score export
- the hoped `Top10%(GSA) ∩ Top10%(PIA)` overlap is currently only `8` members on this board, so even the smallest `k = 16` ladder cannot honestly stay inside pure intersection-only selection
- current `k = 16 / 32 / 64` forget ladders therefore all land on `aggregate-percentile` rather than `top-fraction-intersection`, with one machine-readable matched-nonmember file per rung for forgotten-subset review
- one first actual `k32` bounded retain+forget pilot is now also executed on `cuda` from target `checkpoint-9600` with `32` steps and the frozen exported forget list
- current target-member directory contains duplicate sample IDs, so the live `k32` run actually trains on `33` forget files for `32` unique forget IDs and `967` retain files for `933` unique retain IDs
- one first attack-side forgotten-subset diagnostic now also exists on the same `k32` pilot by borrowing undefended shadow exports and rerunning only the target subset board
- this first diagnostic is currently unfavorable: baseline target-transfer lands `AUC = 0.774691 / TPR@1%FPR = 0.222222 / TPR@0.1%FPR = 0.222222`, while the defended checkpoint lands `AUC = 0.755401 / TPR@1%FPR = 0.027778 / TPR@0.1%FPR = 0.027778`
- one retained high-risk companion board now also exists on the same pilot, and it remains only `mixed/weak`: `AUC` worsens (`0.703431 -> 0.670752`) while low-FPR tails recover only slightly (`0.083333 -> 0.111111`)
- one first full-split review now also exists after fixing the no-allowlist `GSA` scan path to ignore non-image artifacts like `dataset.json`
- the full target-split read remains unfavorable as well: baseline target-transfer lands `AUC = 0.618043 / ASR = 0.5515 / TPR@1%FPR = 0.018 / TPR@0.1%FPR = 0.006`, while the defended checkpoint lands `AUC = 0.596696 / ASR = 0.5665 / TPR@1%FPR = 0.011 / TPR@0.1%FPR = 0.003`
- because all attached reads are still `borrowed-shadow / defense-unaware threshold-transfer`, they still do not close the family, but the available stack is now clearly `forgotten negative + retained mixed/weak + full-split negative`
- this first pilot is therefore no longer just `execution-positive`; it is now `execution-positive + three provisional attack-side reads attached`, and none of those reads justify defense-positive wording
- the repo now also exposes a stronger paired-noise target-review control: `GSA` loss-score export can derive deterministic per-sample noise from `noise_seed + sample_key`, and `review-risk-targeted-unlearning-pilot` can forward the same seed across baseline and defended target exports
- under that stronger same-noise surface, the direction still does not reverse:
  - forgotten subset stays negative (`AUC 0.845679 -> 0.827932`, `TPR@1%FPR 0.25 -> 0.194444`, `TPR@0.1%FPR 0.25 -> 0.194444`)
  - retained companion stays flat/weak (`AUC 0.601307 -> 0.597222`, low-FPR tails unchanged at `0.083333`)
  - full split stays slightly negative (`AUC 0.623331 -> 0.617696`, `ASR 0.5585 -> 0.5805`, `TPR@1%FPR 0.027 -> 0.026`, `TPR@0.1%FPR 0.002 -> 0.002`)
- paired-noise full-split shift decomposition also fails to show a strong concentrated forgotten effect: member mean loss shifts are broadly global (`all members +0.00755`, `forgotten members +0.007678`, `all nonmembers +0.007266`)
- current honest control decision is therefore sharper than before: the current `k32` pilot is **not** worth a defense-aware rerun; if `04` stays alive, it should first change the pilot and re-pass the same-noise target-side gate
- one first changed pilot now also exists inside the same family: `k16` with unchanged training hyperparameters and the narrower forget list
- `k16` materially improves the picture relative to `k32` under the same paired-noise review surface:
  - forgotten subset still loses `AUC` (`0.903509 -> 0.885965`), but both low-FPR tails now rise (`0.315789 -> 0.368421`)
  - retained companion is flat on `AUC` (`0.781046 -> 0.781046`) and improves both low-FPR tails (`0.235294 -> 0.294118`)
  - full split is much closer to neutral (`AUC 0.623331 -> 0.622141`, `TPR@1%FPR 0.027 -> 0.026`, `TPR@0.1%FPR 0.002 -> 0.002`)
- paired-noise full-split shift decomposition for `k16` is also more disciplined: `all members +0.001841`, `forgotten members +0.002674`, `all nonmembers +0.001753`
- current honest control decision is therefore updated again: `k16` replaces `k32` as the current working instantiation, but it still does not justify defense-positive wording or a defense-aware rerun yet
- one first pure-intersection lower-bound pilot now also exists via `k8`
- `k8` proves the pure-overlap path is executable and yields the cleanest full-split drift profile so far (`all members +0.000166`, `forgotten members +0.001872`, `all nonmembers -0.000187`)
- but `k8` is too weak to replace `k16`:
  - forgotten subset becomes almost exact neutrality rather than a clear targeted gain
  - retained companion loses the tail improvement that `k16` still preserved
  - full-split stays near-neutral, but not enough better to justify the narrower forget set
- current honest control decision is therefore sharpened once more: `k16` remains the current best working instantiation, while `k8` is archived as a useful lower-bound cleanliness probe rather than the new lead pilot
- current honest `04` state is therefore `Step-0 prep landed + k32 too weak + k16 best current working instantiation + k8 pure-intersection confirmed executable but over-tightened`
- the next honest bounded follow-up is now frozen more concretely: keep `k = 16`, keep `mixture_lambda = 0.5`, keep `32` steps, and change only `alpha` once (`0.5 -> 0.75`)
- this means `04-H1` no longer needs another `k` reselection pass before the next GPU slot; the next candidate is a single-variable `k16 alpha-up` pilot rather than a broader family reopen
- that `k16 alpha-up` follow-up is now also landed and closes as `negative but useful`: forgotten tails do not improve beyond the old `k16`, retained tails regress back to flat, and full-split `TPR@1%FPR` worsens (`0.026 -> 0.024`), so the original `k16` remains the best working instantiation and `04` returns to CPU-side parameter-selection review
- current post-`alpha-up` control read is now also frozen more explicitly: stronger forget pressure is no longer an open lever, so any same-family continuation must justify a **more selective** variable (for example branch frequency or shorter budget) before another GPU slot is released
- that selective-variable space is now also narrowed to one first conditional candidate: if `04-H1` reopens inside the same family, the next honest rung is `k16 + mixture_lambda-down` rather than another pressure change, with `0.4375` frozen as the first mid-strength candidate because it reduces forget-branch frequency materially without collapsing straight to near-no-op
- that `mixture_lambda-down` candidate is now also landed and closes as `negative but useful`: forgotten and retained low-FPR tails both regress relative to the original `k16`, so the current same-family scalar-tuning space is no longer an honest immediate GPU path
- the post-`H1` family review is now sharpened again by `X-133 / X-134 / X-135 / X-136`: `04-H2 privacy-aware adapter` is no longer only `prototype-implemented / contract-incomplete`, because the full minimal canonical `diffaudit` chain is now landed on current admitted assets
- `probe-h2-assets` now resolves admitted `checkpoint-9600/model.safetensors` plus full-scan `1000 / 1000` `32 x 32 x 3` `RGB` member/nonmember roots under one bounded packet cap, and `prepare-h2-contract` now freezes the first manifest-level workspace contract with packet identity plus runtime hyperparameters
- `run-h2-defense-pilot` now also executes one real bounded admitted-asset pilot on a staged `1 / 1` packet and emits workspace-root config/log/checkpoint artifacts, while `review-h2-defense-pilot` now completes the first same-packet transfer-only attack-side board against a merged review-compatible checkpoint
- that first full-chain review is `negative but useful`: transferred `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` all remain `0.0` on the bounded `1 / 1` packet, so `H2` is now minimally contract-complete but still far below promotion or GPU release
- one minimal packet-scale follow-up is now also landed on the same contract: the `4 / 4` transfer board is no longer pure-zero (`AUC = 0.5 / ASR = 0.375 / TPR@1%FPR = 0.5 / TPR@0.1%FPR = 0.5`), but baseline and defended metrics still match exactly, so `H2` now reads `minimal contract-complete + bounded 4/4 follow-up negative but useful`
- the next honest `04` move is therefore no longer another same-contract enlargement or GPU release; `H2` should yield the next `CPU-first` slot unless a genuinely new bounded hypothesis appears
- `Research -> Runtime -> Platform` remains contract-stable here; this sharper `04` control read does not require any new schema or protocol
- the follow-up system handoff is now also closed: current `X-114` truth only sharpens research-side control wording, while admitted `attack-defense-table`, Runtime endpoints, Platform snapshot shape, and runner capability requirements all remain unchanged
- one final stale-entry sync was still needed after `X-115`: the short bootstrap prompt and challenger-queue top layer still encoded the old `G1-A` / pre-`04` steering state, so fresh sessions could still be misrouted unless those live entry docs were resynced
- one additional consumer-copy hardening step was still justified on the Platform side: public `catalog.json` snapshot wording overstated admitted black-box / gray-box / white-box truth, so the honest fix is snapshot-copy boundary hardening only, with no Runtime schema or endpoint change

Execution rule:

- `06/05/04` are active planning lines
- `02` is a sidecar enabling line, not a competing mainline
- `03` is a medium-horizon gap line
- `01` stays parked until the near-term stack yields or a black-box blocker reappears

30-day intent:

- prove or kill one honest `06` blocker-resolution route
- prove or kill one honest `05` low-FPR shared-score gain
- select exactly one `04` successor family for bounded pilot

90-day intent:

- scale only the lines that survive the first bounded gates
- turn `06 + 05 + 04` into a coherent post-4C research map
- reopen `02/03/01` only when they solve a now-explicit gap rather than a speculative one
- `X-39` is now closed as `positive`: after `X-38`, the visible non-graybox pool still contains no honest ready main-slot lane above blocked/hold branches or stable sidecar maintenance, so the correct next move is another bounded candidate-surface expansion rather than a fake reopen
- `X-40` is now closed as `positive`: the first honest way to expand the now-stale non-graybox pool again is to restore `I-C`, but only as fresh bounded cross-box hypothesis generation rather than same-pair translated-contract hardening
- `X-41` is now closed as `positive`: the first genuinely new bounded `I-C` hypothesis is not another translated intervention retry but a bounded multi-pair agreement-first cross-box hypothesis, asking whether white-box local concentration and gray-box packet-local membership advantage rank the same frozen objects directionally before stronger support claims are attempted
- `X-42` is now closed as `blocked but useful`: the agreement-first idea survives, but the first executable board contract is still missing one second member/nonmember pair freeze under the same overlap authority, so the next honest move is pairboard identity freeze rather than execution
- `X-43` is now closed as `positive but bounded`: the second pairboard identity is now frozen deterministically to member `8` and nonmember `23`, so the new `I-C` agreement-first line now has a real `2 member + 2 nonmember` identity board under the same overlap authority, but still lacks an executable agreement-board contract
- `X-44` is now closed as `blocked but useful`: the identity board is now complete, gray-box already exposes object-level scores, but the first honest agreement-board contract still lacks one frozen white-box board-local concentration scalar and selector policy
- `X-45` is now closed as `positive but bounded`: the white-box blocker is now resolved by freezing `selected_channel_abs_profile_mean` on the board-wide selected-channel set inherited from the already-frozen pair-A selector, so the fresh `I-C` line can now return to an actual four-object board read
- `X-46` is now closed as `negative but useful`: the first honest four-object board read is real and still shows class-mean same-direction residue, but it does not preserve a clean enough same-object broad order to count as positive agreement-first support, so the current fresh `I-C` line should yield the next main-slot decision back to non-graybox reselection
- `X-47` is now closed as `positive`: after the first fresh `I-C` board read landed as `negative but useful`, the most honest immediate next move is not same-board salvage and not immediate `I-A` promotion, but one bounded cross-box / system-consumable stale-entry sync pass because higher-layer readers still see the pre-`X-46` control-plane state
- `X-48` is now closed as `positive`: the active higher-layer entry docs are now aligned again to the post-`X-46` control-plane truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only work
- `X-49` is now closed as `positive`: once the stale-entry sync is cleared, no stronger ready non-graybox branch reopens above `I-A`, so the most honest next main-slot move is a bounded return to `I-A` truth-hardening
- `X-50` is now closed as `positive`: the remaining `I-A` residue sat in higher-layer carry-forward rather than in the underlying packet, and the active higher-layer entry docs now again carry the bounded repeated-query adaptive reading plus mandatory four-metric low-FPR reporting
- `X-51` is now closed as `positive`: once that `I-A` residue is cleared, no blocked/hold non-graybox branch reopens above one remaining materials-facing stale-entry sync, because `competition-evidence-pack` still encodes `SecMI` and `TMIA-DM` with pre-refresh gray-box status
- `X-52` is now closed as `positive`: the active admitted/material-facing evidence pack is now aligned again to current gray-box truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only sync
- `X-53` is now closed as `positive`: once the materials-facing stale entry is cleared, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but `I-B` is now the strongest innovation surface that has not yet received an explicit post-falsifier successor review, so the next honest move is to inspect `I-B` for any genuinely new bounded successor lane rather than to re-promote `I-A` mechanically
- `X-54` is now closed as `negative but useful`: the restored `I-B` surface still does not contain one honest bounded successor lane above its `actual bounded falsifier` freeze, because same-family rescue remains forbidden, distinct-family import is not available on this branch, and no new bounded localization-defense hypothesis is yet visible
- `X-55` is now closed as `positive`: once `I-B` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but fresh `I-C` is now the strongest innovation surface still lacking an explicit post-negative successor review after `X-46`, so the next honest move is to inspect `I-C` for any genuinely new bounded successor lane rather than to force a return to `I-A`
- `X-56` is now closed as `negative but useful`: the fresh `I-C` surface still does not contain one honest bounded successor lane above the first negative agreement-board read, because same-board salvage remains forbidden, black-box corroboration still lacks a frozen bridge surface, and no new bounded cross-permission hypothesis is visible
- `X-57` is now closed as `positive`: once `I-C` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but one active higher-layer entry doc still lags behind the current control-plane state, so the next honest move is one bounded cross-box / system-consumable stale-entry sync pass rather than immediate candidate-surface expansion
- `X-58` is now closed as `positive`: the remaining active higher-layer entry doc is now aligned again to the post-`X-56` control-plane truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only work
- `X-59` is now closed as `positive`: once the stale-entry sync is cleared, no blocked/hold branch honestly reopens above the stable `I-A` sidecar and no fresh successor lane appears inside `I-B` or `I-C`, so the next honest move is bounded non-graybox candidate-surface expansion rather than a forced same-family return
- `X-60` is now closed as `positive`: the restored non-graybox candidate surface is `black-box paper-backed next-family scouting`, because black-box is the only remaining non-graybox area with honest CPU-only expansion room that does not violate current frozen negative or `needs-assets` boundaries
- `X-61` is now closed as `negative but useful`: the remaining paper-backed black-box backlog still does not expose one genuinely new promotable family, because the face-image LDM route is domain-specific, collection-level, and structurally overlaps with `semantic-auxiliary-classifier` and gray-box-owned `CDI`
- `X-62` is now closed as `positive`: once that black-box scouting surface also closes negative and no immediate stale-entry sync need remains, the strongest next live lane returns to `I-A` truth-hardening rather than another artificial candidate-surface expansion
- `X-63` is now closed as `positive`: one remaining materials-facing `I-A` residue existed in the active PIA visual prompt, which still invited `AUC-only` defense storytelling; that residue is now cleared by forcing four-metric plus bounded-adaptive wording into the prompt itself
- `X-64` is now closed as `positive`: once `X-63` clears the last visible `I-A` residue and no stale-entry sync or blocked/hold reopen remains above sidecar maintenance, the honest next move is another bounded non-graybox candidate-surface expansion rather than a forced `I-A` rerun or white-box reopen
- `X-65` is now closed as `positive`: the restored non-graybox candidate surface is `I-B paper-backed localization-defense successor scouting`, because black-box has already closed negative, white-box still should not take the next slot, and the broader `Finding NeMo + local memorization + FB-Mem` mechanism stack still contains CPU-only hypothesis-generation room above same-family rescue churn
- `X-66` is now closed as `negative but useful`: the broader `Finding NeMo + local memorization + FB-Mem` mechanism stack still does not expose one genuinely new bounded successor hypothesis on top of the current `actual bounded falsifier`, because the extra material remains either historical intake scaffolding, observability plumbing, or paper-faithful `SD1.4/LAION` context rather than a current admitted-surface hypothesis
- `X-67` is now closed as `positive`: once the broadened `I-B` stack also freezes below active successor status, no stronger blocked/hold branch honestly reopens above the stable sidecar line, so the strongest next live lane returns to `I-A` truth-hardening
- `X-68` is now closed as `positive but stabilizing`: the current `I-A` contract still contained one real carry-forward task, but only a narrow higher-layer residue in the `Leader` one-page summary table, which still foregrounded `AUC / ASR` before the low-FPR and bounded-adaptive read path
- `X-69` is now closed as `positive`: once that last top-summary `I-A` residue is cleared, no blocked/hold branch honestly reopens above sidecar maintenance, so the next honest move is bounded non-graybox candidate-surface expansion rather than another `I-A` turn
- `X-70` is now closed as `positive`: the next honest restored non-graybox candidate surface is `WB-CH-4 white-box loss-feature challenger family`, because the visible white-box pool was exhausted but the paper-backed loss-feature family (`LSA* / LiRA / Strong LiRA`) had never been promoted into the candidate queue even though it is distinct from the current `GSA` gradient family
- `X-71` is now closed as `positive but bounded`: the restored `WB-CH-4` surface does contain one honest near-term lane, but only as a bounded same-asset `LSA*`-style contract review; `LiRA / Strong LiRA` remain above current bounded host-fit budget
- `X-72` is now closed as `positive but bounded`: current admitted `DDPM/CIFAR10` white-box assets do support one bounded same-asset `LSA*`-style loss-feature contract, but current runtime/mainline still exports gradients only, so execution remains blocked on a loss-score export surface review
- `X-73` is now closed as `positive but bounded`: one honest bounded loss-score export surface does exist, but it should be added first as a separate in-repo internal helper / CLI surface rather than by patching the upstream external extractor or mutating current admitted `run-gsa-runtime-mainline` semantics
- `X-74` is now closed as `positive but bounded`: the repository now exposes one separate bounded internal loss-score export surface on the admitted `DDPM/CIFAR10` white-box asset family, and one real-asset `cpu / extraction_max_samples = 1` smoke succeeded without mutating admitted gradient-mainline semantics
- `X-75` is now closed as `positive but bounded`: the first honest bounded packet is now frozen to a `threshold-style`, `shadow-oriented`, `shadow-threshold-transfer` board on exported scalar loss scores with `extraction_max_samples = 64` per split, while low-FPR fields remain mandatory but still below release-grade honesty at that bounded scale
- `X-76` is now closed as `positive but bounded`: the repository now exposes a separate bounded threshold evaluator surface on top of exported white-box loss-score artifacts via `evaluate-gsa-loss-score-packet`, and one real bounded smoke confirms the intended honesty boundary because the frozen shadow-only orientation/threshold transfer goes negative on target while the target self-board remains positive and therefore diagnostic-only
- `X-77` is now closed as `positive but bounded`: the first real bounded `64`-per-split actual packet now exists on the frozen white-box loss-score contract, and its shadow-transferred target board lands at `AUC = 0.699463 / ASR = 0.632812 / TPR@1%FPR = 0.03125 / TPR@0.1%FPR = 0.03125`, which is real auxiliary evidence but still below release-grade low-FPR honesty
- `X-78` is now closed as `positive but stabilizing`: the first actual loss-score packet is now boundary-reviewed and should be frozen as bounded auxiliary white-box evidence rather than promoted or immediately extended by another same-family packet, because the branch is executable and genuinely positive but still weak at low FPR and lacks a new bounded follow-up hypothesis
- `X-79` is now closed as `positive`: once the white-box loss-score branch is frozen below immediate continuation and no stale sync remains above it, the strongest next live lane returns to `I-A` truth-hardening rather than another box-local follow-up or forced candidate expansion
- `X-80` is now closed as `positive`: one active higher-layer residue still existed in `docs/mainline-narrative.md`, whose current-state paragraph was still frozen at `X-76`; that residue is now cleared and the doc again carries the post-`X-77 / X-78 / X-79` control-plane truth
- `X-81` is now closed as `positive`: after `X-80`, the strongest next move is neither another `I-A` micro-audit nor a fresh candidate-surface reopen, but one bounded `cross-box / system-consumable stale-entry sync`, because active higher-layer docs still expose stale lane state
- `X-82` is now closed as `positive`: the active higher-layer stale-entry surfaces are now aligned again to current control-plane truth, so the strongest next move is no longer another wording-only sync pass
- `X-83` is now closed as `positive`: once the stale-entry sync is cleared, no blocked/hold branch honestly reopens and no new active `I-A` residue is visible, so the honest next move becomes bounded non-graybox candidate-surface expansion rather than another forced `I-A` microtask
- `X-84` is now closed as `positive`: the restored non-graybox candidate surface is `cross-box admitted-summary quality/cost read-path hardening`, because all immediate box-local reopens remain frozen/blocked while the unified table already carries richer `quality_cost / evidence_level / boundary` fields that higher-layer admitted summaries do not foreground enough yet
- `X-85` is now closed as `positive`: the admitted summary now explicitly exposes `Evidence Level` and `Quality / Cost`, so higher-layer readers no longer have to infer execution scale and evidence grade only from headline metrics plus free-text boundary notes
- `X-86` is now closed as `positive`: after X-85, G1-A blocker resolution identified as next honest lane
- `X-87` is now closed as `positive`: G1-A/X-90 deferred to `needs-assets` (TMIA-DM 512-sample gap conflicts with CPU-first classification)
- `X-88` is now closed as `positive`: after G1-A defer, redirected to root ROADMAP B-M0 Candidate A per 2026-04-17 override
- `B-M0 Candidate A bounded GPU review` is now closed as `hold-review-only`: shadow-LR white-box loss-score follow-up is CPU-bound offline evaluation, not GPU question
- `X-89` is now closed as `positive`: after B-M0 window close (both candidates CPU-bound, no GPU release), return to I-A CPU sidecar
- the current live lane is now `X-141 non-graybox next-lane reselection after X-140 stale-entry sync` (CPU-first control lane)
- `active_gpu_question = none`
- `next_gpu_candidate = none` (G1-A deferred-needs-assets, B-M0 Candidate A hold-review-only)
- the current CPU sidecar is `I-A higher-layer boundary maintenance`
- any new GPU release still requires a separate bounded GPU review that freezes shared-surface identity, host-fit budget, story delta, and kill gate before any fire decision
- repo-hygiene sidecar truth is now sharper as well: the storage-boundary cleanup is now physically aligned, because `Research/external/downloads` has been removed, its raw-intake subtrees now live under `D:\Code\DiffAudit\Download\`, `recon-assets` is verified at `Download/black-box/supplementary/recon-assets`, `workspaces/README.md` is resynced, and `external/README.md`, `third_party/README.md`, `Download/README.md`, plus the tracked `third_party/secmi/LOCAL_ROLE.md` now explain the intended split; the remaining dual-surface cases are explicit rather than hidden: `external/SecMI` = full upstream reference clone, `third_party/secmi` = minimal vendored integration, `external/CLiD` = working clone, and `Download/.../clid-mia-supplementary` = raw supplementary mirror

Near-term priority order:

1. review `G1-A larger shared-surface tri-score rerun` as a separate bounded GPU candidate after `X-89`, not as an auto-fire continuation
2. keep `I-A higher-layer boundary maintenance` as the active CPU sidecar while GPU remains free
3. keep `LiRA / Strong LiRA`, same-family `I-B` rescue, `I-D`, and other hold branches below execution release until a genuinely new bounded hypothesis or an explicit `G1-A` review approval appears

---

## 5. Innovation Tracks

These tracks are not marketing slogans.

They are the current innovation ladder for the repository. Each one must be backed by concrete evidence, explicit gates, and honest anti-overclaim rules.

### 5.1 `I-A` Trajectory-Consistency -> Inference-Time Randomization Defense

Current status: `near-term primary innovation track / first truth-hardening packet landed`

Core claim:

- the gray-box membership signal exposed by `PIA` is best understood as `epsilon-trajectory consistency`
- inference-time randomization defenses such as `stochastic-dropout` weaken that consistency without requiring retraining

Why it matters:

- this is the strongest candidate for a real, already-nearby technical innovation
- it is stronger than merely saying "`G-1 = stochastic-dropout`"
- it creates a mechanistic story rather than an engineering toggle story

Required gates before stronger promotion:

- [x] `I-A.1` write one formal mechanism statement linking `PIA` signal to trajectory consistency
- [x] `I-A.2` add adaptive-attacker evaluation or bounded adaptive review
- [x] `I-A.3` make low-FPR reporting mandatory alongside `AUC/ASR`
- [x] `I-A.4` update higher-layer wording so this is described as a mechanistic defense result, not a narrative flourish

Current truth packet:

- canonical evidence anchor:
  - `workspaces/implementation/2026-04-17-ia-trajectory-consistency-truth-hardening.md`
  - `workspaces/implementation/2026-04-17-ia-refresh-after-negative-id4-verdict.md`
  - `workspaces/implementation/2026-04-17-ia-refresh-after-negative-ib14-verdict.md`
- current mechanistic wording:
  - `PIA` exposes `epsilon-trajectory consistency`
  - `stochastic-dropout(all_steps)` weakens that signal via inference-time randomization
- current adaptive boundary:
  - supported only for bounded repeated-query review, not for a fully defense-aware retrained attacker
- current higher-layer read rule:
  - always report `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` together

Anti-overclaim rule:

- do not claim this is validated privacy protection until low-FPR and adaptive-attacker boundaries are explicit

### 5.2 `I-B` Memory Localization -> Surgical Defense

Current status: `mid-term execution candidate / intake-gated / first bridge packet landed / first observable selected / first intervention proposal landed / first metric contract landed`

Core claim:

- memorization is likely concentrated in sparse internal units
- a stronger innovation than full defended-model replacement would be to detect those units and intervene locally

Current repository mapping:

- `GSA` provides a white-box signal surface
- `Finding NeMo` provides the nearest localization/intervention execution branch, but its current frozen boundary is `non-admitted actual bounded falsifier`, not intake-only readiness

Required gates before promotion:

- [x] `I-B.1` establish the minimum honest protocol bridge between current admitted white-box assets and localization tooling
- [x] `I-B.2` identify one bounded localization observable worth trusting
- [x] `I-B.3` define one bounded local intervention proposal
- [x] `I-B.4` define the quality-vs-defense metric contract for that intervention
- [x] `I-B.5` select one first bounded localization/intervention packet that can actually be executed on current admitted assets
- [x] `I-B.6` implement the first bounded localization/intervention packet on current admitted assets
- [x] `I-B.7` select one bounded attack-side evaluation packet for the first honest quality-vs-defense review
- [x] `I-B.8` implement bounded attack-side evaluation packet control on the admitted `GSA` surface
- [x] `I-B.9` select the first honest intervention-on/off bounded attack-side review contract on the admitted `GSA` surface
- [x] `I-B.10` implement the target-anchored fixed-mask intervention-on/off bounded attack-side review surface on the admitted `GSA` surface
- [x] `I-B.11` review execution budget and host-fit for the first admitted target-anchored fixed-mask intervention-on/off bounded packet
- [x] `I-B.12` implement extraction-side bounded dataset cap for the target-anchored fixed-mask intervention review surface
- [x] `I-B.13` review launch config and release honesty for the first truly bounded admitted target-anchored fixed-mask intervention-on/off packet
- [x] `I-B.14` execute and review the first truly bounded admitted target-anchored fixed-mask intervention-on/off packet
- [x] `I-B.15` review branch boundary and next-step honesty after the first negative actual bounded admitted packet
- [x] `I-A refresh after first actual negative I-B packet`
- [x] `XB-CH-2` transfer / portability blocker refresh review

Current truth packet:

- canonical evidence anchor:
  - `workspaces/white-box/2026-04-17-finding-nemo-minimum-honest-protocol-bridge.md`
  - `workspaces/white-box/2026-04-17-finding-nemo-bounded-attack-side-evaluation-packet-control-verdict.md`
  - `workspaces/white-box/2026-04-17-finding-nemo-first-intervention-on-off-bounded-review-contract-selection.md`
  - `workspaces/white-box/2026-04-17-finding-nemo-intervention-on-off-bounded-review-surface-verdict.md`
  - `workspaces/white-box/2026-04-17-finding-nemo-first-admitted-fixed-mask-packet-execution-budget-review.md`
  - `workspaces/white-box/2026-04-17-finding-nemo-extraction-side-bounded-cap-verdict.md`
  - `workspaces/white-box/runs/finding-nemo-bounded-attack-side-eval-control-20260417-r1/summary.json`
- current minimum honest bridge:
  - admitted `GSA` asset root
  - fixed target checkpoint root
  - activation-only migrated DDPM observability
  - one fixed selector at a time
  - fixed member/control sample binding
  - CPU-only read-only artifact export
- current first trusted localization observable:
  - raw sample-level activation tensor under fixed selector/timestep contract
  - `summary_stat` is metadata only
  - `grad_norm` remains a supporting comparator candidate
- current first bounded intervention proposal:
  - `single-selector top-k channel attenuation mask`
  - default shape:
    - selector `mid_block.attentions.0.to_v`
    - timestep `999`
    - `top-k = 8`
    - `alpha = 0.5`
  - channel-local only, not neuron identity proof
- current quality-vs-defense review contract:
  - always report `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
  - always report one mandatory no-sampling control-surface drift metric
  - report locality budget fields and compute-cost fields together
  - no positive reading from `AUC` alone
- current bounded attack-side evaluation control:
  - `run-gsa-runtime-mainline --max-samples`
  - first bounded packet fixes `max_samples = 64`
  - first bounded packet metrics:
    - `AUC = 0.988159`
    - `ASR = 0.90625`
    - `TPR@1%FPR = 0.453125`
    - `TPR@0.1%FPR = 0.0`
    - `target_eval_size = 128`
  - honest reading:
    - `control-positive / defense-unproven`
- current first intervention-on/off review contract:
  - same bounded attack-side packet:
    - `max_samples = 64`
  - same intervention object for both target and shadows:
    - selector `mid_block.attentions.0.to_v`
    - timestep `999`
    - `mask_kind = top_abs_delta_k`
    - `k = 8`
    - `alpha = 0.5`
    - frozen channel indices:
      - `[374, 471, 269, 1, 62, 360, 187, 394]`
  - no per-model mask reselection in the first contract
  - must compare:
    - one baseline bounded board
    - one intervened bounded board
  - must keep the native `965 / 467` canary-control packet as the mandatory locality / drift anchor
- current intervention-on/off review surface:
  - command:
    - `run-gsa-runtime-intervention-review`
  - reads one frozen `inmodel-packet-export` summary as the target-anchored mask source
  - emits:
    - `baseline.metrics`
    - `intervened.metrics`
    - `metric_deltas`
    - `locality_anchor`
  - current reading:
    - `implementation-positive / admitted-execution-pending`
- current execution-budget boundary:
  - current admitted family size per board:
    - `8000` images
  - current dual-run packet size:
    - `16000` image-level extractions
  - current blocker:
    - `max_samples` only bounds evaluation, not extraction
  - honest reading:
    - `blocked but useful`
- current extraction-side boundedness:
  - command surface:
    - `run-gsa-runtime-intervention-review --extraction-max-samples`
  - fallback rule:
    - if omitted, use `max_samples`
  - current reading:
    - `execution-budget blocker cleared`
- current next step:
  - `X-19 non-graybox next-lane reselection after refreshed transfer blocker review`

Anti-overclaim rule:

- until those gates land, `Finding NeMo` remains a localization-defense track candidate, not a current mainline result

### 5.3 `I-C` Cross-Permission Signal Unification

Current status: `research hypothesis / first falsifiable packet landed / first mask family frozen / first support contract landed`

Core claim:

- black-box, gray-box, and white-box member signals may be different projections of the same internal memorization structure

Why it matters:

- this would turn the project from three parallel attack lanes into one unified theory-bearing framework

Required gates before promotion:

- [x] `I-C.1` write one falsifiable minimal experiment
- [x] `I-C.2` define which internal units or masks would be tested
- [x] `I-C.3` define which black-box / gray-box / white-box metrics must move together to count as support
- [x] `I-C.4` decide whether one bounded white-gray bridge packet is honest to release
- [x] `I-C.5` define the minimum executable surface scaffolding for that packet
- [x] `I-C.6` implement the minimum CPU-first scaffold for that packet
- [x] `I-C.7` interpret the CPU canaries and decide whether the GPU candidate should be restored
- [x] `I-C.8` define the same-packet identity and in-model intervention contract
- [x] `I-C.9` implement canonical-index bridge binding and freeze the first membership-consistent matched pair
- [x] `I-C.10` implement the in-model white-box intervention surface and CPU matched-pair co-movement canary
- [x] `I-C.11` review selector-alias and architecture compatibility for gray-box bridge intervention
- [x] `I-C.12` implement a gray-box translated-contract alias probe on `middleblocks.0.attn.proj_v`
- [x] `I-C.13` review whether the executed `I-C.10 + I-C.12` packet set deserves any stronger bridge verdict than `translated-contract canary only`
- [x] `I-C.14` execute one translated-contract targeted-vs-random falsifier on the frozen pair under the same locality budget
- [x] `X-12` reseat the next live lane after the translated `I-C` packet yields both executability and falsifier truth
- [x] `X-13` sync the sharper `I-C` boundary into cross-box / system-consumable higher-layer truth
- [x] `I-D.1` define one honest conditional target contract
- [x] `I-D.2` define one bounded CFG-scale probe
- [x] `I-D.3` define one bounded CFG-randomization defense idea
- [x] `X-14` sync the first bounded `I-D` attack/defense packet pair into higher-layer truth and freeze whether a runner-level hidden-jitter rerun deserves release review
- [x] `I-D.4` execute one actual runner-level hidden-guidance-jitter rerun on the frozen `8 / 8 / 8` packet

Current truth packet:

- canonical evidence anchor:
  - `workspaces/implementation/2026-04-17-id3-bounded-cfg-randomization-defense-idea-verdict.md`
- current minimal falsifiable packet:
  - one local `DDPM/CIFAR10` overlap surface only
  - one bounded white-box local attenuation mask imported from `I-B`
  - one matched random-mask control with the same locality budget
  - one white-box internal directional readout:
    - targeted local activation contrast should shrink
  - one gray-box external directional readout:
    - `PIA` member advantage should weaken on the same packet
  - one mandatory drift guard:
    - the effect must survive the already-frozen `I-B.4` control-surface budget
- current falsifier:
  - if the targeted mask does not beat the matched random mask, or the white-box change does not co-move with gray-box weakening, the first local unification reading fails on this surface
- current first test mask family:
  - unit type:
    - channel indices on `mid_block.attentions.0.to_v` at timestep `999`
  - primary mask:
    - `top_abs_delta_k`
  - controls:
    - `random_k_seeded`
    - `bottom_abs_delta_k`
  - current default budget:
    - `k = 8`
    - `alpha = 0.5`
  - explicit exclusions:
    - no neuron naming
    - no multi-selector or multi-timestep masks
    - no `grad_norm`-derived mask family in the first rung
- current support-counting contract:
  - white-box local movement alone is necessary but not sufficient
  - first valid support tier is:
    - `white-gray bridge support`
  - mandatory white-box metrics:
    - `selected_channel_abs_delta`
    - `selected_delta_retention_ratio`
    - one `off_mask_drift` reading
  - mandatory gray-box packet-local metric:
    - `PIA` member-control score gap on the matched packet
  - mandatory gray-box split-level bundle:
    - `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
  - black-box current admitted metrics are not valid first-packet support metrics on this surface
  - future black-box contribution is corroboration-only until a compatible bridge lands
- current release-review verdict:
  - one bounded white-gray bridge packet is currently `blocked`
  - blocker A:
    - white-box mask execution surface is missing
  - blocker B:
    - gray-box matched-packet `PIA` score-gap export surface is missing
  - current GPU posture:
    - `active_gpu_question = none`
    - `next_gpu_candidate = none`
- current minimum unblock scaffold:
  - white-box side:
    - `export-gsa-observability-masked-packet`
    - pre/post-mask tensors
    - `selected_channel_abs_delta / selected_delta_retention_ratio / off_mask_drift`
  - gray-box side:
    - `export-pia-packet-scores`
    - `sample_scores.jsonl`
    - `member_control_score_gap`
    - explicit fixed member/non-member packet indices
- current CPU-first scaffold truth:
  - white-box canary:
    - `workspaces/white-box/runs/cross-permission-masked-packet-canary-20260417-r1/summary.json`
    - `selected_delta_retention_ratio = 0.5`
  - gray-box canary:
    - `workspaces/gray-box/runs/pia-packet-score-export-20260417-r1/summary.json`
    - `member_control_score_gap = 18.217201`
  - current reading:
    - execution surface exists on both sides
    - support truth is still not established
    - current canaries are not yet one joint packet
  - current `I-C.7` interpretation:
    - blocker A:
      - white-box and gray-box packet identity is still mismatched
    - blocker B:
      - white-box intervention remains offline tensor masking, not in-model bridge execution
    - current GPU posture:
      - `active_gpu_question = none`
      - `next_gpu_candidate = none`
- current same-packet identity contract:
  - canonical packet key:
    - `CIFAR10 canonical_index`
  - canonical membership semantics:
    - `member = PIA mia_train_idxs`
    - `nonmember = PIA mia_eval_idxs`
  - first honest packet size:
    - `1 member + 1 nonmember`
  - white-box `split/sample_id` remains an auxiliary locator, not the primary membership authority
  - current white-box control canary:
    - `target-nonmember/00-data_batch_1-00467.png`
  - current contract consequence:
    - that control sample maps to `canonical_index = 467`, which is a `PIA` member rather than a `PIA` nonmember, so the old white-box pair is not same-packet-ready
- current matched-pair freeze:
  - member object:
    - `target-member/00-data_batch_1-00965.png`
    - `canonical_index = 965`
    - `PIA member_offset = 1238`
  - nonmember object:
    - `target-nonmember/00-data_batch_1-01278.png`
    - `canonical_index = 1278`
    - `PIA nonmember_offset = 1803`
  - current CPU-first artifact pair:
    - `workspaces/white-box/runs/cross-permission-matched-pairfreeze-20260417-r1/summary.json`
    - `workspaces/gray-box/runs/pia-packet-score-export-matched-pairfreeze-20260417-r1/summary.json`
  - current reading:
    - same-packet identity is now frozen on one honest `1 + 1` pair
    - support truth is still not established
    - current gray-box packet score gap on the frozen pair is `-6.157752`
- current in-model intervention contract:
  - the white-box intervention must be applied inside the forward path on the same:
    - `canonical packet`
    - `checkpoint`
    - `timestep`
    - `noise_seed`
    - `prediction_type`
    - `layer_selector`
    - `mask_kind / channel_indices / alpha`
  - offline tensor masking remains scaffold-only and does not satisfy the bridge contract
  - the first bridge-ready packet must emit:
    - `baseline` and `intervened` forward results
    - one downstream white-box readout
    - one gray-box matched-packet `PIA member_control_score_gap`
- current `I-C.10` execution reading:
  - white-box in-model packet canary:
    - `workspaces/white-box/runs/cross-permission-inmodel-packet-canary-20260417-r1/summary.json`
  - current white-box reading:
    - `selected_delta_retention_ratio = 0.5`
    - `epsilon_prediction_rms_drift_mean = 2.78113e-07`
    - `epsilon_prediction_max_abs_drift_mean = 1.549721e-06`
  - current blocker:
    - gray-box does not expose the same selector name as white-box
    - the closest `PIA` structural alias is shape-mismatched with the current white-box channel contract
  - current GPU posture:
    - `active_gpu_question = none`
    - `next_gpu_candidate = none`
- current `I-C.11` selector-alias review:
  - primary honest gray-box alias:
    - `middleblocks.0.attn.proj_v`
  - secondary fallback alias:
    - `middleblocks.0.attn`
  - current runtime alias shape on the frozen matched pair:
    - `(1, 256, 4, 4)`
  - current hard blocker:
    - white-box channel contract assumes `channel_dim = last axis`
    - gray-box alias exposes `channel_dim = 1`
    - white-box selector width/operator family does not match the gray-box alias directly
  - current reading:
    - bridge remains `blocked but useful`
    - no honest same-spec gray-box reuse exists yet
    - the next task is an alias-scoped translated-contract probe, not GPU release
- current `I-C.12` translated-contract canary:
  - canonical run artifact:
    - `workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/summary.json`
  - canonical packet:
    - member `canonical_index = 965`
    - nonmember `canonical_index = 1278`
  - executed translation contract:
    - alias selector `middleblocks.0.attn.proj_v`
    - translated from `mid_block.attentions.0.to_v`
    - `translation_kind = translated-contract`
    - `same_spec_reuse = false`
    - tensor layout `BCHW`
    - `channel_dim = 1`
  - executed runtime facts:
    - `alias_weight_shape = (256, 256, 1, 1)`
    - `alias_activation_shape = (1, 256, 4, 4)`
    - `selected_delta_retention_ratio = 0.5`
    - `off_mask_drift = 0.0`
    - `baseline_member_control_score_gap = -6.157752`
    - `intervened_member_control_score_gap = -6.191175`
    - `member_control_score_gap_delta = -0.033422`
  - current reading:
    - translated-contract execution now exists on the frozen matched pair
    - gray-box packet-local scores move non-trivially, with a larger delta on the member than the nonmember
    - this is still below same-spec bridge support and below GPU release
- current `I-C.13` bridge-promotion review:
  - canonical review note:
    - `workspaces/white-box/2026-04-17-cross-permission-bridge-verdict-review.md`
  - current blocking facts:
    - `same_spec_reuse = false`
    - no translated-contract targeted-vs-random falsifier exists yet
    - no split-level gray-box bundle exists yet on the translated packet
    - the current frozen pair starts from a negative gray-box member-control gap rather than an obvious support-eligible member advantage
  - current reading:
    - promotion to `support` remains blocked
    - the current packet set should be read as `translated-contract canary only`
    - the track is not `no-go`, but immediate GPU release still remains dishonest
- current `I-C.14` translated-contract falsifier:
  - canonical review note:
    - `workspaces/white-box/2026-04-17-cross-permission-translated-falsifier-review.md`
  - compared masks:
    - targeted `top_abs_delta_k`
    - control `random_k_seeded`
    - control `bottom_abs_delta_k`
  - current translated-surface reading:
    - targeted alias-local contrast is much larger than both controls
    - targeted gray-box packet delta does not beat matched random on the support-facing readout
    - the first translated-contract falsifier is therefore negative on the current frozen pair
  - current consequence:
    - `I-C` remains below support
    - immediate `I-C` GPU release remains dishonest
    - further work on the same frozen pair now requires a genuinely new bounded hypothesis rather than more packet churn
- current `X-41` fresh-hypothesis generation:
  - the first genuinely new bounded `I-C` hypothesis after the translated-falsifier freeze is now:
    - `I-C.15 bounded multi-pair agreement-first hypothesis`
  - current reading:
    - the next honest support object should be a tiny canonical pairboard rather than another single translated intervention packet
    - the first question is directional agreement-before-intervention, not immediate causal support promotion
    - current GPU posture remains:
      - `active_gpu_question = none`
      - `next_gpu_candidate = none`
- current `X-42` agreement-first contract review:
  - the first honest board shape can now be frozen to:
    - `2 members + 2 nonmembers`
    - CPU-first only
    - no GPU release
  - current narrow blocker:
    - only one matched `1 + 1` pair is frozen in active repo truth
    - one second member/nonmember pair is still missing
- current `X-12` reselection reading:
  - black-box still has no honest ready next-family promotion candidate
  - white-box still has no honest immediate next-hypothesis execution lane
  - `I-A` remains a CPU sidecar maintenance track rather than the best new main lane
  - the highest-value immediate next move is therefore cross-box / system-consumable sync
- current `X-13` system-sync reading:
  - challenger queue now no longer reports a stale `I-C.8` live lane
  - leader-facing summary now explicitly preserves the `translated-contract-only + negative falsifier` boundary
  - no new admitted metric, no new consumer schema, and no new GPU release are implied by this sync
- current exclusion boundary:
  - black-box is intentionally outside `I-C.1` and belongs to `I-C.3`
- current next step:
  - `I-D.2 bounded CFG-scale probe`

Anti-overclaim rule:

- this track remains hypothesis-only until at least one executed packet survives the targeted-vs-random falsifier, the white-gray support contract, and a release review grounded in a real executable surface; current admitted `recon` metrics still do not count as first-packet corroboration

### 5.4 `I-D` Conditional Diffusion / CFG Guidance Audit

Current status: `contract-frozen / first bounded probe pending`

Core claim:

- conditional diffusion likely exposes a different and more deployment-relevant leakage surface through CFG guidance behavior

Why it matters:

- it aligns with the most commercial target family (`Stable Diffusion / DiT / Kandinsky` style systems)
- it can become both a new attack surface and a new defense/randomization surface

Frozen `I-D.1` contract:

- `target family = Stable Diffusion v1.5` style `text-conditioned latent diffusion`
- `concrete local target = stable-diffusion-v1-5 base + celeba_partial_target/checkpoint-25000 LoRA`
- `reading = latent-diffusion local canary contract`, not full conditional-diffusion coverage
- current `recon DDIM public` runtime evidence supports this family only as a real conditional/runtime entry surface
- `DiT`, `Kandinsky`, and `Finding NeMo` remain outside the frozen first contract

Required gates before promotion:

- [x] `I-D.1` define one honest conditional target contract
- [x] `I-D.2` define one bounded CFG-scale probe
- [x] `I-D.3` define one bounded CFG-randomization defense idea

Anti-overclaim rule:

- current `DDPM/CIFAR10` results must not be presented as if they already establish conditional-diffusion audit capability
- current `recon DDIM public` runtime evidence must not be presented as if it already covers all conditional families or paper-faithful member semantics
- current `CFG` probe results must not be presented as if one scale already dominates universally or as if low-FPR portability has been established
- current hidden-guidance-jitter results must not be presented as if defense success or adaptive robustness has already been established

Frozen `I-D.2` probe:

- same local family only: `SD1.5 + celeba_partial_target/checkpoint-25000`
- same `DDIM 10 / 10` execution surface only
- same `8 / 8 / 8` packet only
- vary `generation_guidance_scale` only: `3.5` vs `7.5`
- current read:
  - `7.5` widens raw separation on the frozen packet
  - but cross-scale threshold portability fails

Frozen `I-D.3` idea:

- same local family only: `SD1.5 + celeba_partial_target/checkpoint-25000`
- no new family / scheduler / prompt-source change
- first defense idea = hidden guidance jitter between `3.5` and `7.5`
- current read:
  - mixed hidden-guidance packets usually degrade attack accuracy relative to fixed-scale packets
  - but the effect is seed-sensitive and low-FPR/adaptive truth is still below release

### 5.5 Long-Horizon Research Phases

`Research` should evolve in phases, not as an endless pile of unrelated tasks.

#### `R-Phase-1` Immediate truth-hardening (`now -> 2026-04-30`)

Goals:

- complete `X-66` broadened-`I-B` scoping and `X-67` reselection, then test whether `I-A` still contains one live carry-forward truth-hardening task
- keep `I-B` frozen as `actual bounded falsifier` unless a genuinely new localization-defense hypothesis appears
- keep `I-C` frozen as `translated-contract-only + negative falsifier` unless a genuinely new cross-permission hypothesis appears
- keep `I-D` frozen as `bounded conditional packet landed + negative actual defense rerun` unless a genuinely new conditional successor hypothesis appears
- keep `I-A` boundary/provenance maintenance alive as a CPU sidecar, not a fresh rerun appetite
- keep `gpu_release = none` unless a genuinely new bounded hypothesis appears

#### `R-Phase-2` Mid-term bridge building (`2026-05`)

Goals:

- search for one genuinely new bounded successor on `I-B`
- search for one genuinely new bounded successor on `I-C`
- search for one genuinely new bounded successor on `I-D`
- keep each box supplied with one honest champion/challenger structure

#### `R-Phase-3` National Innovation expansion (`2026-06+`)

Goals:

- either promote `I-C` into a unified-framework branch or freeze falsifier-first boundaries cleanly enough to stop churn
- only widen `I-D` beyond the current `SD1.5 + bounded CFG packet` surface if a genuinely new bounded successor survives review
- keep all outputs system-consumable and reusable by higher layers

### 5.6 Rolling Execution Windows

#### `Now | 24h`

- write and review `Research/workspaces/2026-04-17-g1a-bounded-gpu-review.md` before any GPU fire decision
- review `G1-A larger shared-surface tri-score rerun` as a separate bounded GPU candidate after `X-89`
- keep `active GPU question = none / next_gpu_candidate = G1-A larger shared-surface tri-score rerun review pending separate bounded GPU review`
- preserve `I-A higher-layer boundary maintenance` as the active CPU sidecar

#### `Next | 72h`

- either freeze `G1-A` as `ready-for-bounded-gpu-review` with explicit host-fit / story-delta / kill-gate evidence, or hold / no-go it cleanly without auto-firing GPU
- if `G1-A` fails review, keep expanding challenger/intake/innovation surfaces rather than reopening blocked or hold branches mechanically
- keep `Finding NeMo` same-family rescue, `GB-CH-2`, and `XB-CH-2` below execution until a genuinely new bounded hypothesis or new assets appear
- keep `I-D` bounded and do not widen the contract to `DiT`, `Kandinsky`, or white-box `SD1.4` unless a genuinely new bounded hypothesis appears first
- force one explicit successor-or-freeze decision on whichever of `I-B / I-C / I-D` currently has the most plausible fresh hypothesis surface

#### `Near | 2 weeks`

- produce one genuinely new successor verdict on at least one of:
  - `I-B`
  - `I-C`
  - `I-D`
- or freeze all three as "executed-but-currently-no-successor" and expand a new candidate surface elsewhere

#### `Mid | 1-2 months`

- land at least one nontrivial secondary innovation successor packet:
  - `I-B` genuinely new localization-defense successor
  - or `I-C` genuinely new support/disconfirmation packet
  - or `I-D` genuinely new conditional successor packet

### 5.7 Promotion / Kill Criteria

#### Promote to GPU only when

- same-packet or same-asset identity is frozen
- machine-readable artifact schema is frozen
- the task can still change a project-level story rather than only a local metric
- host-health cost is acceptable relative to expected information value

#### Hold or kill early when

- two consecutive reviews remain blocked on packet identity or asset mismatch
- only same-family micro-optimization is left
- control / drift / comparability contract is still ambiguous
- the machine-cost or babysitting cost now exceeds likely research value

#### Expand instead of resting when

- a track closes cleanly but opens an adjacent stronger question
- a negative result clarifies selection and reveals a new bounded branch
- higher-layer wording, queue truth, or artifact truth has become stale enough to change decisions

### 5.8 Cross-Repo Handoff Policy

`Researcher` may directly interface with `Platform/` or `Runtime-Server/` when that creates real leverage.

Good triggers:

- a result changes exported field requirements
- a result changes summary or recommendation logic
- a result requires a new packet/export contract
- a result requires a runner/runtime capability to stay executable

Rules:

- handoff is optional, not mandatory every loop
- default to note-level handoff first
- only escalate to cross-repo implementation when the research result is already stable enough to justify consumer changes
- if code changes are needed outside `Research/`, explicitly state:
  - target repo
  - affected files or interfaces
  - urgency
  - whether the handoff is blocking future research

---

## 6. Task Selection Heuristic

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

## 7. Active Backlog

### 7.0 Report-driven mainline stack (2026-04-18)

Goal:

- convert GPT-5.4 round-2 report convergence into a stable long-horizon execution order
- keep `05 -> 04` as the near-term active chain while preserving `06` as governance fallback
- prevent `01/02/03` from re-expanding into equal-priority parallel slots

Tasks:

- [ ] `R2-1` keep `ROADMAP.md` and `docs/comprehensive-progress.md` aligned with any new local validation verdict derived from the GPT-5.4 report bundles
- [x] `R2-2` execute `06-H1` teacher-calibrated temporal QR surrogate scoping under `64 -> 128 -> 256` gates
- [x] `R2-3` execute `05-H1/H2` shared-score validation on `GSA + PIA` before any three-box or portability expansion
- [x] `R2-4` select exactly one `04` successor family for bounded pilot; default `H1 SISS`, fallback `H2 adapter`
- [ ] `R2-5` land `02-H1 SimA` only as a sidecar second signal serving `05` and `04`
- [ ] `R2-6` keep `03-H1/H4` as medium-horizon white-box preparation, not a near-term slot preemption
- [ ] `R2-7` keep `01` as parked candidate pool until `recon` comparator freeze and higher-priority lines settle

`R2-3` current status (2026-04-18):

- reusable pairboard infra is landed in-repo
- first actual tiny shared-subset read is landed on current artifacts
- first targeted larger matched packet is now also landed (`GSA target export 89/77`, pairboard `45/35`)
- exact-index `PIA` packet export is now landed for explicit overlap IDs
- enlarged full-overlap pairboard is now landed (`shared member = 461`, `shared nonmember = 474`)
- `weighted_average` remains auxiliary only, but `logistic_2feature` now wins `AUC` in `4/5` and both low-FPR tails in `5/5` repeated held-out splits
- current `R2-3` verdict is now `positive`: stable low-FPR tail lift is confirmed on the enlarged matched packet
- bounded `05-H4` is now also landed as `negative but useful`: it behaves like an auxiliary cost-saver, not a promoted performance line
- `R2-3` therefore closes with promoted `H1/H2 logistic_2feature` and yields the next active slot to `R2-4`

`R2-4` current status (2026-04-18):

- `H1 risk-targeted SISS / retain-forget mixture` is still the selected single-family bounded pilot, while `H2 privacy-aware adapter` now reads as `probe + prepare + run + review landed / bounded 4/4 follow-up negative but useful / no new GPU question`
- one real CPU-first prep surface now exists in-repo via `prepare-risk-targeted-unlearning-pilot`
- one real full-overlap prep run now exists on the current `GSA + PIA` shared board (`461 member / 474 nonmember`)
- the current `Top10%` member overlap across `GSA` and `PIA` is only `8`, so the first honest `k = 16 / 32 / 64` ladders all use `aggregate-percentile` selection rather than pure intersection-only selection
- machine-readable forget-set and matched-nonmember files are now frozen for `k = 16 / 32 / 64`
- one first actual `k32 / 32-step / cuda` retain+forget pilot is now also landed, so the family-selection loop is fully closed
- forgotten-subset and retained-companion reviews are both now landed, and neither gives a defense-positive read
- the first full-split review is now also landed on `1000 / 1000` target samples after a real `GSA` scan fix for the no-allowlist path
- current attached read stack is `forgotten negative + retained mixed/weak + full-split negative`
- a same-noise review control is now also landed for target export, and paired-noise reruns keep the current `k32` pilot non-positive rather than rescuing it
- one first changed pilot inside the same family now also exists via `k16`, and its paired-noise tri-board is materially better than `k32`
- one first pure-intersection `k8` lower-bound pilot now also exists, and it confirms that shrinking all the way to overlap-only makes the line cleaner but too weak
- `R2-4` therefore now means "pilot family is real, `k32` was too weak, `k8` is too tight, and `k16` remains the current best working instantiation pending one more bounded follow-up around the `k16` regime"

`R2-2` current status (2026-04-18):

- the first actual `64 -> 128 -> 256` `H1` packet is completed
- `64/128` showed a real but still weak teacher-aligned signal
- `256` misses the main hard gates (`Spearman < 0.8`, `Pearson < 0.8`, `AUC delta vs teacher > 0.05`)
- the first fixed `H2` packet is now also completed and does not justify `512` frozen transfer
- `H5` governance fallback is now explicitly reviewed and preserved as internal-only set-level fallback truth
- `06` has now yielded the near-term active slot back to `05`, so the next honest move is no longer another immediate per-sample `H1/H2` rerun

Carry-forward rule:

- no new heavyweight GPU branch is justified before the current `05 -> 04` near-term chain is honestly judged
- no second `04` family is allowed to open before the first one reaches a real go/kill verdict

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

#### ✅ `X-7` Phase E high-layer doc sync

Goal:

- synchronize the highest-value `Phase E` summary docs after the machine-readable registry was reduced to a sparse-hold surface

Tasks:

- [x] `X-7.1` identify stale `Phase E` ordering wording in high-layer docs
- [x] `X-7.2` rewrite those docs around current sparse-hold truth
- [x] `X-7.3` preserve the distinction between document-layer conditional items, intake-only candidates, and executed bounded exploration branches

Status:

- current high-layer doc sync completed
- canonical evidence anchor:
  - `workspaces/intake/2026-04-16-phase-e-high-layer-doc-sync-verdict.md`

Verdict:

- `docs/future-phase-e-intake.md` and `docs/reproduction-status.md` were stale
- they still carried old `DP-LoRA / SecMI unblock / TMIA-DM intake` candidate-ordering wording after the sparse-registry refresh
- they now reflect the current honest split:
  - `PIA paper-aligned confirmation = document-layer conditional only`
  - `Finding NeMo = the only remaining intake-only candidate, under zero-GPU hold`
  - `DP-LoRA / SMP-LoRA = bounded exploration branch`
  - `SecMI = independent corroboration line`
  - `TMIA-DM = strongest packaged gray-box challenger`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `X-8` Finding NeMo reconsideration-gate review

Goal:

- decide whether the sparse `Phase E` registry now justifies opening the separate `hypothesis / budget review` required to reopen `Finding NeMo`

Tasks:

- [x] `X-8.1` review whether sparse-registry status changes the technical or governance boundary of `Finding NeMo`
- [x] `X-8.2` decide whether a fresh reconsideration review should be opened now
- [x] `X-8.3` record the carry-forward rule for live-lane reselection

Status:

- current reconsideration-gate review completed
- canonical evidence anchor:
  - `workspaces/intake/2026-04-16-finding-nemo-reconsideration-gate-review.md`

Verdict:

- sparse-registry status changes candidate visibility, not readiness
- `Finding NeMo` remains `adapter-complete zero-GPU hold`
- the required separate `hypothesis / budget review` should not be opened automatically just because `Finding NeMo` is now the only remaining intake-only candidate
- the current honest next move is still new bounded candidate generation or another lane, not automatic `Finding NeMo` reopen

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `X-9` Cross-box closure-round system sync review

Goal:

- synchronize the summary layer after both gray-box and white-box closed their immediate next-lane reviews, and make sure the repository no longer points at stale box-local priorities

Tasks:

- [x] `X-9.1` audit whether current summary-layer entry points reflect the new closure-round truth
- [x] `X-9.2` update any stale system-facing queue or summary entry that still points at old blockers or outdated next-lane assumptions
- [x] `X-9.3` freeze the next live CPU-first lane after the closure round

Status:

- completed for the current closure-round sync
- canonical evidence anchor:
  - `workspaces/implementation/2026-04-17-crossbox-closure-round-system-sync-review.md`

Verdict:

- summary-layer sync was required
- `docs/comprehensive-progress.md` now reflects that:
  - `active GPU question = none`
  - gray-box and white-box immediate execution lanes are both currently closed
  - current priority should move to cross-box sync plus black-box candidate-generation refresh
- `workspaces/implementation/challenger-queue.md` was stale and is now refreshed around current repo truth
- `workspaces/black-box/plan.md` now points at:
  - `black-box next-family candidate-generation refresh review`
- `next_live_cpu_lane` is now:
  - `black-box next-family candidate-generation refresh review`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `X-10` Post-gray-box-yield next-lane reselection review

Goal: freeze the next non-gray-box `CPU-first` lane after `GB-67` yielded the immediate slot and before any new GPU question is reopened

Current read:

- `active GPU question = none`
- `next_gpu_candidate = none`
- gray-box switching work reached a clean closure point
- the highest-value next move should now come from:
  - non-gray-box lane reselection,
  - `I-A` truth-hardening,
  - or another cross-box/system-structure question

Tasks:

- [x] `X-10.1` compare the best non-gray-box candidates by story impact, asset readiness, and bounded cost
- [x] `X-10.2` freeze one current execution lane, one next GPU candidate status, and one CPU sidecar
- [x] `X-10.3` expand the roadmap if the current visible non-gray-box options are still too stale or too weak

Status:

- completed for the current post-`GB-67` reselection round
- current execution lane:
  - `I-A trajectory-consistency truth-hardening`
- frozen posture:
  - `next_gpu_candidate = none`
  - `cpu_sidecar = PIA provenance / higher-layer boundary sync`
- black-box and white-box remain closure-negative for immediate reopen

Canonical evidence anchor:

- `workspaces/implementation/2026-04-17-post-graybox-yield-next-lane-reselection-review.md`

Value: ⭐⭐⭐
Budget: CPU-only

---

#### ✅ `X-11` PIA provenance / higher-layer boundary sync

Goal: synchronize competition-facing and summary-facing `PIA` wording to the current `I-A` mechanistic packet and the already-closed provenance blocker, so higher-layer readers no longer drift back to an `AUC-only` or `provenance-only` partial reading

Tasks:

- [x] `X-11.1` identify which higher-layer docs still compress `PIA` into a partial reading
- [x] `X-11.2` align those docs to the same mechanistic wording, bounded adaptive boundary, and low-FPR read order
- [x] `X-11.3` freeze the next live CPU-first lane after this sidecar sync closes

Status:

- completed for the current sidecar sync round
- synchronized the main higher-layer `PIA` entry docs around:
  - `epsilon-trajectory consistency`
  - `inference-time randomization`
  - bounded repeated-query adaptive review
  - `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
  - `paper-aligned blocked by checkpoint/source provenance`
- the next live CPU-first lane now moves to:
  - `I-B minimum honest protocol bridge`

Canonical evidence anchor:

- `workspaces/implementation/2026-04-17-pia-provenance-higher-layer-boundary-sync.md`

Value: ⭐⭐
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

#### ✅ `BB-8` Black-box next-family candidate-generation refresh review

Goal: decide whether black-box now has any honest next-family promotion candidate after the closure round, or whether the box should remain frozen until a real new family or asset change appears

Current read:

- `Recon` remains the frozen black-box headline
- `semantic-auxiliary-classifier` remains the leading challenger
- `CLiD` remains a boundary-quality corroboration line rather than a new family
- `variation` remains `contract-ready blocked`
- the remaining question is therefore candidate-generation truth, not rerun appetite

Tasks:

- [x] `BB-8.1` recheck whether any current black-box candidate is actually selection-ready rather than same-family, boundary-only, or needs-assets
- [x] `BB-8.2` compare whether `dataset-audit-track` should become a black-box promotion lane or stay outside immediate black-box promotion
- [x] `BB-8.3` freeze the next live CPU-first lane after the black-box refresh review

Canonical evidence anchor:

- `workspaces/black-box/2026-04-17-blackbox-next-family-candidate-generation-refresh-review.md`

Selection verdict:

- `BB-8` now closes as `negative but clarifying`
- black-box still does **not** expose a ready next-family promotion candidate
- current visible black-box candidates are either:
  - same-family continuation (`semantic-aux` refresh),
  - boundary-only (`CLiD`),
  - needs-assets (`variation`),
  - or better classified outside immediate black-box promotion (`dataset-audit-track / CDI`)
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- `next_live_cpu_lane` is now:
  - `second gray-box defense mechanism selection`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `GB-63` Second gray-box defense mechanism selection review

Goal: freeze which current gray-box defense mechanism should count as the honest second mechanism after black-box candidate refresh closed negatively and gray-box regained the near-term innovation slot

Current read:

- `PIA + stochastic-dropout(all_steps)` remains the admitted defended headline
- `TMIA-DM late-window + temporal-striding(stride=2)` is already the strongest defended challenger-specific branch
- cheap perturbation candidates have already closed negatively
- `Noise as a Probe` has no honest defended-extension gate on the current contract
- `MoFit` remains `current-contract hold`

Tasks:

- [x] `GB-63.1` compare current defended mechanism candidates by mechanism distinctness, repeat evidence, and narrative value
- [x] `GB-63.2` reject candidates that are already negative, contract-blocked, or hold-only
- [x] `GB-63.3` freeze the next live CPU-first lane after the selection

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-second-graybox-defense-mechanism-selection-review.md`

Selection verdict:

- `GB-63` now closes as `positive but bounded`
- freeze:
  - `PIA + stochastic-dropout(all_steps)` as admitted defended headline
  - `TMIA-DM late-window + temporal-striding(stride=2)` as the selected second gray-box defense mechanism
- do not rewrite this as a project-wide replacement defense
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- `next_live_cpu_lane` is now:
  - `distinct white-box defended-family import / selection`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `WB-20` Distinct white-box defended-family import / selection review

Goal: decide whether white-box now has any honest import-ready distinct defended family after black-box and gray-box near-term selection both closed

Current read:

- `W-1 = DPDM strong-v3 full-scale` remains the defended main rung
- `DP-LoRA` remains a bounded metric-split branch
- `Finding NeMo` remains `zero-GPU hold / not-requestable`
- `GSA2` remains same-family corroboration
- the remaining question is therefore family-import truth, not same-family continuation

Tasks:

- [x] `WB-20.1` recheck whether visible white-box candidates actually qualify as distinct defended-family imports
- [x] `WB-20.2` reject same-family, hold-only, and observability-only options
- [x] `WB-20.3` freeze the next live CPU-first lane after the white-box import review

Canonical evidence anchor:

- `workspaces/white-box/2026-04-17-distinct-whitebox-defended-family-import-selection-review.md`

Selection verdict:

- `WB-20` now closes as `negative but clarifying`
- no distinct white-box defended family is import-ready in the current round
- current visible options are still:
  - same-family corroboration (`GSA2`)
  - bounded branch continuation (`DP-LoRA`)
  - observability hold (`Finding NeMo`)
  - or family alias collapse (`Local Mirror`)
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- `next_live_cpu_lane` is now:
  - `ranking-sensitive variable search`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `GB-64` Ranking-sensitive variable search review

Goal: decide which ranking-sensitive gray-box variable search is now most worth opening after the recent cross-box closure round

Current read:

- `PIA vs SecMI` disagreement already closed negatively for naive fusion
- `PIA vs TMIA-DM` same-split disagreement already closed positively but bounded
- the remaining question is therefore not whether disagreement exists, but which bounded decision rule is honest to pursue next

Tasks:

- [x] `GB-64.1` compare whether the next variable-search lane should reopen `PIA vs SecMI` or pivot to `PIA vs TMIA-DM`
- [x] `GB-64.2` reject variable-search directions that still amount to naive fusion restatement
- [x] `GB-64.3` freeze the next live CPU-first lane after the review

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-ranking-sensitive-variable-search-review.md`

Selection verdict:

- `GB-64` now closes as `positive but bounded`
- the next honest gray-box CPU-first lane should be:
  - `PIA vs TMIA-DM confidence-gated switching design review`
- do not reopen `PIA vs SecMI` without a sharper bounded hypothesis
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- `next_live_cpu_lane` is now:
  - `PIA vs TMIA-DM confidence-gated switching design review`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `GB-65` PIA vs TMIA-DM confidence-gated switching design review

Goal: freeze the smallest honest next design for a ranking-sensitive `PIA/TMIA-DM` switching rule after `GB-64` selected this branch

Current read:

- bounded same-split `z-score sum` is already positive
- another fixed fusion rerun would add little
- the next step should test a real gating variable using only attack-side scores

Tasks:

- [x] `GB-65.1` reject supervision-heavy or class-conditional designs that current repo truth cannot support honestly
- [x] `GB-65.2` freeze one bounded confidence-gated rule family
- [x] `GB-65.3` set the next live CPU-first lane after the design review

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-pia-tmiadm-confidence-gated-switching-design-review.md`

Selection verdict:

- `GB-65` now closes as `positive but bounded`
- freeze the next bounded design as:
  - normalized-score dominant-method identity
  - margin-gap threshold
  - fallback to bounded `z-score sum`
- `next_live_cpu_lane` is now:
  - `PIA vs TMIA-DM confidence-gated switching offline packet`
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ✅ `GB-66` PIA vs TMIA-DM confidence-gated switching offline packet

Goal: run the first real bounded switching packet on aligned `PIA/TMIA-DM` score surfaces and decide whether it deserves promotion beyond a design note

Tasks:

- [x] `GB-66.1` execute the bounded offline packet on aligned undefended and defended surfaces
- [x] `GB-66.2` compare the best switching threshold against `PIA`, `TMIA-DM`, and bounded `z-score sum`
- [x] `GB-66.3` record whether the branch should promote, hold, or close

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-pia-tmiadm-confidence-gated-switching-offline-packet.md`
- `workspaces/gray-box/runs/pia-tmiadm-confidence-switch-20260417-r1/summary.json`

Selection verdict:

- `GB-66` now closes as `negative but useful`
- the first confidence-gated switching packet is real and honest, but:
  - it does not beat bounded `z-score sum` on the aligned undefended packets
  - and it underperforms on the defended packet
- therefore it should remain a bounded ranking-sensitive analysis packet rather than a promoted scorer family
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- `next_live_cpu_lane` is now:
  - `gray-box post-switch lane reselection review`

Value: ⭐⭐
Budget: CPU-only

#### ✅ `GB-67` Gray-box post-switch lane reselection review

Goal: decide whether gray-box should keep the next live CPU-first slot after the switching packet closed as `negative but useful`

Tasks:

- [x] `GB-67.1` review whether gray-box still exposes a more urgent CPU-first question than the rest of the repo
- [x] `GB-67.2` reject same-family switching-packet inflation as the default next move
- [x] `GB-67.3` freeze whether gray-box keeps or yields the next live CPU-first slot

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-graybox-post-switch-lane-reselection-review.md`

Selection verdict:

- `GB-67` now closes as `positive`
- gray-box no longer exposes a more urgent immediate CPU-first lane than the best remaining cross-box or other-box questions
- treat the switching packet as a clean closure point for the current gray-box ranking-sensitive branch
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- gray-box should now yield the next live CPU-first slot

Value: ⭐⭐
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

#### ⬜ `GB-19` Post-noise next-family reselection

Goal: decide what should become the next live CPU-first research lane after `Noise as a Probe` finished its bounded branch and current black-box / gray-box / white-box live questions all contracted

Current read:

- black-box is currently `no-new-gpu-question`
- white-box breadth is still below release and `Finding NeMo` remains `zero-GPU hold`
- gray-box remains the strongest narrative box, but both `TMIA` and `Noise as a Probe` are already packaged for the current round
- the next honest move is therefore a new bounded candidate-generation selection, not another same-family rerun

Tasks:

- [x] `GB-19.1` compare whether the next live lane should come from black-box reopen, white-box reconsideration, or a new gray-box family
- [x] `GB-19.2` choose one next lane and reject the weaker alternatives for now
- [x] `GB-19.3` define the immediate task shape for that selected lane

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-post-noise-next-family-reselection.md`

Selection verdict:

- `GB-19` now closes as `positive`
- the selected next live lane is `GB-20 MoFit protocol / asset contract`
- current reason is not that `MoFit` is already executable, but that it is now the cleanest remaining genuinely new gray-box mechanism with a bounded CPU-first entry
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-20` MoFit protocol / asset contract

Goal: decide whether the current repo can support one honest first `MoFit`-style gray-box smoke on a local latent-diffusion target family, or whether the branch is still too under-specified even for a CPU-first live lane

Current read:

- `GB-19` already selected `MoFit` as the next live lane
- the repo already has one workable latent-diffusion target-family surface:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- local caption/bootstrap assumptions and latent-diffusion loading pieces already exist
- but surrogate optimization, fitted-embedding optimization, and MoFit-specific artifact schema are still not frozen

Tasks:

- [x] `GB-20.1` lock one honest local target family and caption bootstrap source
- [x] `GB-20.2` decide what implementation pieces already exist and what still blocks a first smoke
- [x] `GB-20.3` define one bounded next step and keep `gpu_release` honest

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-protocol-asset-contract.md`

Selection verdict:

- `GB-20` now closes as `positive but bounded`
- current repo is ready for a real `MoFit` contract-first lane
- current repo is not yet ready for an honest first smoke
- selected first contract:
  - `target_family = SD1.5 + celeba_partial_target/checkpoint-25000`
  - `caption_source = local BLIP bootstrap or cached local caption fallback`
- next step:
  - `MoFit implementation-surface review / scaffold decision`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-21` MoFit implementation-surface review

Goal: decide whether the first `MoFit` implementation step should extend an existing latent-diffusion script or start from a dedicated scaffold, and identify the exact missing method-specific loops

Current read:

- `GB-20` already locked the first honest local `MoFit` contract
- the repo already has:
  - caption bootstrap via metadata + `BLIP` fallback
  - text-conditioned latent-diffusion loading
  - VAE / tokenizer / text encoder / scheduler / latent encode-decode
- but it still lacks:
  - surrogate optimization
  - fitted-embedding optimization
  - `MoFit`-specific score artifact schema

Tasks:

- [x] `GB-21.1` review which existing scripts already provide reusable latent-diffusion substrate
- [x] `GB-21.2` decide whether `MoFit` should extend an existing script or use a dedicated scaffold
- [x] `GB-21.3` record the next bounded method-specific scaffold step

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-implementation-surface-review.md`

Selection verdict:

- `GB-21` now closes as `positive but bounded`
- current repo already has the loading and caption substrate needed for a dedicated `MoFit` lane
- the honest next step is a `dedicated MoFit scaffold / schema decision`
- do not overload `structural memorization` or `semantic-aux` scripts
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-22` MoFit scaffold / schema decision

Goal: freeze the first dedicated `MoFit` scaffold shape and minimum artifact schema before any honest smoke is allowed

Current read:

- `GB-21` already fixed that `MoFit` must use a dedicated scaffold
- the repo already has reusable latent-diffusion substrate and caption bootstrap
- the remaining ambiguity is now:
  - exact script surface
  - exact minimum output schema

Tasks:

- [x] `GB-22.1` choose the dedicated scaffold form
- [x] `GB-22.2` freeze one minimum artifact schema
- [x] `GB-22.3` define the next bounded implementation step

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-scaffold-schema-decision.md`

Selection verdict:

- `GB-22` now closes as `positive`
- dedicated scaffold:
  - `scripts/run_mofit_interface_canary.py`
- minimum schema:
  - `summary.json`
  - `records.jsonl`
  - surrogate / embedding trace artifacts
- next step:
  - implement the dedicated scaffold script under the frozen schema
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-23` MoFit dedicated scaffold implementation

Goal: implement the frozen `MoFit` dedicated scaffold and verify that it can create the minimum artifact set without pretending to run the method yet

Current read:

- `GB-22` already froze the dedicated scaffold name and minimum schema
- the next honest step is now real code, not more note-only design
- that code should still stay below smoke and below GPU

Tasks:

- [x] `GB-23.1` add the minimal scaffold module and script
- [x] `GB-23.2` verify with a failing test first, then passing test
- [x] `GB-23.3` verify fresh script execution creates the expected scaffold artifacts

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-scaffold-implementation-verdict.md`

Selection verdict:

- `GB-23` now closes as `positive but bounded`
- current repo now has:
  - `src/diffaudit/attacks/mofit_scaffold.py`
  - `scripts/run_mofit_interface_canary.py`
  - `tests/test_mofit_scaffold.py`
- test evidence:
  - first red: `ModuleNotFoundError`
  - then green: `Ran 1 test ... OK`
- fresh script execution also emitted:
  - `summary.json`
  - `records.jsonl`
  - `traces/surrogate/`
  - `traces/embedding/`
- the lane still remains below smoke:
  - no surrogate optimization yet
  - no fitted-embedding optimization yet
  - no real `L_MoFit` score yet
- `gpu_release = none`

Value: ⭐⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-24` MoFit record-schema integration

Goal: integrate the minimum per-sample record schema into the dedicated `MoFit` scaffold before real optimization loops are added

Current read:

- `GB-23` already created the dedicated scaffold code and run-level artifact shell
- the next honest gap is record-level integration:
  - per-sample trace paths
  - `l_cond`
  - `l_uncond`
  - `mofit_score`

Tasks:

- [x] `GB-24.1` write a failing test for per-sample record append
- [x] `GB-24.2` implement placeholder record append plus trace-file creation
- [x] `GB-24.3` rerun tests and confirm the schema is now real

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-record-schema-integration-verdict.md`

Selection verdict:

- `GB-24` now closes as `positive but bounded`
- the scaffold now supports:
  - `records.jsonl` append
  - per-sample surrogate trace placeholder
  - per-sample embedding trace placeholder
  - placeholder `l_cond / l_uncond / mofit_score`
- the lane still remains below smoke because those fields are not yet populated by real optimization code
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-25` MoFit score / trace update path

Goal: connect the dedicated `MoFit` scaffold to a real score/trace update API so future optimization loops can write values into the frozen schema

Current read:

- `GB-24` already created per-sample records and placeholder traces
- the next honest missing piece is not new schema, but a real update path for:
  - surrogate trace
  - embedding trace
  - `l_cond / l_uncond / mofit_score`

Tasks:

- [x] `GB-25.1` write a failing test for score/trace update
- [x] `GB-25.2` implement the minimal update helper
- [x] `GB-25.3` rerun tests and confirm the update path is real

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-score-trace-update-verdict.md`

Selection verdict:

- `GB-25` now closes as `positive but bounded`
- the scaffold now has a real update path for:
  - surrogate trace JSON
  - embedding trace JSON
  - `l_cond / l_uncond / mofit_score`
- the lane still remains below smoke because the optimization loops that generate those values are not yet implemented
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-26` MoFit optimization helpers

Goal: add the smallest reusable optimization helpers needed before wiring `MoFit` into the real target-model latent path

Current read:

- `GB-25` already created the update path for scores and traces
- the next honest missing piece is now the optimization substrate itself:
  - a surrogate helper
  - an embedding helper
- these helpers should be verified on toy losses before touching the real latent path

Tasks:

- [x] `GB-26.1` write failing tests for surrogate and embedding optimization helpers
- [x] `GB-26.2` implement the minimal helpers
- [x] `GB-26.3` rerun tests and confirm the helpers actually reduce toy loss

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-optimization-helper-verdict.md`

Selection verdict:

- `GB-26` now closes as `positive but bounded`
- the scaffold now exposes:
  - `run_surrogate_optimization(...)`
  - `run_embedding_optimization(...)`
- both are verified on toy losses with fresh tests
- the lane still remains below smoke because those helpers are not yet wired into the real target-model path
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-27` MoFit latent-path loss contract

Goal: encode the real latent-path loss contract in code so future `MoFit` optimization loops optimize the same score structure the lane intends to report

Current read:

- `GB-26` already added minimal optimization helpers
- the next honest missing piece is now the score contract itself:
  - `L_cond`
  - `L_uncond`
  - `mofit_score`
- that contract should be verified on toy differentiable predictors before wiring into real SD1.5 code

Tasks:

- [x] `GB-27.1` write failing tests for the loss-contract helpers
- [x] `GB-27.2` implement the minimal contract helpers
- [x] `GB-27.3` rerun tests and confirm the score contract is now real

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-latent-loss-contract-verdict.md`

Selection verdict:

- `GB-27` now closes as `positive but bounded`
- the scaffold now exposes:
  - `compute_mofit_loss_terms(...)`
  - `build_surrogate_loss_fn(...)`
  - `build_embedding_loss_fn(...)`
- this unifies:
  - optimization helpers
  - record schema
  - `mofit_score` semantics
- the lane still remains below smoke because the contract is not yet wired into the real SD1.5 target-model path
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-28` MoFit real target-path wiring

Goal: wire the `MoFit` loss-contract helpers into the actual `SD1.5` target-model noise-prediction path shape so future loops consume real `UNet`-style outputs rather than toy-only closures

Current read:

- `GB-27` already fixed the score contract itself
- the next honest missing piece is now the helper bridge that:
  - adapts `UNet(...).sample` into the existing predictor contract
  - constructs guided target noise from real `cond / uncond` predictions
- this should still stay CPU-only and helper-layer only; it is not yet an end-to-end smoke

Tasks:

- [x] `GB-28.1` write failing tests for a real `UNet`-style predictor bridge
- [x] `GB-28.2` implement the minimal target-path wiring helpers
- [x] `GB-28.3` rerun tests and confirm the loss helpers now accept `UNet`-like outputs

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-real-target-path-wiring-verdict.md`

Selection verdict:

- `GB-28` now closes as `positive but bounded`
- the scaffold now additionally exposes:
  - `build_unet_noise_predictor(...)`
  - `compute_guided_target_noise(...)`
- this wires:
  - actual `UNet(...).sample`-style outputs
  - guided target-noise construction from `cond / uncond` predictions
  - the existing `L_cond / L_uncond / mofit_score` helpers onto a real target-path shape
- the lane still remains below smoke because caption bootstrap, sample-level record execution, and end-to-end optimization loops are not yet assembled
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-29` MoFit sample-level execution assembly

Goal: assemble one bounded per-sample `MoFit` execution helper that joins prompt bootstrap, record lifecycle, real target-path helper wiring, optimization loops, and final score writeback

Current read:

- `GB-28` already exposed the real target-path helper bridge
- the next honest missing piece is now the sample-level assembly that:
  - resolves prompt text from metadata or bounded caption fallback
  - appends a record before execution
  - runs surrogate + embedding optimization under the existing helper contracts
  - finalizes `L_cond / L_uncond / mofit_score` plus traces back into schema
- this should still stay CPU-only and helper-layer only; it is not yet a script-level real-asset smoke

Tasks:

- [x] `GB-29.1` write failing tests for sample-level execution assembly
- [x] `GB-29.2` implement the minimal prompt-resolution and per-sample execution helpers
- [x] `GB-29.3` rerun tests and confirm the sample-level record path is real

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-sample-level-execution-assembly-verdict.md`

Selection verdict:

- `GB-29` now closes as `positive but bounded`
- the scaffold now additionally exposes:
  - `resolve_mofit_prompt(...)`
  - `execute_mofit_sample(...)`
- this assembles:
  - metadata / fallback prompt resolution
  - record append/finalize over one sample
  - real target-path helper wiring
  - optimization trace persistence
  - final `L_cond / L_uncond / mofit_score` writeback
- the lane still remains below smoke because the helper is not yet mounted into a script-level run over actual local assets and model-loading substrate
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-30` MoFit script-level canary execution review

Goal: decide the smallest honest script-level path for mounting the new `MoFit` sample helper onto real local assets and the current `SD1.5` target-family stack

Current read:

- `GB-29` already closed the sample-level helper assembly gap
- the next honest missing piece is now script-level mounting:
  - bounded row loading
  - prompt / latent / embedding preparation from real local assets
  - reuse of the existing `UNet`-backed local target stack
- before coding that path, the repo should decide whether to extend the existing canary script or start another script

Tasks:

- [x] `GB-30.1` inspect the current `MoFit` canary script surface
- [x] `GB-30.2` inspect the existing local target-stack substrate already used by structural memorization
- [x] `GB-30.3` freeze the next script-level implementation surface and reject unnecessary script duplication

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-script-level-canary-execution-review.md`

Selection verdict:

- `GB-30` now closes as `positive but bounded`
- the shortest honest next path is to extend `run_mofit_interface_canary.py`
- the review rejects opening a second new `MoFit` script because the current canary entry already owns the correct CLI contract
- the next implementation should reuse the local substrate already present in `run_structural_memorization_smoke.py` for:
  - row loading
  - caption bootstrap
  - prompt encoding
  - image-to-latent encoding
  - `UNet`-backed target-path calls
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-31` MoFit script-level canary execution implementation

Goal: turn the existing `run_mofit_interface_canary.py` entry from scaffold initialization into a bounded script-level execution path

Current read:

- `GB-30` already froze the script-level implementation surface
- the next honest missing piece is now the orchestration itself:
  - bounded row loading with `offset/limit`
  - script-level split execution
  - mounting the existing sample-level helper into the canary entry
  - updating `summary.json` to reflect execution rather than scaffold-only state
- this should still stay CPU-only and unit-test-backed first; it is not yet a fresh real-asset launch

Tasks:

- [x] `GB-31.1` write failing tests for script-level row loading and canary orchestration
- [x] `GB-31.2` implement the bounded canary execution path in `run_mofit_interface_canary.py`
- [x] `GB-31.3` rerun `MoFit` test sweeps and confirm no helper-layer regressions

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-script-level-canary-implementation-verdict.md`

Selection verdict:

- `GB-31` now closes as `positive but bounded`
- the script now additionally exposes:
  - `load_rows(...)`
  - `MoFitCanaryRunner.execute_row(...)`
  - `run_split(...)`
  - `run_canary(...)`
- this upgrades `run_mofit_interface_canary.py` from scaffold-only initialization to bounded canary orchestration
- the current evidence is still unit-test-backed rather than a fresh real-asset launch on the admitted local stack
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-32` MoFit real-asset canary launch gate review

Goal: decide whether the new script-level `MoFit` canary should be launched immediately on the admitted local CPU stack, or whether the first-launch budget should be tightened first

Current read:

- `GB-31` already made the canary entry executable
- the next honest missing piece is now launch discipline:
  - current CPU defaults still imply repeated gradient-bearing `UNet` calls
  - the first admitted local launch should avoid spending unnecessary CPU on an over-wide budget
- before a fresh real-asset launch, the repo should decide whether the current budget is narrow enough

Tasks:

- [x] `GB-32.1` inspect the current launch defaults
- [x] `GB-32.2` assess whether the first admitted CPU launch should be delayed pending budget tightening
- [x] `GB-32.3` freeze the next task as budget tightening or direct launch

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-real-asset-canary-launch-gate-review.md`

Selection verdict:

- `GB-32` now closes as `hold-before-launch`
- the current script is runnable, but the first real local CPU launch should not fire yet under the present budget
- the next honest live task is launch-budget tightening rather than immediate execution
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-33` MoFit launch-budget tightening

Goal: tighten the first-launch budget in code so the default `MoFit` canary path is safe for one future admitted local CPU launch

Current read:

- `GB-32` held the launch because the default budget was still too wide
- the next honest missing piece is now explicit default-budget tightening:
  - `member_limit = 1`
  - `nonmember_limit = 1`
  - `surrogate_steps = 1`
  - `embedding_steps = 2`
  - `device = cpu`
- this should land in code and be test-verified before any fresh real-asset launch

Tasks:

- [x] `GB-33.1` write failing tests for launch-profile tightening
- [x] `GB-33.2` implement the bounded CPU-first launch profile and propagate it through `run_canary`
- [x] `GB-33.3` rerun `MoFit` regression sweeps and confirm the tightened profile is now default behavior

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-launch-budget-tightening-verdict.md`

Selection verdict:

- `GB-33` now closes as `positive`
- the script now exposes `apply_launch_profile(...)`
- the default first-launch behavior is now `bounded-cpu-first`
- the next honest live step is a fresh real local CPU canary under the tightened profile
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-34` MoFit fresh real-asset canary

Goal: execute one fresh bounded local CPU `MoFit` canary on the admitted `SD1.5 + celeba_partial_target/checkpoint-25000` stack

Current read:

- `GB-33` already tightened the first-launch profile in code
- the next honest missing piece is now a fresh real execution on the admitted local stack
- this should stay strictly bounded:
  - `member_limit = 1`
  - `nonmember_limit = 1`
  - `surrogate_steps = 1`
  - `embedding_steps = 2`
  - `device = cpu`

Tasks:

- [x] `GB-34.1` verify admitted local asset paths
- [x] `GB-34.2` execute one fresh bounded local CPU canary
- [x] `GB-34.3` record outcome and decide whether execution feasibility is now closed

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-fresh-real-asset-canary-verdict.md`

Selection verdict:

- `GB-34` now closes as `positive but bounded`
- the first fresh admitted local CPU canary now exists at:
  - `workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4`
- this closes the remaining “can it run at all on the real local stack?” question
- the observed score gap is still tiny, so execution success does not yet justify direct scale-up
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-35` MoFit canary score-shape review

Goal: decide whether the first fresh `MoFit` canary shows enough score directionality to justify immediate rung expansion

Current read:

- `GB-34` already proved fresh real local execution
- the next honest question is now score shape, not execution feasibility
- before spending more CPU, the repo should decide whether the current canary is:
  - scale-positive
  - scale-negative
  - or still inconclusive under too-small budget

Tasks:

- [x] `GB-35.1` inspect `records.jsonl` and trace artifacts
- [x] `GB-35.2` classify current canary as scale-positive, scale-negative, or inconclusive
- [x] `GB-35.3` freeze the next task as scale-up, no-go, or micro-rung design

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-canary-score-shape-review.md`

Selection verdict:

- `GB-35` now closes as `inconclusive but still alive`
- the canary is execution-positive, but the current member/nonmember score gap is too small to justify direct rung expansion
- the next honest live task is a bounded CPU micro-rung design/review
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-36` MoFit CPU micro-rung design

Goal: freeze the smallest next CPU rung that can add signal after the fresh canary without reopening wasteful budget behavior

Current read:

- `GB-35` already concluded that the fresh canary is execution-positive but score-inconclusive
- the next honest question is no longer whether to run more, but exactly how much more to run while staying bounded
- before launching again, the rung budget, stop condition, and promotion rule should be written down explicitly

Tasks:

- [x] `GB-36.1` define the smallest next bounded row/step envelope
- [x] `GB-36.2` define stop conditions and no-go triggers
- [x] `GB-36.3` freeze the next live task as micro-rung execution rather than open-ended exploration

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-design.md`

Selection verdict:

- `GB-36` now closes as `positive`
- the next bounded CPU rung is frozen to:
  - `member_limit = 2`
  - `nonmember_limit = 2`
  - `surrogate_steps = 2`
  - `embedding_steps = 4`
  - `device = cpu`
- the next honest live task is now `MoFit CPU micro-rung execution`
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-37` MoFit CPU micro-rung execution

Goal: execute the frozen `2x2 / 2+4 / cpu` `MoFit` micro-rung on the admitted local target-family stack

Current read:

- `GB-36` already froze the smallest next bounded rung
- the next honest missing piece is now the execution itself
- this rung should remain:
  - `member_limit = 2`
  - `nonmember_limit = 2`
  - `surrogate_steps = 2`
  - `embedding_steps = 4`
  - `device = cpu`

Tasks:

- [x] `GB-37.1` ensure the script can express the frozen micro-rung profile in code
- [x] `GB-37.2` execute one fresh bounded local CPU micro-rung
- [x] `GB-37.3` record outcome and decide whether execution feasibility is closed at this rung

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-execution-verdict.md`

Selection verdict:

- `GB-37` now closes as `positive but bounded`
- the first valid micro-rung now exists at:
  - `workspaces/gray-box/runs/mofit-sd15-celeba-microrung-20260416-cpu-r2`
- this closes the remaining “can the frozen 2x2 rung run at all?” question
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-38` MoFit CPU micro-rung score review

Goal: determine whether the first valid `2x2 / 2+4 / cpu` micro-rung shows enough score improvement to justify one final bounded CPU rung

Current read:

- `GB-37` already proved the micro-rung runs end to end
- the next honest question is now score shape and promotion value, not execution feasibility
- before any further budget is spent, the repo should decide whether this rung is:
  - still direction-inconclusive
  - weak-positive
  - or strong enough for another bounded expansion

Tasks:

- [x] `GB-38.1` inspect score means and trace behavior on the valid micro-rung
- [x] `GB-38.2` classify the rung as direction-negative, inconclusive, weak-positive, or scale-positive
- [x] `GB-38.3` freeze the next live task as final review rung, no-go, or promotion

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-cpu-microrung-score-review.md`

Selection verdict:

- `GB-38` now closes as `weak-positive but still below promotion`
- member scores are now consistently less negative than nonmember scores, but the gap remains tiny
- the next honest live task is one final bounded CPU review-rung design, not GPU escalation
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-39` MoFit final CPU review rung

Goal: run one final bounded CPU review rung to test whether the weak-positive micro-rung signal materially strengthens under a slightly larger optimization budget

Current read:

- `GB-38` already concluded that the micro-rung is weak-positive but still below promotion
- the next honest question is whether one last bounded CPU rung changes that conclusion materially
- this rung should stay bounded:
  - `member_limit = 2`
  - `nonmember_limit = 2`
  - `surrogate_steps = 3`
  - `embedding_steps = 6`
  - `device = cpu`

Tasks:

- [x] `GB-39.1` implement an exact `cpu-review-rung` launch profile
- [x] `GB-39.2` execute one fresh bounded local CPU review rung
- [x] `GB-39.3` record whether the signal becomes promotion-worthy or remains tiny

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-final-cpu-reviewrung-verdict.md`

Selection verdict:

- `GB-39` now closes as `weak-positive but still bounded`
- the final review rung now exists at:
  - `workspaces/gray-box/runs/mofit-sd15-celeba-reviewrung-20260416-cpu-r2`
- the gap improves again, but remains far below promotion-worthy strength
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-40` MoFit current-contract hold verdict

Goal: decide whether the current `MoFit` contract should continue consuming runtime after the fresh canary, micro-rung, and final bounded CPU review rung

Current read:

- `GB-34..39` already closed execution feasibility and bounded CPU review
- the next honest question is now lane governance:
  - continue scaling
  - hold under the current contract
  - or reopen only if the contract changes materially

Tasks:

- [x] `GB-40.1` compare the canary, micro-rung, and final review-rung gaps
- [x] `GB-40.2` decide whether the current contract remains worth additional runtime
- [x] `GB-40.3` freeze the lane as hold/no-go or select a new live lane

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-mofit-current-contract-hold-verdict.md`

Selection verdict:

- `GB-40` now closes as `current-contract hold`
- the lane is execution-positive but signal-weak under the current local contract
- no further mechanical CPU rung expansion is justified
- `gpu_release = none`

Value: ⭐⭐
Budget: CPU-only

#### ⬜ `GB-41` Post-MoFit gray-box next-family reselection

Goal: decide which bounded live lane is now most worth opening after `MoFit` closed as `current-contract hold` and all three boxes still say `gpu_release = none`

Current read:

- black-box is already stabilized at `no-new-gpu-question`
- white-box reopen candidates remain either `zero-GPU hold` or `metric-split bounded exploration branch + no-new-gpu-question`
- gray-box still offers the best project-level story impact, but its current single-sample side branches are now closed, held, or already packaged
- the next honest move is therefore another bounded gray-box family / contract selection, not more `MoFit` runtime

Tasks:

- [x] `GB-41.1` compare black-box hold, white-box reconsideration, old gray-box branch reopen, and remaining genuinely new gray-box directions
- [x] `GB-41.2` select one next live lane and reject the weaker alternatives for now
- [x] `GB-41.3` define the immediate first task shape for the selected lane

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-post-mofit-next-family-reselection.md`

Selection verdict:

- `GB-41` now closes as `positive`
- the selected next live lane is:
  - `GB-42 CDI protocol / asset contract`
- reason:
  - current repo already has reusable `PIA / SecMI` gray-box score surfaces on a shared local `CIFAR-10 DDPM` contract
  - `CDI` opens a genuinely new `collection-level audit / evidence aggregation` direction with stronger audit-facing value than another single-sample rerun
  - `SIDe` remains more bridge-like and less immediately reusable on current repo surfaces
- `gpu_release = none`
- `next_gpu_candidate = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-42` CDI protocol / asset contract

Goal: decide whether the current repo can support one honest first `CDI`-style gray-box collection-level audit lane, and freeze the minimum contract before any execution claim

Current read:

- `GB-41` already selected `CDI` as the next live lane
- current repo already has:
  - mature gray-box signal producers (`PIA`, `SecMI`)
  - same-split comparison truth
  - machine-readable per-sample score artifacts on the shared local `CIFAR-10 DDPM` surface
- current repo still lacks:
  - `P/U/control/test` collection schema
  - set-level significance pipeline
  - first audit-summary artifact contract

Tasks:

- [x] `GB-42.1` freeze one honest local target family and initial score source
- [x] `GB-42.2` define the minimum artifact schema for a first `CDI`-style internal audit canary
- [x] `GB-42.3` record explicit no-go triggers and the next contract-first follow-up task

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-protocol-asset-contract.md`

Selection verdict:

- `GB-42` now closes as `positive but bounded`
- the repo is ready for a real `CDI` contract-first lane
- but only as:
  - `gray-box collection-level audit extension`
  - `CPU-only`
  - `paper-inspired, not paper-faithful`
- the selected local surface is:
  - `CIFAR-10 DDPM` shared-score contract
- the selected initial signal source is:
  - existing gray-box per-sample score artifacts, starting from `SecMI stat` and optionally paired `PIA`
- the next honest live task is:
  - `CDI feature / collection-surface review`
- `gpu_release = none`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-43` CDI feature / collection-surface review

Goal: decide which already-landed per-sample artifact surface is honest for the first bounded `CDI` canary, how to construct one local `P/U` collection split, and whether the first canary should be one-method or paired-method

Current read:

- `GB-42` already froze `CDI` onto the shared local `CIFAR-10 DDPM` score contract
- current repo already has:
  - `PIA 1024 adaptive` score export with indices
  - `SecMI` per-sample score export on the same bounded disagreement surface
  - alignment evidence from `disagreement_analysis.json`
- the remaining question is therefore no longer feasibility, but which feature/collection surface is the cleanest first canary

Tasks:

- [x] `GB-43.1` audit the currently reusable per-sample score artifacts
- [x] `GB-43.2` freeze one bounded local `P/U/control/test` collection split
- [x] `GB-43.3` decide whether the first canary should be `SecMI-only` or paired `PIA + SecMI`

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-feature-collection-surface-review.md`

Selection verdict:

- `GB-43` now closes as `positive but bounded`
- the repo already supports one honest bounded internal `CDI` canary
- the frozen first canary is:
  - shared local `CIFAR-10 DDPM` surface
  - deterministic `512 / 512` control-test partition inside the existing `1024 / 1024` member/non-member surface
  - `SecMI stat only`
- paired `PIA + SecMI` remains the next extension rather than the first default
- `next_gpu_candidate` is now:
  - larger `PIA` shared-score surface refresh for later paired-method `CDI` follow-up
- `gpu_release = none` for the first CDI canary itself

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-44` CDI internal canary execution

Goal: execute the frozen first bounded `CDI` canary on existing gray-box score artifacts and decide whether the current repo can already produce one honest internal `audit_summary` object

Current read:

- `GB-43` already froze the first canary shape to:
  - shared local `CIFAR-10 DDPM` surface
  - deterministic `512 / 512` control-test partition
  - `SecMI stat only`
- the next honest question is now execution truth, not more contract prose

Tasks:

- [x] `GB-44.1` add a bounded internal `CDI` canary entrypoint plus tests
- [x] `GB-44.2` execute one real canary on current `SecMI` score artifacts
- [x] `GB-44.3` decide whether the resulting `audit_summary` is good enough to keep the lane alive

Canonical evidence anchor:

- `workspaces/gray-box/runs/cdi-internal-canary-20260416-r1/audit_summary.json`
- `workspaces/gray-box/2026-04-16-cdi-internal-canary-verdict.md`

Selection verdict:

- `GB-44` now closes as `positive but bounded`
- the repo can now execute one real internal `CDI`-shape canary on already-landed gray-box score artifacts
- the first canary stays:
  - `SecMI stat only`
  - internal audit-shape check only
- one real implementation gap was also closed honestly:
  - raw `SecMI` score direction needed memberness normalization before the one-sided Welch test
- `gpu_release = none` for this canary itself
- the active GPU follow-up remains:
  - `PIA 2048 shared-score surface refresh for CDI paired follow-up`

Value: ⭐⭐⭐
Budget: CPU-only

#### ⬜ `GB-45` PIA 2048 CDI rung runtime-health review

Goal: decide whether the currently active `PIA 2048 shared-score surface` GPU rung is still healthy enough to keep alive for the `CDI` lane, or whether it has already become a wasteful silent run

Current read:

- the active rung is:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1`
- the same `diffaudit-research` Python process remains alive and keeps occupying `cuda:0`
- GPU utilization and memory use still indicate real compute activity
- but the run has not emitted first-wave artifacts yet, so it cannot be left on indefinite patience

Tasks:

- [x] `GB-45.1` compare the active rung against the known `PIA 1024 adaptive` runtime reference
- [x] `GB-45.2` decide whether the active rung should continue or be stopped immediately
- [x] `GB-45.3` record an explicit runtime-health cap instead of leaving the GPU task open-ended

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-pia-2048-cdi-rung-runtime-health-review.md`

Selection verdict:

- `GB-45` now closes as `positive but bounded`
- keep the active `PIA 2048` GPU rung alive for now
- but only under an explicit runtime-health cap
- if the rung still fails to emit first-wave artifacts after roughly `40` minutes wall-clock from launch, stop it and record a blocker verdict rather than continuing indefinite silent burn

Value: ⭐⭐
Budget: CPU-only governance review

#### ⬜ `GB-46` PIA 2048 CDI rung verdict

Goal: decide whether the completed `PIA 2048 shared-score surface` rung should be retained as a useful `CDI` paired-feature surface, or classified as too expensive to matter

Current read:

- the rung completed and emitted:
  - `summary.json`
  - `scores.json`
  - `adaptive-scores.json`
- compared with the `1024` reference rung, the signal remains alive but cost grew much more sharply than the cleanest linear expectation

Tasks:

- [x] `GB-46.1` compare `2048` against the `1024 adaptive` reference on signal quality
- [x] `GB-46.2` compare runtime cost and decide whether the rung remains worth keeping
- [x] `GB-46.3` freeze the next GPU move for the `CDI` lane

Canonical evidence anchor:

- `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1/summary.json`
- `workspaces/gray-box/2026-04-16-pia-2048-cdi-rung-verdict.md`

Selection verdict:

- `GB-46` now closes as `positive but cost-heavy`
- the new `2048` rung is worth retaining as a reusable `PIA` shared-score surface for `CDI`
- but it does not justify more same-family `PIA` scaling by itself
- the next highest-value GPU move is now:
  - bounded `SecMI` export / disagreement analysis on the same `2048` subset, using this new `PIA` score surface

Value: ⭐⭐⭐
Budget: one bounded GPU rung completed

#### ⬜ `GB-47` SecMI-PIA 2048 paired-surface verdict

Goal: decide whether the widened `SecMI` export on the same `2048` subset is strong enough to support immediate `CDI paired-feature extension`

Current read:

- the new `PIA 2048` surface is already landed
- the next shortest GPU move was one bounded `SecMI` export / disagreement run on the same `2048` subset
- this task decides whether that paired surface is stable enough for promotion, or only useful as a warning signal

Tasks:

- [x] `GB-47.1` execute the widened `SecMI` export / disagreement run on the same `2048` subset
- [x] `GB-47.2` compare it against the earlier `1024` paired surface
- [x] `GB-47.3` decide whether paired `CDI` can be promoted immediately

Canonical evidence anchor:

- `workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/summary.json`
- `workspaces/gray-box/2026-04-16-secmi-pia-2048-paired-surface-verdict.md`

Selection verdict:

- `GB-47` now closes as `mixed but useful`
- the widened paired surface is execution-positive
- but it is not stable enough for immediate paired `CDI` promotion
- `SecMI stat` drops sharply on the `2048` surface, while cross-method agreement also weakens materially

Value: ⭐⭐⭐
Budget: one bounded GPU rung completed

#### ⬜ `GB-48` CDI paired-feature extension review

Goal: decide whether the `CDI` lane should now promote into paired `PIA + SecMI` feature scoring, or whether the new `2048` mismatch should be reviewed first

Current read:

- `GB-44` already proved the first `CDI` canary is real
- `GB-46` already landed a larger reusable `PIA` surface
- `GB-47` now says the matching `SecMI 2048` surface is not yet promotion-grade

Tasks:

- [x] `GB-48.1` compare the value of immediate paired-feature promotion against the mismatch risk
- [x] `GB-48.2` decide whether the lane should promote, pause, or fall back
- [x] `GB-48.3` freeze the next live task

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-paired-feature-extension-review.md`

Selection verdict:

- `GB-48` now closes as `negative but useful`
- do not promote the lane yet into paired `PIA + SecMI` feature scoring
- keep `CDI` first canary as landed truth
- treat paired extension as review-required rather than promotion-ready
- `gpu_release = none`
- the next live task is:
  - `CDI paired-surface mismatch review`

Value: ⭐⭐
Budget: CPU-only review

#### ⬜ `GB-49` CDI paired-surface mismatch review

Goal: decide whether the weak `SecMI 2048` paired surface reflects real scale degradation, subset mismatch, or export/config drift

Current read:

- the widened `2048` paired surface already failed the promotion gate
- the next honest question is diagnostic, not promotional
- this task must explain whether the current weak packet should be preserved as real method truth or isolated as a contract bug

Tasks:

- [x] `GB-49.1` compare the old strong `1024` packet against the new `2048` packet at prefix/suffix level
- [x] `GB-49.2` audit whether the new `SecMI` export changed score contract or runtime parameters
- [x] `GB-49.3` freeze the next post-review lane and GPU posture

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-paired-surface-mismatch-review.md`

Selection verdict:

- `GB-49` now closes as `negative but clarifying`
- the current mismatch is best explained by `export/config drift`, not by proven `SecMI` scale collapse
- the strongest concrete drift is:
  - old strong references used `t_sec = 100`
  - the weak `2048` export used `t_sec = 20`
- keep `1024` as the last aligned paired reference
- keep the weak `2048` packet as mismatch truth, not promotion truth
- `gpu_release = none`
- the next live task is:
  - `SecMI paired-surface repair contract review`

Value: ⭐⭐⭐
Budget: CPU-only review

#### ⬜ `GB-50` SecMI paired-surface repair contract review

Goal: freeze the exact `SecMI` contract that a repaired paired-surface export must satisfy before another `2048` rerun or `CDI` promotion decision

Current read:

- `GB-49` already showed the weak `2048` packet is drift-heavy
- the next useful task is to convert that diagnosis into a concrete repair contract
- that contract should be strict enough to justify one bounded rerun, but not vague enough to reopen arbitrary GPU spend

Tasks:

- [x] `GB-50.1` compare the admitted `SecMI` mainline contract against old strong paired `1024` evidence
- [x] `GB-50.2` rule in or rule out split-root drift
- [x] `GB-50.3` freeze the repaired paired-export contract and next GPU gate

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-secmi-paired-surface-repair-contract-review.md`

Selection verdict:

- `GB-50` now closes as `positive`
- the repaired paired export must realign to the admitted `SecMI stat` contract:
  - `t_sec = 100`
  - `timestep = 10`
  - `batch_size = 64`
  - canonical split root = `external/SecMI/mia_evals/member_splits`
- split drift is not the blocker:
  - the `SecMI` and `PIA` CIFAR-10 half-split files are byte-identical
- one bounded GPU rerun is now justified
- `next_gpu_candidate = repaired SecMI 2048 paired-surface rerun`

Value: ⭐⭐⭐
Budget: CPU-only review plus script hardening

#### ⬜ `GB-51` SecMI-PIA 2048 repaired paired-surface rerun

Goal: rerun the weak `2048` paired surface under the repaired `SecMI` contract and decide whether the paired reference is actually recoverable

Current read:

- `GB-50` already froze the repair contract
- the next honest GPU question is just one bounded rerun under that repaired contract

Tasks:

- [x] `GB-51.1` run repaired `SecMI 2048` paired export with the admitted `SecMI stat` contract
- [x] `GB-51.2` compare repaired `2048` metrics against weak `r2` and strong `1024` reference
- [x] `GB-51.3` decide whether repaired `2048` becomes the active paired reference

Canonical evidence anchor:

- `workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r3/summary.json`
- `workspaces/gray-box/2026-04-16-secmi-pia-2048-repaired-paired-surface-verdict.md`

Selection verdict:

- `GB-51` now closes as `positive`
- repaired `2048` paired surface recovered cleanly:
  - `secmi_stat_auc = 0.876912`
  - `combined_spearman = 0.906879`
  - `disagreement_rate = 0.121582`
- the old weak `r2` packet is now mismatch-history evidence, not active paired truth

Value: ⭐⭐⭐
Budget: one bounded GPU rerun completed

#### ⬜ `GB-52` CDI paired-feature re-promotion review

Goal: decide whether the repaired `2048` paired surface is now strong enough to reopen paired `PIA + SecMI` feature promotion for the `CDI` lane

Current read:

- the original promotion stop was correct under the weak drifted packet
- the repaired `2048` packet now restores the paired surface

Tasks:

- [x] `GB-52.1` review whether the original promotion blocker has been removed
- [x] `GB-52.2` decide whether paired feature promotion can reopen
- [x] `GB-52.3` freeze the next CPU lane

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-paired-feature-repromotion-review.md`

Selection verdict:

- `GB-52` now closes as `positive but bounded`
- paired `PIA + SecMI` feature promotion may now reopen
- the landed `SecMI-only` first canary remains valid history and is not replaced retroactively
- `gpu_release = none`
- the next live task is:
  - `CDI paired-feature scorer design`

Value: ⭐⭐
Budget: CPU-only review

#### ⬜ `GB-53` CDI paired-feature scorer design

Goal: implement and validate the smallest honest paired `CDI` scorer on the repaired `2048` shared surface

Current read:

- paired `PIA + SecMI` promotion has already reopened
- the next real task is no longer “whether paired scoring is allowed”
- it is “what exact paired scorer can be frozen without hiding behind a learned black box”

Tasks:

- [x] `GB-53.1` design a control-fitted paired scorer that remains auditable
- [x] `GB-53.2` implement the scorer in the current internal canary path
- [x] `GB-53.3` validate it on the repaired `2048` paired surface

Canonical evidence anchor:

- `workspaces/gray-box/runs/cdi-paired-canary-20260416-r1/audit_summary.json`
- `workspaces/gray-box/2026-04-16-cdi-paired-feature-scorer-design.md`

Selection verdict:

- `GB-53` now closes as `positive but bounded`
- the frozen scorer is:
  - `control-z-linear`
  - `SecMI stat + PIA`
  - control-fitted z-standardization plus positive-gap linear weights
- bounded validation is positive:
  - `paired_t = 30.027926`
  - `secmi_t = 29.637878`
  - `pia_t = 27.141034`
- `gpu_release = none`
- the next live task is:
  - `CDI paired-scorer boundary review`

Value: ⭐⭐⭐
Budget: CPU-only implementation and validation

#### ⬜ `GB-54` CDI paired-scorer boundary review

Goal: decide whether the new paired scorer should remain a one-off internal canary artifact or become the default internal paired scorer for the repaired `2048` shared surface

Current read:

- `GB-53` already established one bounded positive paired scorer
- the missing question is no longer implementation
- it is boundary:
  - fluke vs stable
  - internal default vs canary-only
  - useful scorer vs headline overclaim

Tasks:

- [x] `GB-54.1` check whether the scorer remains positive under reversed half-split orientation
- [x] `GB-54.2` compare paired strength against the component features honestly
- [x] `GB-54.3` freeze the current internal-use boundary

Canonical evidence anchor:

- `workspaces/gray-box/runs/cdi-paired-canary-20260416-r2-reverse/audit_summary.json`
- `workspaces/gray-box/2026-04-16-cdi-paired-scorer-boundary-review.md`

Selection verdict:

- `GB-54` now closes as `positive but constrained`
- the paired scorer is not a one-split fluke:
  - reverse split remains strongly positive
  - weights stay stable near `SecMI 0.52 / PIA 0.48`
- but it is not a clean dominance scorer over `SecMI`
- current honest boundary is:
  - `default internal paired scorer for CDI on the repaired 2048 surface`
  - not a headline scorer
  - not external copyright-grade evidence
- `gpu_release = none`
- the next live task is:
  - `CDI paired-scorer summary-layer sync`

Value: ⭐⭐⭐
Budget: CPU-only review

#### ⬜ `GB-55` CDI paired-scorer summary-layer sync

Goal: update summary-layer entry points so new sessions and higher-layer materials reflect the frozen `CDI` paired-scorer boundary

Current read:

- `GB-54` already froze the current scorer boundary
- but key summary-layer entry points still lag behind that boundary or omit `CDI` entirely

Tasks:

- [x] `GB-55.1` update gray-box README wording
- [x] `GB-55.2` update Research README summary wording
- [x] `GB-55.3` update comprehensive progress gray-box wording

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-paired-scorer-summary-layer-sync.md`

Selection verdict:

- `GB-55` now closes as `positive`
- summary-layer entry points now reflect the current `CDI` truth:
  - first internal canary landed
  - repaired paired `2048` surface landed
  - `control-z-linear` = default internal paired scorer
  - not headline scorer
  - not external evidence
- `competition_material_sync = recommended`
- `gpu_release = none`
- the next live task is:
  - `CDI paired-scorer default-run policy note`

Value: ⭐⭐
Budget: CPU-only sync

#### ⬜ `GB-56` CDI paired-scorer default-run policy note

Goal: freeze a short execution policy so later internal `CDI` runs do not keep renegotiating when paired scoring should auto-enable and what must be reported together

Current read:

- `GB-55` already synchronized the summary layer
- but execution policy was still implicit in lane-local notes and manual CLI choices

Tasks:

- [x] `GB-56.1` define the default paired-scorer policy
- [x] `GB-56.2` align the canary script default behavior with that policy
- [x] `GB-56.3` ensure the policy preserves component-stat reporting and non-headline boundary

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-16-cdi-paired-scorer-default-run-policy-note.md`

Selection verdict:

- `GB-56` now closes as `positive`
- default policy is now:
  - `--paired-scorer auto`
- meaning:
  - paired inputs present -> use `control-z-linear`
  - paired inputs absent -> stay on `SecMI-only`
- whenever paired scoring is enabled, always report together:
  - paired statistic
  - `SecMI` component statistic
  - `PIA` component statistic
- `gpu_release = none`
- the next live task is:
  - `CDI paired-scorer machine-readable contract note`

Value: ⭐⭐
Budget: CPU-only policy sync

#### ⬜ `GB-57` CDI paired-scorer machine-readable contract note

Goal: freeze the machine-readable field contract that higher-layer consumers should rely on when reading internal paired `CDI` canary outputs

Current read:

- `GB-56` already froze the execution policy
- the next missing layer is machine-readable consumption:
  - which fields are required
  - which policy values mean paired mode is active
  - which boundary flags forbid headline or external use

Tasks:

- [x] `GB-57.1` define the machine-readable contract fields
- [x] `GB-57.2` emit those fields in a fresh paired canary artifact
- [x] `GB-57.3` freeze the higher-layer consumption rule

Canonical evidence anchor:

- `workspaces/gray-box/runs/cdi-paired-canary-20260417-r3-contract/audit_summary.json`
- `workspaces/gray-box/2026-04-17-cdi-paired-scorer-machine-readable-contract-note.md`

Selection verdict:

- `GB-57` now closes as `positive`
- internal paired `CDI` runs now emit a stable machine-readable contract:
  - `contract.name`
  - `contract.version`
  - `contract.feature_mode`
  - `contract.paired_scorer_policy_requested`
  - `contract.paired_scorer_policy_effective`
  - `contract.component_reporting_required`
  - `contract.headline_use_allowed`
  - `contract.external_evidence_allowed`
- higher-layer consumers can now distinguish paired mode from component-only mode without re-reading lane-local notes
- `gpu_release = none`
- the next live task is:
  - `CDI paired-scorer consumer handoff note`

Value: ⭐⭐
Budget: CPU-only contract sync

#### ⬜ `GB-58` CDI paired-scorer consumer handoff note

Goal: freeze what higher-layer consumers should read first from paired `CDI` `audit_summary.json`, what is safe for summary/materials use, and what must remain internal-only diagnostic detail

Current read:

- `GB-57` already froze the machine-readable contract fields
- the remaining missing layer is consumer behavior:
  - which fields `Leader/materials` should read first
  - which fields future `Platform/Runtime` consumers should hard-gate on
  - which metrics or scorer details must not be overinterpreted as headline evidence

Tasks:

- [x] `GB-58.1` define consumer-specific read priority
- [x] `GB-58.2` define summary/materials-safe field subset
- [x] `GB-58.3` freeze anti-overclaim rules for future internal consumers

Canonical evidence anchor:

- `workspaces/gray-box/runs/cdi-paired-canary-20260417-r3-contract/audit_summary.json`
- `workspaces/gray-box/2026-04-17-cdi-paired-scorer-consumer-handoff-note.md`

Selection verdict:

- `GB-58` now closes as `positive`
- higher-layer consumers now have an explicit read order:
  - `contract`
  - `feature_mode`
  - `metrics`
  - `notes`
  - `analysis`
- summary/materials-safe consumption is limited to:
  - `contract.name`
  - `contract.version`
  - `contract.feature_mode`
  - `contract.paired_scorer_policy_effective`
  - `contract.component_reporting_required`
  - `contract.headline_use_allowed`
  - `contract.external_evidence_allowed`
  - `notes`
- future `Platform/Runtime` consumers should hard-gate on the contract flags and treat scorer weights / centers / scales / control gaps as diagnostic-only
- this handoff explicitly forbids over-reading:
  - `paired_t > secmi_t` as a new headline
  - local fitted weights as importance or portability claims
  - artifact paths or runtime duration as semantic evidence
- `gpu_release = none`
- the next live task is:
  - `gray-box post-CDI lane reselection review`

Value: ⭐⭐
Budget: CPU-only handoff sync

#### ⬜ `GB-59` Gray-box post-CDI lane reselection review

Goal: decide which gray-box lane is now most worth opening after the `CDI` paired-scorer boundary and consumer handoff are both closed

Current read:

- `CDI` is now frozen as an internal collection-level audit extension
- `TMIA-DM + temporal-striding` is already the strongest defended challenger reference
- `MoFit` is on current-contract hold
- `SimA` and `structural memorization` remain below reopen threshold
- `Noise as a Probe` is the strongest unpromoted genuinely new gray-box mechanism still carrying an unresolved promotion/comparability question

Tasks:

- [x] `GB-59.1` compare post-`CDI` candidate lanes against current gray-box story needs
- [x] `GB-59.2` select one next live CPU-first lane and reject weaker reopens for now
- [x] `GB-59.3` freeze GPU posture and immediate task shape for the selected lane

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-graybox-post-cdi-lane-reselection-review.md`

Selection verdict:

- `GB-59` now closes as `positive`
- gray-box still has `gpu_release = none`
- the selected next live lane is:
  - `Noise as a Probe promotion-gap review`
- reason:
  - `Noise as a Probe` is now the strongest unpromoted new gray-box mechanism
  - defended extension on the current latent-diffusion contract already closed as `no-go for now`
  - so the highest-value remaining CPU-side question is whether the branch has any honest promotion/comparability path at all
- rejected for now:
  - `TMIA-DM` packaging reopen
  - `SimA reopen`
  - `MoFit reopen`
  - `structural memorization reopen`
- `next_gpu_candidate = none`
- the next live task is:
  - `Noise as a Probe promotion-gap review`

Value: ⭐⭐
Budget: CPU-only reselection

#### ⬜ `GB-60` Noise as a Probe promotion-gap review

Goal: decide what exact gap still blocks `Noise as a Probe` from promotion into the packaged gray-box challenger layer, and whether any honest promotion path exists on the current latent-diffusion contract

Current read:

- `Noise as a Probe` already has repeat-positive bounded evidence
- it is already mentionable above lane-local notes
- defended extension already closed as `no-go for now`
- the remaining unresolved question is therefore no longer execution, but comparability and promotion boundary

Tasks:

- [x] `GB-60.1` define the exact blocker between bounded candidate and packaged challenger
- [x] `GB-60.2` decide whether the current local contract supports any honest direct promotion path
- [x] `GB-60.3` freeze the next CPU-side lane if promotion is still blocked

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-noise-as-probe-promotion-gap-review.md`

Selection verdict:

- `GB-60` now closes as `negative but clarifying`
- current blocker is:
  - `same-surface comparability`
- `Noise as a Probe` remains:
  - `strengthened bounded challenger candidate`
  - `mentionable`
  - `not promotion-ready`
- defended extension is confirmed not to be the missing shortcut
- current local latent-diffusion contract has:
  - `no honest direct promotion path`
- `gpu_release = none`
- `next_gpu_candidate = none`
- the next live task is:
  - `Noise as a Probe contract-shift review`

Value: ⭐⭐
Budget: CPU-only review

#### ⬜ `GB-61` Noise as a Probe contract-shift review

Goal: decide whether it is worth establishing a separate latent-diffusion same-surface comparison contract for `Noise as a Probe`, or whether that would still create low-value pseudo-comparability

Current read:

- `Noise as a Probe` is the only latent-diffusion branch on its current surface with repeat-positive bounded signal
- `MoFit` on the same broad latent-diffusion target-family direction is still `current-contract hold`
- `structural memorization` on the same broad surface is still direction-negative
- so the remaining question is whether a latent-diffusion auxiliary board would create real leverage or only more bookkeeping

Tasks:

- [x] `GB-61.1` audit the smallest honest latent-diffusion same-surface board candidate
- [x] `GB-61.2` decide whether current same-surface evidence is strong enough to justify that board
- [x] `GB-61.3` freeze the next post-review lane and GPU posture

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-noise-as-probe-contract-shift-review.md`

Selection verdict:

- `GB-61` now closes as `negative but clarifying`
- do **not** build the latent-diffusion same-surface board yet
- reason:
  - only `Noise as a Probe` is currently repeat-positive on that surface
  - `MoFit` is still tiny weak-positive under hold
  - `structural memorization` is still direction-negative
  - and their score semantics remain too heterogeneous for a strong packaged comparator board
- `gpu_release = none`
- `next_gpu_candidate = none`
- the next live task is:
  - `gray-box post-noise contract-shift reselection review`

Value: ⭐⭐
Budget: CPU-only review

#### ⬜ `GB-62` Gray-box post-noise contract-shift reselection review

Goal: decide whether gray-box should still keep the next live CPU-first slot after the `Noise as a Probe` promotion and contract-shift questions both closed negatively

Current read:

- gray-box headline / defended challenger / corroboration / `CDI` extension are all already stable
- `Noise as a Probe` no longer has either:
  - an honest direct promotion path, or
  - an honest latent-diffusion same-surface board worth building now
- `MoFit` remains hold and `structural memorization` remains negative
- so the remaining question is lane priority, not another gray-box sub-branch

Tasks:

- [x] `GB-62.1` compare gray-box keep-going vs lane-yield options
- [x] `GB-62.2` decide whether gray-box still owns the next live CPU-first slot
- [x] `GB-62.3` freeze the next lane and GPU posture

Canonical evidence anchor:

- `workspaces/gray-box/2026-04-17-graybox-post-noise-contract-shift-reselection-review.md`

Selection verdict:

- `GB-62` now closes as `negative but clarifying`
- gray-box should now yield the next live CPU-first slot
- reason:
  - current gray-box mainline truth is stable
  - latent-diffusion side branches no longer expose a strong immediate next question
  - further immediate gray-box reselection would be low-value churn
- `gpu_release = none`
- `next_gpu_candidate = none`
- the next live task is:
  - `white-box post-breadth next-hypothesis selection review`

Value: ⭐⭐
Budget: CPU-only reselection

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

#### ⬜ `WB-19` White-box post-breadth next-hypothesis selection review

Goal: decide whether white-box still contains one honest next-hypothesis lane worth taking the next live CPU-first slot after breadth closed and `DP-LoRA` stabilized as a bounded branch

Current read:

- `WB-3` already closed as `none selected / not-requestable`
- `Finding NeMo` remains `zero-GPU hold`
- `GSA2` remains same-family corroboration rather than a distinct second line
- `DP-LoRA` remains alive but already entered `no-new-gpu-question`

Tasks:

- [x] `WB-19.1` compare `DP-LoRA`, `Finding NeMo`, and `GSA2` against current white-box needs
- [x] `WB-19.2` decide whether white-box still owns the next live CPU-first slot
- [x] `WB-19.3` freeze the next lane and GPU posture

Canonical evidence anchor:

- `workspaces/white-box/2026-04-17-whitebox-post-breadth-next-hypothesis-selection-review.md`

Selection verdict:

- `WB-19` now closes as `negative but clarifying`
- white-box currently has no honest immediate next-hypothesis execution lane
- reason:
  - `DP-LoRA` is bounded but not newly opened
  - `Finding NeMo` remains not-requestable
  - `GSA2` remains same-family corroboration only
- `gpu_release = none`
- `next_gpu_candidate = none`
- the next live task is:
  - `cross-box closure-round system sync review`

Value: ⭐⭐
Budget: CPU-only selection review

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

## 8. Near-Term Priority Ladder

This is a preference order, not a prison.

### Top now

`GB-66` and `GB-67` are now closed.

No immediate GPU lane should be opened until a new bounded comparison or defended-extension question is selected.

Current release posture:

- `gpu_release = none`
- `next_gpu_candidate = none`
- `gray-box` should now yield the next live CPU-first slot after the switching closure round

### Next

1. ✅ `GB-67` gray-box post-switch lane reselection review
2. ✅ `GB-66` PIA vs TMIA-DM confidence-gated switching offline packet
3. ✅ `GB-65` PIA vs TMIA-DM confidence-gated switching design review
4. ✅ `GB-64` ranking-sensitive variable search review
5. ✅ `WB-20` distinct white-box defended-family import / selection review
6. ✅ `GB-63` second gray-box defense mechanism selection review
7. ✅ `BB-8` black-box next-family candidate-generation refresh review
8. ✅ `X-9` cross-box closure-round system sync review
8. ✅ `WB-13` DP-LoRA comparator admission packet refresh
9. ✅ `WB-12` DP-LoRA comparator release-review refresh
10. ✅ `WB-11` DP-LoRA local comparator board refresh verdict
11. ✅ `WB-10` DP-LoRA W-1 local-surface refresh feasibility
12. ✅ `WB-9` DP-LoRA comparator schema-alignment contract
13. ✅ `WB-8` DP-LoRA comparator artifact-board preflight
14. ✅ `WB-7` DP-LoRA comparator contract reconciliation
15. ✅ `WB-6` DP-LoRA comparator release review
16. ✅ `GB-17` Noise-as-a-Probe defended-extension feasibility review
17. ✅ `GB-16` Noise-as-a-Probe summary-layer sync
18. ✅ `GB-15` Noise-as-a-Probe challenger-boundary review
19. ✅ `GB-14` Noise-as-a-Probe larger-rung repeat
20. ✅ `GB-13` Noise-as-a-Probe larger bounded rung
21. ✅ `GB-12` Noise-as-a-Probe threshold hardening
22. ✅ `GB-11` Noise-as-a-Probe expansion repeat
23. ✅ `GB-10` Noise-as-a-Probe first expansion rung
24. ✅ `GB-9` Noise-as-a-Probe calibration / expansion policy
25. ✅ `GB-8` Noise-as-a-Probe canary scaffold
26. ✅ `GB-7` Noise-as-a-Probe implementation-surface review
27. ✅ `GB-6` Noise-as-a-Probe protocol / asset contract
28. ✅ `GB-5` genuinely-new-family selector
29. ✅ `WB-5` DP-LoRA comparability dossier
30. ✅ `BB-6` same-protocol cross-method score package
31. ✅ `WB-3` white-box defense breadth
32. ✅ `GB-1` second gray-box defense
33. ✅ `BB-1` second-signal black-box expansion
34. ✅ `INF-2` research automation health
35. ✅ `INF-3` subagent leverage experiments
36. ✅ `WB-4` white-box feature/trajectory upgrade
37. ✅ `X-3` system-consumable sync
38. ✅ `BB-3` CLiD boundary-quality upgrade
39. ✅ `X-4` cross-box exploration lane

### Then

16. ✅ `WB-2` second white-box verdict
17. ✅ `GB-3` new gray-box family
18. ✅ `BB-4` mitigation-aware black-box evaluation

---

## 9. Success Conditions

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

## 10. Changelog

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
| 2026-04-16 15:00 | Closed `X-7` positively: the highest-value `Phase E` summary docs are now aligned to the sparse-hold registry; `PIA paper-aligned confirmation` remains document-layer conditional only, `Finding NeMo` is the only remaining intake-only candidate under zero-GPU hold, and the old `DP-LoRA / SecMI unblock / TMIA-DM intake` candidate-ordering wording has been removed in favor of current lane truth |
| 2026-04-16 15:10 | Closed `X-8` as `negative but stabilizing`: the sparse registry does not by itself justify opening the separate `Finding NeMo` hypothesis/budget review; `Finding NeMo` remains `adapter-complete zero-GPU hold`, and the next honest live-lane search should prefer new bounded candidate generation or another lane rather than automatic reopen |
| 2026-04-16 15:25 | Closed `GB-19` positively: after current live branches all contracted, the next honest live CPU-first lane is `GB-20 MoFit protocol / asset contract`; black-box reopen remains too blocked, `Finding NeMo` remains on zero-GPU hold, and `MoFit` is now the cleanest remaining genuinely new gray-box mechanism to evaluate next |
| 2026-04-16 15:35 | Closed `GB-20` as `positive but bounded`: `MoFit` now has an honest first local contract on `SD1.5 + celeba_partial_target/checkpoint-25000` with `BLIP`/cached-caption bootstrap, but the repo still lacks frozen surrogate optimization, fitted-embedding optimization, and artifact schema; the next live CPU-first step is implementation-surface review, and `gpu_release` remains `none` |
| 2026-04-16 15:45 | Closed `GB-21` as `positive but bounded`: `MoFit` should start from a dedicated scaffold rather than overloading `structural memorization` or `semantic-aux`; the repo already has caption bootstrap and latent-diffusion substrate, but still lacks surrogate optimization, fitted-embedding optimization, and `L_MoFit`-style schema, so `gpu_release` remains `none` |
| 2026-04-16 15:55 | Closed `GB-22` positively: `MoFit` now has a frozen dedicated scaffold choice (`scripts/run_mofit_interface_canary.py`) plus a minimum artifact schema (`summary.json`, `records.jsonl`, surrogate/embedding traces); the next live CPU-first step is to implement that script, and `gpu_release` remains `none` |
| 2026-04-16 16:10 | Closed `GB-23` as `positive but bounded`: the frozen `MoFit` dedicated scaffold is now real code and passes both TDD-style unit verification and a fresh script execution check, producing `summary.json`, `records.jsonl`, and trace directories; the lane still stays below smoke because surrogate optimization, fitted-embedding optimization, and real `L_MoFit` scoring are not yet implemented |
| 2026-04-16 16:20 | Closed `GB-24` as `positive but bounded`: the `MoFit` scaffold now supports per-sample record append and trace placeholder creation, and `records.jsonl` now carries `l_cond / l_uncond / mofit_score` as frozen placeholder fields; the lane still remains below smoke because those values are not yet produced by actual optimization loops |
| 2026-04-16 16:30 | Closed `GB-25` as `positive but bounded`: the `MoFit` scaffold now has a real score/trace update path, so future optimization loops can write step traces and `l_cond / l_uncond / mofit_score` back into the frozen schema; the lane still remains below smoke because the actual optimization loops are not yet implemented |
| 2026-04-16 16:40 | Closed `GB-26` as `positive but bounded`: the `MoFit` lane now has minimal reusable surrogate and embedding optimization helpers, verified on toy losses with fresh tests; the next step is to wire those helpers into the real latent-diffusion target path so `L_cond / L_uncond / mofit_score` become target-model-derived rather than placeholder values |
| 2026-04-16 16:50 | Closed `GB-27` as `positive but bounded`: the `MoFit` lane now has a real latent-path loss contract in code, so future optimization loops can target `L_cond / L_uncond / mofit_score` consistently; the next step is to wire that contract into the actual SD1.5 target-model path rather than toy differentiable predictors |
| 2026-04-16 17:05 | Closed `GB-28` as `positive but bounded`: the `MoFit` lane now has a real helper-layer bridge onto the `SD1.5` target-model path, including `UNet(...).sample` predictor adaptation and guided target-noise construction, both verified by fresh TDD-style tests; the lane still remains below smoke because caption bootstrap, sample-level record execution, and end-to-end optimization loops are not yet assembled |
| 2026-04-16 17:20 | Closed `GB-29` as `positive but bounded`: the `MoFit` lane now has a real sample-level helper path that resolves prompts from metadata or bounded fallback, appends/finalizes records, runs surrogate and embedding optimization, and writes traces plus `L_cond / L_uncond / mofit_score` back into schema; the lane still remains below smoke because this helper is not yet mounted into a script-level run over actual local assets and model-loading substrate |
| 2026-04-16 17:30 | Closed `GB-30` as `positive but bounded`: the script-level execution surface is now frozen; the next honest path is to extend `run_mofit_interface_canary.py` rather than create another `MoFit` script, and to reuse the structural-memorization substrate for row loading, caption bootstrap, prompt encoding, image-to-latent encoding, and `UNet` target-path calls |
| 2026-04-16 17:45 | Closed `GB-31` as `positive but bounded`: `run_mofit_interface_canary.py` now owns bounded row loading, split execution, runner-side mounting of the sample-level helper, and summary-state upgrade from `scaffold_only` to `canary_executed`, all verified by fresh script-level TDD plus a full `MoFit` regression sweep; the lane still remains below smoke because this path has not yet been freshly launched on the admitted local asset stack |
| 2026-04-16 17:55 | Closed `GB-32` as `hold-before-launch`: after the script-level canary became executable, the launch gate review concluded that the current first-run CPU budget is still too wide for an honest first admitted local launch; the next live task should tighten launch defaults or add an explicit bounded launch profile before spending real local runtime |
| 2026-04-16 18:10 | Closed `GB-33` positively: the `MoFit` canary now defaults to a bounded CPU-first launch profile (`member=1 / nonmember=1 / surrogate=1 / embedding=2 / cpu`), with the effective profile propagated through `run_canary` and `summary.json`, verified by fresh script-level TDD plus full `MoFit` regression sweep; the next honest live step is now one fresh real local CPU canary under that tightened profile |
| 2026-04-16 18:35 | Closed `GB-34` as `positive but bounded`: the first fresh admitted local CPU `MoFit` canary now exists on `SD1.5 + celeba_partial_target/checkpoint-25000` under the bounded CPU-first profile, after resolving two real execution blockers (`resolution` propagation and inference-tensor graph freezing); execution feasibility is now closed, but the observed score gap is still tiny |
| 2026-04-16 18:40 | Closed `GB-35` as `inconclusive but still alive`: the first fresh canary shows monotonic optimization traces but only tiny negative `mofit_score` gaps for both member and nonmember, so the family is not execution-dead but also not ready for direct rung expansion; the next live task should be a bounded CPU micro-rung design/review |
| 2026-04-16 18:45 | Closed `GB-36` positively: the next bounded CPU rung is now frozen to `2x2` rows with `2/4` optimization steps on CPU, plus explicit stop conditions and no-go triggers; the next honest live task is micro-rung execution rather than another open-ended design loop |
| 2026-04-16 19:05 | Closed `GB-37` as `positive but bounded`: after fixing the `cpu-micro-rung` profile semantics to exact `2x2 / 2+4 / cpu`, the first valid micro-rung now runs end to end on the admitted local stack and lands at `mofit-sd15-celeba-microrung-20260416-cpu-r2`; execution feasibility at this rung is now closed |
| 2026-04-16 19:10 | Closed `GB-38` as `weak-positive but still below promotion`: the valid micro-rung improves score direction relative to the canary, with member mean `-0.0019168` above nonmember mean `-0.0022683`, but the absolute gap (`0.0003515`) remains small; the next honest live task is one final bounded CPU review rung or a direct hold/no-go if that rung is rejected on cost grounds |
| 2026-04-16 20:10 | Closed `GB-39` as `weak-positive but still bounded`: after replaying the final bounded CPU review rung on the current environment (`mofit-sd15-celeba-reviewrung-20260416-cpu-r2`), the lane again shows member mean `-0.0039209` above nonmember mean `-0.0044675`, but the gap (`0.0005466`) remains too small for promotion or GPU release |
| 2026-04-16 20:15 | Closed `GB-40` as `current-contract hold`: with execution feasibility, micro-rung, and final review-rung all closed, the current `MoFit` contract is now documented as execution-positive but signal-weak; further mechanical CPU rung expansion is no longer justified, and reopen requires a materially changed contract rather than more of the same |
| 2026-04-16 20:35 | Closed `GB-41` positively: after `MoFit` moved to `current-contract hold` and black-box / white-box still remained at `gpu_release = none`, the next honest live lane is not a reopen of old branches but a new gray-box extension; `CDI` wins because it can reuse current `PIA / SecMI` score surfaces to open a collection-level audit direction without new GPU cost |
| 2026-04-16 20:50 | Closed `GB-42` as `positive but bounded`: the repo is now judged ready for a real `CDI` contract-first lane on the shared local `CIFAR-10 DDPM` score surface, but only as a CPU-only collection-level audit extension; the next live task should freeze reusable per-sample score inputs and the first honest `P/U` collection contract before any execution claim |
| 2026-04-16 21:10 | Closed `GB-43` as `positive but bounded`: the current repo already supports one bounded internal `CDI` canary on the existing `1024 / 1024` shared local surface; the first canary should stay `SecMI stat only` with a deterministic `512 / 512` control-test partition, while paired `PIA + SecMI` is pushed to the next extension and the most useful GPU follow-up becomes a larger `PIA` shared-score surface refresh |
| 2026-04-16 21:40 | Closed `GB-44` as `positive but bounded`: the first real internal `CDI` canary now runs on existing `SecMI` score artifacts and emits `collections.json + sample_scores.jsonl + audit_summary.json`; the resulting Welch test is strongly same-directional after explicit memberness normalization, so the lane now has execution truth rather than only a contract note |
| 2026-04-16 21:55 | Closed `GB-45` as `positive but bounded`: the active `PIA 2048` rung for the `CDI` lane is still genuinely alive on `cuda:0`, but it has remained artifact-silent longer than the cleanest naive runtime expectation; keep it alive for now, but only under an explicit `~40min from launch` runtime-health cap rather than indefinite silent waiting |
| 2026-04-16 22:00 | Closed `GB-46` as `positive but cost-heavy`: the `PIA 2048` rung finished successfully and preserved the gray-box signal (`AUC = 0.833109`, `ASR = 0.769043`), so it is worth keeping as the reusable `PIA` surface for `CDI` paired follow-up, but its `1723s` runtime is a real cost warning and does not justify more same-family `PIA` scaling by itself |
| 2026-04-16 22:15 | Closed `GB-47` as `mixed but useful`: the bounded `SecMI 2048` paired-surface export succeeded technically, but the resulting `stat` quality fell sharply (`AUC = 0.569096`) and cross-method agreement weakened materially, so the widened paired surface is not yet stable enough for immediate paired `CDI` promotion |
| 2026-04-16 22:20 | Closed `GB-48` as `negative but useful`: `CDI` now has a landed first canary plus a larger `PIA` surface, but the new `SecMI 2048` mismatch means the lane should not yet promote into paired `PIA + SecMI` feature scoring; the next honest move is a CPU-side mismatch review rather than another GPU escalation |
| 2026-04-16 22:35 | Closed `GB-49` as `negative but clarifying`: the new weak `SecMI 2048` paired packet is not primarily a scale-collapse result, because its own first `1024` prefix is already weak; the strongest current explanation is export/config drift, especially the jump from the admitted `SecMI` contract at `t_sec = 100` to the new export at `t_sec = 20`, so `gpu_release` stays `none` and the next live lane becomes `SecMI paired-surface repair contract review` |
| 2026-04-16 22:50 | Closed `GB-50` as `positive`: the repaired paired export contract is now frozen to the admitted `SecMI stat` mainline (`t_sec = 100`, `timestep = 10`, `batch_size = 64`, canonical split root under `external/SecMI/mia_evals/member_splits`), while split drift is explicitly ruled out because the `SecMI` and `PIA` CIFAR-10 half-split files are byte-identical; one bounded repaired `2048` rerun is now justified |
| 2026-04-16 23:05 | Closed `GB-51` as `positive`: the repaired `SecMI 2048` paired rerun restored the paired surface to the same quality regime as the old strong `1024` reference (`AUC = 0.876912`, `combined Spearman = 0.906879`, `disagreement = 0.121582`), so the weak `r2` packet is now demoted to mismatch-history evidence rather than active paired truth |
| 2026-04-16 23:10 | Closed `GB-52` as `positive but bounded`: once the repaired `2048` paired surface landed, the original paired-promotion blocker disappeared; paired `PIA + SecMI` feature promotion may now reopen, but the landed `SecMI-only` first canary remains valid history and the next live lane should be `CDI paired-feature scorer design`, not another immediate GPU question |
| 2026-04-16 23:25 | Closed `GB-53` as `positive but bounded`: the repo now has one honest paired `CDI` scorer design on the repaired `2048` shared surface, implemented as a control-fitted `SecMI stat + PIA` `control-z-linear` scorer; the bounded internal canary is positive (`paired_t = 30.027926`) and modestly exceeds the stronger single feature, but the line still remains internal audit-shape evidence rather than external copyright-grade proof |
| 2026-04-16 23:40 | Closed `GB-54` as `positive but constrained`: the paired scorer remains strongly positive under the reversed half-split review and its weights stay stable near `SecMI 0.52 / PIA 0.48`, so it is no longer just a one-off paired canary; however it does not consistently dominate `SecMI`, so the honest boundary is `default internal paired scorer on the repaired 2048 surface`, not headline scorer or external evidence |
| 2026-04-16 23:50 | Closed `GB-55` as `positive`: gray-box README, Research README, and comprehensive progress now all reflect the current `CDI` truth and boundary, so fresh sessions no longer need to reconstruct it indirectly from lane-local notes; this round does not change GPU posture, but it does create a clear materials-sync recommendation |
| 2026-04-17 00:00 | Closed `GB-56` as `positive`: the paired scorer now has an explicit default-run policy and the canary script default aligns to it: `--paired-scorer auto` enables `control-z-linear` when paired inputs are present and falls back to component-only mode otherwise; paired runs must continue to report `paired + SecMI + PIA` together, so the line is easier to reuse without widening its claim boundary |
| 2026-04-17 00:10 | Closed `GB-57` as `positive`: internal paired `CDI` canaries now emit a stable machine-readable contract in `audit_summary.json`, including policy request/effective mode, component-reporting requirement, and explicit `headline/external=false` boundary flags; higher-layer consumers no longer need to infer these rules from Markdown alone |
| 2026-04-17 00:20 | Closed `GB-58` as `positive`: higher-layer consumers now have an explicit paired-`CDI` read order and anti-overclaim rule; `Leader/materials` should consume contract-first and keep to boundary flags plus notes, while any future `Platform/Runtime` consumer must hard-gate on the machine-readable contract and keep scorer weights/details diagnostic-only, so `gpu_release` remains `none` and the next live CPU lane becomes `gray-box post-CDI lane reselection review` |
| 2026-04-17 00:35 | Closed `GB-59` as `positive`: once `CDI` scorer interpretation was frozen, the highest-value gray-box question was no longer another `CDI` follow-up or a weak-family reopen; the best next CPU-first lane is now `Noise as a Probe promotion-gap review`, because it is the strongest unpromoted new mechanism and its defended-extension path already closed as `no-go`, leaving promotion/comparability as the real blocker; `gpu_release` and `next_gpu_candidate` both remain `none` |
| 2026-04-17 00:50 | Closed `GB-60` as `negative but clarifying`: `Noise as a Probe` is no longer blocked on execution truth, but it still has no honest direct promotion path because the packaged gray-box challenger board remains anchored on the shared `DDPM/CIFAR10` surface while this branch lives on a separate `SD1.5 + target-family LoRA` latent-diffusion contract; defended extension is also already a `no-go`, so the next CPU-side question becomes `Noise as a Probe contract-shift review`, with `gpu_release` and `next_gpu_candidate` still fixed at `none` |
| 2026-04-17 01:05 | Closed `GB-61` as `negative but clarifying`: even after widening the question to a latent-diffusion same-surface board, the current evidence still does not justify building one, because `Noise as a Probe` is the only repeat-positive branch there while `MoFit` remains hold and `structural memorization` remains direction-negative; the board would add pseudo-comparability rather than stronger truth, so `gpu_release` and `next_gpu_candidate` remain `none`, and the next live CPU lane becomes `gray-box post-noise contract-shift reselection review` |
| 2026-04-17 01:20 | Closed `GB-62` as `negative but clarifying`: after both latent-diffusion follow-up questions closed, gray-box no longer exposes a high-value immediate next-family lane; its headline, challenger, corroboration, and internal `CDI` extension are all already stable, so the next live CPU-first slot should now move to `white-box post-breadth next-hypothesis selection review`, while `gpu_release` and `next_gpu_candidate` stay at `none` |
| 2026-04-17 01:35 | Closed `WB-19` as `negative but clarifying`: white-box also no longer exposes an honest immediate next-hypothesis execution lane, because `DP-LoRA` is already frozen as a bounded metric-split branch, `Finding NeMo` remains not-requestable, and `GSA2` is only same-family corroboration; therefore the next live CPU-first slot should now move to `cross-box closure-round system sync review`, while `gpu_release` and `next_gpu_candidate` remain `none` |
| 2026-04-17 01:50 | Closed `X-9` as `positive`: after the gray-box and white-box closure round, the summary layer and challenger queue were partially stale; they now reflect the current repo truth that `active GPU question = none`, box-local immediate execution lanes are closed, and the next live CPU-first slot should move to `black-box next-family candidate-generation refresh review` |
| 2026-04-17 02:05 | Closed `BB-8` as `negative but clarifying`: the black-box refresh review found no honest ready next-family promotion candidate, because the visible options are either same-family continuation (`semantic-aux`), boundary-only (`CLiD`), needs-assets (`variation`), or better classified outside immediate black-box promotion (`dataset-audit-track / CDI`); keep `gpu_release = none`, keep `next_gpu_candidate = none`, and move the next live CPU-first slot to `second gray-box defense mechanism selection` |
| 2026-04-17 02:20 | Closed `GB-63` as `positive but bounded`: after black-box candidate refresh yielded the near-term slot back, gray-box second-defense selection could now be frozen cleanly; `TMIA-DM late-window + temporal-striding(stride=2)` is the only materially different defense mechanism with repeat/scale support, while cheap perturbations are already negative, `Noise as a Probe` has no defended-extension gate, and `MoFit` remains hold, so keep `gpu_release = none`, keep `next_gpu_candidate = none`, and move the next live CPU-first slot to `distinct white-box defended-family import / selection` |
| 2026-04-17 02:35 | Closed `WB-20` as `negative but clarifying`: white-box still does not expose a distinct defended-family import-ready lane, because the visible options are either same-family corroboration (`GSA2`), bounded branch continuation (`DP-LoRA`), observability hold (`Finding NeMo`), or family-alias collapse (`Local Mirror`); keep `gpu_release = none`, keep `next_gpu_candidate = none`, and move the next live CPU-first slot to `ranking-sensitive variable search` |
| 2026-04-17 02:50 | Closed `GB-64` as `positive but bounded`: among current ranking-sensitive directions, `PIA vs SecMI` still only supports a negative naive-fusion verdict, while `PIA vs TMIA-DM` already shows bounded-positive same-split actionability gain; therefore the next honest gray-box CPU-first lane is `PIA vs TMIA-DM confidence-gated switching design review`, while `gpu_release` and `next_gpu_candidate` remain `none` |
| 2026-04-17 03:05 | Closed `GB-65` as `positive but bounded`: the next `PIA/TMIA-DM` step should not be another static fusion rerun but a confidence-gated switching design that uses only attack-side normalized scores, dominant-method identity, and a frozen margin-gap threshold with `z-score sum` fallback; keep `gpu_release = none`, keep `next_gpu_candidate = none`, and move the next live CPU-first slot to `PIA vs TMIA-DM confidence-gated switching offline packet` |
| 2026-04-17 03:20 | Closed `GB-66` as `negative but useful`: the first real `PIA vs TMIA-DM` confidence-gated switching offline packet executed honestly, but it did not beat bounded `z-score sum` on the aligned undefended surfaces and degraded further on the defended surface, so it remains a bounded ranking-sensitive analysis packet rather than a promoted scorer family |
| 2026-04-17 03:30 | Closed `GB-67` as `positive`: after the switching packet closed, gray-box no longer exposed a more urgent immediate CPU-first lane than the best remaining cross-box or other-box questions, so gray-box should now yield the next live CPU-first slot while keeping `gpu_release = none` and the current packaged gray-box truth unchanged |
| 2026-04-17 03:45 | Closed `X-10` as `positive`: the post-gray-box-yield reselection review found that black-box and white-box immediate reopens are still lower-value than finishing the innovation-truth gap on the current strongest mechanistic line, so the live `CPU-first` lane is now frozen to `I-A truth-hardening`, with `next_gpu_candidate = none` and `cpu_sidecar = PIA provenance / higher-layer boundary sync` |
| 2026-04-17 04:00 | Closed the current `I-A` truth-hardening packet as `positive but bounded`: `PIA + stochastic-dropout` now has a frozen formal mechanism statement, a bounded repeated-query adaptive-attacker reading, mandatory low-FPR reporting, and higher-layer mechanistic wording, but it still remains below any claim of validated privacy protection or general adaptive-attacker closure |
| 2026-04-17 04:15 | Closed `X-11` as `positive`: the higher-layer `PIA` entry docs now all carry the same mechanistic reading (`epsilon-trajectory consistency -> inference-time randomization`), bounded repeated-query adaptive boundary, four-metric read order, and long-term provenance blocker, so the summary layer no longer drifts between partial `AUC-only` or `provenance-only` readings |
| 2026-04-17 04:20 | After the higher-layer provenance/boundary sync, the next live CPU-first lane moves to `I-B minimum honest protocol bridge`; keep `next_gpu_candidate = none`, and retain higher-layer `PIA provenance / I-A` wording maintenance as the carry-forward CPU sidecar |
| 2026-04-17 04:35 | Closed `I-B.1` as `positive but bounded`: the minimum honest protocol bridge is now frozen to `activation-only migrated DDPM observability on admitted GSA assets`, not paper-faithful `Finding NeMo`; the bridge is CPU-only, read-only, and still below release, so the next live CPU-first lane now moves to `I-B.2 bounded localization observable selection` while `next_gpu_candidate` remains `none` |
| 2026-04-17 04:45 | Closed `I-B.2` as `positive but bounded`: the first honest localization observable is now frozen to the raw sample-level activation tensor under the fixed admitted `GSA` selector/timestep/sample-pair contract; scalar `summary_stat` remains metadata only, `grad_norm` remains a supporting comparator candidate, and the next live CPU-first lane now moves to `I-B.3 bounded local intervention proposal` while `next_gpu_candidate` stays `none` |
| 2026-04-17 05:05 | Closed `I-B.3` as `positive but bounded`: the first honest local intervention proposal is now frozen to a selector-local `top-k` channel attenuation mask on the admitted `GSA` activation bridge, using one fixed selector/timestep and a small bounded mask rather than hard ablation or global perturbation; it remains below release and below defense claims, and the next live CPU-first lane now moves to `I-B.4 quality-vs-defense metric contract` while `next_gpu_candidate` stays `none` |
| 2026-04-17 05:20 | Closed `I-B.4` as `positive but bounded`: the intervention line now has an explicit multi-axis review contract requiring defense metrics, mandatory control-surface drift, locality budget, and compute-cost fields together, so future review can no longer hide behind `AUC-only` gains or vague “local” wording; the next live CPU-first lane now moves to `I-C.1 falsifiable minimal experiment` while `next_gpu_candidate` stays `none` |
| 2026-04-17 05:35 | Closed `I-C.1` as `positive but bounded`: the cross-permission hypothesis now has one explicit falsifiable minimal packet on the local `DDPM/CIFAR10` overlap surface, requiring a bounded white-box targeted mask to beat a matched random mask and directionally co-reduce white-box local activation contrast and gray-box `PIA` member advantage without escaping the existing drift budget; black-box remains intentionally outside this first packet, and the next live CPU-first lane now moves to `I-C.2 define which internal units or masks would be tested` while `next_gpu_candidate` stays `none` |
| 2026-04-17 05:50 | Closed `I-C.2` as `positive but bounded`: the first cross-permission mask family is now frozen to channel-local masks on `mid_block.attentions.0.to_v` at timestep `999`, with one primary targeted mask (`top_abs_delta_k`) and two same-budget controls (`random_k_seeded`, `bottom_abs_delta_k`); this keeps the line below neuron or mechanism claims while preventing future `I-C` work from hiding behind vague “local unit” wording, and the next live CPU-first lane now moves to `I-C.3 define which black-box / gray-box / white-box metrics must move together to count as support` while `next_gpu_candidate` stays `none` |
| 2026-04-17 06:10 | Closed `I-C.3` as `positive but bounded`: the first cross-permission packet now has an explicit support-counting contract, where white-box local movement is necessary but insufficient, the first valid positive tier is only `white-gray bridge support`, and current admitted `recon` metrics are explicitly excluded from first-packet corroboration because they live on a different semantic surface; this both hardens anti-overclaim rules and creates the first honest hot-standby GPU candidate (`bounded I-C white-gray targeted-mask packet on the local DDPM/CIFAR10 overlap surface`), while the next live CPU-first lane now moves to `I-C.4 bounded white-gray bridge packet release review` |
| 2026-04-17 06:25 | Closed `I-C.4` as `blocked`: the first hot-standby white-gray bridge packet is still not honest to release, because the repo has only a read-only white-box activation-export adapter rather than a mask executor, and `PIA` currently exposes split-level summary plus runtime preview rather than the exact matched-packet score-gap export required by the packet contract; therefore `active_gpu_question` and `next_gpu_candidate` both return to `none`, and the next live CPU-first lane now moves to `I-C.5 minimum executable surface scaffolding for the white-gray bridge packet` |
| 2026-04-17 06:40 | Closed `I-C.5` as `positive but bounded`: the minimum unblock surface is now frozen to two explicit scaffold families, `export-gsa-observability-masked-packet` on the white-box side and `export-pia-packet-scores` on the gray-box side, each with a small machine-readable artifact contract; this keeps the next task focused on executable surface rather than more theory drift, and the next live CPU-first lane now moves to `I-C.6 implement the minimum CPU-first scaffold for the white-gray bridge packet` while `next_gpu_candidate` stays `none` |
| 2026-04-17 07:05 | Closed `I-C.6` as `positive but bounded`: the minimum CPU-first scaffold is now implemented and verified on real local assets, with a white-box masked-packet canary and a gray-box packet-score export canary both returning `ready`; this clears the implementation blocker from `I-C.4`, but still does not establish support truth or justify immediate GPU release, so the next live CPU-first lane now moves to `I-C.7 bounded white-gray CPU bridge canary interpretation and GPU re-release review` while `next_gpu_candidate` remains `none` |
| 2026-04-17 07:20 | Closed `I-C.7` as `blocked`: the new CPU canaries do not yet form one joint bridge packet, because the white-box side is still anchored on one named sample pair while the gray-box side currently exports a different fixed split-index packet, and the white-box manipulation is still offline tensor masking rather than in-model intervention; therefore GPU re-release remains `none`, and the next live CPU-first lane now moves to `I-C.8 same-packet identity and in-model intervention contract` |
| 2026-04-17 07:40 | Closed `I-C.8` as `positive`: the white-gray bridge now has an explicit same-packet contract keyed on `CIFAR10 canonical_index`, with `PIA` split semantics as the membership authority, a first honest packet shape of `1 member + 1 nonmember`, and an explicit in-model intervention requirement that rejects offline tensor masking as scaffold-only; because the current white-box control canary maps to `canonical_index = 467`, which is a `PIA` member rather than a `PIA` nonmember, the old pair cannot be reused as the first matched bridge packet, and `next_gpu_candidate` therefore remains `none` while the next live CPU-first lane moves to `I-C.9 canonical-index bridge binding and membership-consistent pair freeze` |
| 2026-04-17 08:00 | Closed `I-C.9` as `positive but bounded`: the first honest `same-packet` bridge pair is now frozen to member `canonical_index = 965` and nonmember `canonical_index = 1278`, with exact `PIA` offsets `1238 / 1803`, and both the white-box masked-packet scaffold and the gray-box packet-score scaffold have already emitted real CPU-first artifacts on that exact `1 + 1` pair; this resolves pair-selection drift but still does not establish support or restore GPU release, so the next live CPU-first lane now moves to `I-C.10 implement the in-model white-box intervention surface and CPU matched-pair co-movement canary` while `next_gpu_candidate` remains `none` |
| 2026-04-17 08:35 | Closed `I-C.10` as `blocked but useful`: the new `export-gsa-observability-inmodel-packet` surface now lands a real matched-pair white-box in-model canary on `965 / 1278`, with targeted local movement preserved and a tiny but nonzero downstream `epsilon`-prediction drift, so the white-box-side execution blocker is cleared; however a same-spec gray-box co-movement canary is still not honest, because `PIA` does not expose the same selector and the closest structural alias is shape-mismatched with the current white-box channel contract, so `next_gpu_candidate` remains `none` and the next live CPU-first lane now moves to `I-C.11 selector-alias and architecture-compatibility review for gray-box bridge intervention` |
| 2026-04-17 09:05 | Closed `I-C.11` as `blocked but useful`: the alias review found one honest primary gray-box selector candidate, `middleblocks.0.attn.proj_v`, plus one weaker block-level fallback, but it also confirmed a real architecture blocker: the frozen white-box channel contract assumes `channel_dim = last axis` on a `(512, 512)` value projection, while the gray-box alias is a `(256, 256, 1, 1)` `Conv2d` output with `channel_dim = 1`; therefore the bridge is not `no-go`, but direct same-spec reuse is still dishonest, `next_gpu_candidate` remains `none`, and the next live CPU-first lane now moves to `I-C.12 gray-box translated-contract alias probe on middleblocks.0.attn.proj_v` |
| 2026-04-17 09:45 | Closed `I-C.12` as `positive but bounded`: the new `export-pia-translated-alias-probe` surface now executes a real gray-box translated-contract canary on the frozen `965 / 1278` pair, with explicit `alias_selector = middleblocks.0.attn.proj_v`, `translated_from = mid_block.attentions.0.to_v`, `same_spec_reuse = false`, `tensor_layout = BCHW`, and `channel_dim = 1`; the probe records the honest gray-box alias shape `(1, 256, 4, 4)` and weight shape `(256, 256, 1, 1)`, preserves the local mask-retention signal at `0.5`, and produces a nonzero packet-local score-gap delta `-0.033422`, but this still counts only as translated-contract execution rather than same-spec white-gray support, so `next_gpu_candidate` remains `none` and the next live CPU-first lane now moves to `I-C.13 bridge verdict review after translated-contract canary` |
| 2026-04-17 10:20 | Closed `I-C.13` as `blocked`: the first bridge-promotion review confirms that the executed `I-C.10 + I-C.12` packet set still cannot be promoted beyond `translated-contract canary only`, because same-spec reuse is explicitly false, no translated-contract targeted-vs-random falsifier exists yet, no split-level gray-box support bundle exists yet on the translated packet, and the current frozen pair starts from a negative gray-box member-control gap rather than a clean support-eligible member advantage; therefore `next_gpu_candidate` remains `none`, `gpu_release` remains `none`, and the next live CPU-first lane now moves to `I-C.14 translated-contract targeted-vs-random falsifier on the frozen pair` |
| 2026-04-17 10:45 | Closed `I-C.14` as `negative but useful`: the first translated-contract targeted-vs-random falsifier on the frozen `965 / 1278` pair shows that `top_abs_delta_k` is genuinely more concentrated than both `random_k_seeded` and `bottom_abs_delta_k` on alias-local contrast, but it still does not beat the matched random control on the gray-box support-facing packet readout; targeted produces `member_control_score_gap_delta = -0.033422`, random produces `+0.031760`, and bottom stays near-flat at `-0.003209`, so the current translated surface yields a useful negative falsifier rather than first support, `next_gpu_candidate` remains `none`, and the next live CPU-first lane now moves to `X-12 non-graybox next-lane reselection after translated I-C falsifier` |
| 2026-04-17 11:10 | Closed `X-12` as `positive`: after the translated `I-C` packet produced both its first executability result and its first negative falsifier, the next honest move is a reselection away from stale box-local reopens rather than an `I-A` rebound; black-box still has no ready next-family candidate, white-box still has no immediate next-hypothesis lane, and `I-A` remains a sidecar, so the next live CPU-first lane now moves to `X-13 cross-box / system-consumable sync after translated I-C falsifier`, while `next_gpu_candidate` stays `none` |
| 2026-04-17 11:35 | Closed `X-13` as `positive`: the higher-layer sync pass now propagates the sharper `I-C` boundary into system-consumable entry points, including a corrected challenger queue state and Leader-facing wording that preserves `translated-contract-only + negative falsifier` without changing admitted metrics or consumer schema; `next_gpu_candidate` remains `none`, and the next live CPU-first lane now moves to `I-D.1 honest conditional target contract` |
| 2026-04-17 12:00 | Closed `I-D.1` as `positive but bounded`: the first honest conditional target contract is now frozen to `Stable Diffusion v1.5 base + celeba_partial_target/checkpoint-25000` on a text-conditioned latent-diffusion local canary surface, with current `recon DDIM public` evidence retained only as family-level runtime support rather than general conditional coverage; this explicitly excludes `DDPM/CIFAR10` transfer, `DiT/Kandinsky` parity, and `Finding NeMo` paper-faithful white-box claims, so `active_gpu_question` stays `none`, `next_gpu_candidate` becomes `I-D.2 SD1.5 CFG micro-probe on the frozen local canary contract`, and the next live CPU-first lane now moves to `I-D.2 bounded CFG-scale probe` |
| 2026-04-17 03:49 | Closed `I-D.2` as `positive but bounded`: the first real conditional `CFG` packet on the frozen `SD1.5 + celeba_partial_target/checkpoint-25000` contract now exists on a single disjoint `8 / 8 / 8` GPU micro-probe, with `DDIM 10 / 10`, fixed `inversion_guidance_scale = 1.0`, and `generation_guidance_scale = 3.5` vs `7.5`; `7.5` widens raw member-vs-nonmember `MSE` separation (`1081.7483 -> 1580.4574`) and lowers same-run `FPR` (`0.25 -> 0.125`) at the cost of slightly lower `TPR` (`1.0 -> 0.875`), but cross-scale threshold portability fails entirely (`g35 threshold on g75: TPR 0.125 / FPR 0.0`; `g75 threshold on g35: TPR 1.0 / FPR 1.0`), so `active_gpu_question` returns to `none`, the next live CPU-first lane now moves to `I-D.3 bounded CFG-randomization defense idea`, and `next_gpu_candidate` becomes `I-D.3 SD1.5 hidden-guidance-jitter micro-probe (pending adaptive-review contract freeze)` |
| 2026-04-17 04:05 | Closed `I-D.3` as `positive but bounded`: the first bounded conditional-randomization defense idea is now frozen to hidden-guidance jitter on the same `SD1.5 + celeba_partial_target/checkpoint-25000` contract, using CPU-side recombination of the real `g=3.5` and `g=7.5` GPU packets over one fixed `8 / 8 / 8` packet; across seeds `0..9`, mixed hidden-guidance thresholding drops mean attack accuracy to `0.675` with mean `TPR = 0.6375` and mean `FPR = 0.2875`, and no mixed seed beats the fixed packets on accuracy, but the effect remains seed-sensitive and does not yet establish low-FPR or adaptive robustness, so `active_gpu_question` stays `none`, the next live CPU-first lane now moves to `X-14 cross-box / system-consumable sync after first bounded I-D packet pair`, and `next_gpu_candidate` becomes `I-D.3 actual runner-level hidden-guidance-jitter rerun on the frozen 8/8/8 packet (pending adaptive-review contract freeze)` |
| 2026-04-17 04:15 | Closed `X-14` as `positive`: the first bounded `I-D` attack/defense packet pair is now synchronized into higher-layer truth, so Leader-facing and system-consumable entry points explicitly carry `local conditional canary only + bounded CFG probe + bounded hidden-guidance jitter idea`, while admitted tables, consumer schema, and competition-facing claims remain unchanged |
| 2026-04-17 04:25 | Closed `I-D.4` as `negative but useful`: the first actual runner-level deterministic hidden-guidance-jitter rerun on the frozen `8 / 8 / 8` packet is now real and reproducible, but under the honest `seed + split + file_name` selection rule it collapses to `accuracy = 0.5 / TPR = 0.625 / FPR = 0.625` with separation only `47.2605`, so the earlier optimistic CPU-side mixed-packet reading does not survive runner-level contract freeze; `active_gpu_question` stays `none`, `next_gpu_candidate` returns to `none`, and the next live CPU-first lane now moves to `I-A truth-hardening refresh after negative actual I-D rerun` |
| 2026-04-17 04:45 | Closed the `I-A` refresh after negative actual `I-D.4` rerun as `positive but stabilizing`: the failed runner-level conditional-defense rerun narrows a future-surface branch but does not weaken the existing mechanistic `PIA + stochastic-dropout` packet, so `I-A` remains the strongest near-term innovation track, `next_gpu_candidate` stays `none`, and the next live CPU-first lane now moves to `X-15 non-graybox next-lane reselection after I-A refresh` |
| 2026-04-17 05:00 | Closed `X-15` as `positive`: after the `I-A` refresh, black-box transfer remains `needs-assets`, white-box distinct-family reopen remains closed-negative, and the conditional future-surface just produced a real negative runner-level rerun, so the next honest non-graybox `CPU-first` lane now moves to `I-B.5 first bounded localization/intervention packet selection`; `next_gpu_candidate` stays `none`, and the carry-forward CPU sidecar remains `higher-layer PIA provenance / I-A boundary maintenance` |
| 2026-04-17 05:15 | Closed `I-B.5` as `positive`: the first executable localization/intervention packet is now frozen to a white-box-only, CPU-only, in-model `top_abs_delta_k` attenuation review on the native `I-B` pair (`target-member/00-data_batch_1-00965.png` vs `target-nonmember/00-data_batch_1-00467.png`) under `checkpoint-9600`, `mid_block.attentions.0.to_v`, `timestep = 999`, `k = 8`, and `alpha = 0.5`; this keeps the packet inside the original `I-B` contract instead of importing `I-C` bridge semantics, so `next_gpu_candidate` stays `none` and the next live CPU-first lane now moves to `I-B.6 implement first bounded localization/intervention packet` |
| 2026-04-17 05:30 | Closed `I-B.6` as `positive but bounded`: the first bounded localization/intervention packet now executes successfully on current admitted assets through `export-gsa-observability-inmodel-packet` for the native `965 / 467` pair, producing the expected local attenuation (`selected_delta_retention_ratio = 0.5`) with `off_mask_drift = 0.0` and only tiny `epsilon`-prediction drift (`2.81695e-07` mean RMS), but this still counts only as execution-positive rather than defense-positive because no attack-side evaluation bundle is yet attached; `next_gpu_candidate` stays `none`, and the next live CPU-first lane now moves to `I-B.7 bounded attack-side evaluation packet selection` |
| 2026-04-17 05:45 | Closed `I-B.7` as `positive`: the first honest attack-side review surface should stay on the admitted `GSA` asset family with one bounded evaluation-size override rather than reusing the full-scale admitted board or creating a duplicate subset asset root; this keeps the first quality-vs-defense review bounded and on-contract, so `next_gpu_candidate` stays `none` and the next live CPU-first lane now moves to `I-B.8 implement bounded attack-side evaluation packet control on admitted GSA surface` |
| 2026-04-17 06:05 | Closed `I-B.8` as `positive but bounded`: the admitted `GSA` runtime surface now exposes a real bounded attack-side evaluation control via `run-gsa-runtime-mainline --max-samples`, and the first CPU-only `max_samples = 64` review on reused `epoch300 rerun1` gradients lands at `AUC = 0.988159 / ASR = 0.90625 / TPR@1%FPR = 0.453125 / TPR@0.1%FPR = 0.0` with `target_eval_size = 128`; this is control-positive rather than defense-positive, so `next_gpu_candidate` stays `none` and the next live CPU-first lane now moves to `I-B.9 select first honest intervention-on/off bounded attack-side review contract` |
| 2026-04-17 06:40 | Closed `I-B.9` as `positive`: the first honest intervention-on/off review is now frozen to a target-anchored fixed-mask dual-run bounded board on the same admitted `GSA` `max_samples = 64` packet, with one frozen `top_abs_delta_k` mask object from the native `965 / 467` packet replayed across target and shadow extractions and no per-model mask reselection; this is still contract-only rather than defense evidence, so `next_gpu_candidate` remains `none` and the next live CPU-first lane now moves to `I-B.10 implement target-anchored fixed-mask intervention-on/off bounded attack-side review surface on admitted GSA assets` |
| 2026-04-17 07:05 | Closed `I-B.10` as `positive but bounded`: the repository now exposes `run-gsa-runtime-intervention-review`, a bounded dual-run review surface that reads one frozen target-anchored mask summary, runs baseline and intervened boards on the same packet definition, and emits `baseline.metrics + intervened.metrics + metric_deltas + locality_anchor`; unit coverage now includes the new CLI surface, but no admitted real-asset packet has been launched yet, so the honest reading is `implementation-positive / admitted-execution-pending`, `active_gpu_question` stays `none`, and `next_gpu_candidate` now becomes `I-B.11 actual target-anchored fixed-mask intervention-on/off bounded packet on admitted GSA assets (pending execution-budget review)` |
| 2026-04-17 07:25 | Closed `I-B.11` as `blocked but useful`: execution-budget review shows the first admitted packet is still not honest to release, because the new dual-run surface bounds only the closed-loop evaluation stage while extraction still iterates the full admitted board; the admitted manifest is `1000 + 1000 + 3*(1000 + 1000) = 8000` images per board, so baseline plus intervened would still traverse `16000` image-level extractions. `next_gpu_candidate` therefore returns to `none`, and the next live CPU-first lane now moves to `I-B.12 implement extraction-side bounded dataset cap for target-anchored fixed-mask intervention review` |
| 2026-04-17 07:50 | Closed `I-B.12` as `positive but bounded`: the dual-run review surface now supports extraction-side boundedness through `extraction_max_samples`, with fallback from `max_samples`, and the updated tests verify both the runtime field and that the bounded cap is actually passed into the extractor; this clears the execution-budget blocker from `I-B.11`, but no admitted real-asset packet has run yet, so the honest reading is still implementation-level only, `active_gpu_question` stays `none`, and the next live CPU-first lane now moves to `I-B.13 launch review for first truly bounded admitted target-anchored fixed-mask intervention-on/off packet` while `next_gpu_candidate` becomes that actual bounded packet |
| 2026-04-17 08:05 | Closed `I-B.13` as `positive`: the first truly bounded admitted launch config is now frozen to `max_samples = extraction_max_samples = 64` on admitted `GSA epoch300 rerun1` assets, reusing the native `965 / 467` frozen mask summary on `cuda` under the paper-aligned runtime schedule; the total extraction budget is now honestly bounded to `1024` images instead of a hidden full-board replay, so the next honest action is the actual packet execution itself, `active_gpu_question` moves to that running packet, and `next_gpu_candidate` returns to `none` while it is active |
| 2026-04-17 08:20 | Closed `I-B.14` as `negative but useful`: the first actual bounded admitted fixed-mask packet now exists on real `GSA epoch300 rerun1` assets and keeps the locality anchor clean (`selected_delta_retention_ratio = 0.5 / off_mask_drift = 0.0`), but the bounded attack-side board moves in the wrong direction under intervention (`AUC 0.992065 -> 0.995605`, `ASR 0.960938 -> 0.976562`, `TPR@1%FPR 0.734375 -> 0.765625`, `TPR@0.1%FPR 0.0 -> 0.0`); the branch therefore gains one real falsifier, `active_gpu_question` returns to `none`, `next_gpu_candidate` remains `none`, and the next live CPU-first lane becomes `I-B.15 post-first-actual-packet boundary / reselection review` |
| 2026-04-17 08:30 | Closed `I-B.15` as `negative but clarifying`: after the first real bounded admitted packet, the current `Finding NeMo / I-B` branch should no longer be read as either `zero-GPU hold` or a defense-positive line; the strongest honest wording is now `actual bounded falsifier`, so same-family GPU rescue reruns stay below release until a genuinely new bounded hypothesis appears, and the next live CPU-first lane moves to `X-16 non-graybox next-lane reselection after first actual negative I-B packet` |
| 2026-04-17 08:35 | Closed `X-16` as `positive`: after the sharper `I-B` boundary, reopening another box-local execution family is weaker than one sync pass, because black-box transfer is still asset-blocked, white-box same-family reruns are now explicitly below release, and `I-A` remains sidecar-strength rather than the best new main lane; the next live CPU-first lane therefore moves to `X-17 cross-box / system-consumable sync after first actual negative I-B packet`, while `next_gpu_candidate` stays `none` |
| 2026-04-17 08:40 | Closed `X-17` as `positive`: higher-layer entry points now reflect that `Finding NeMo` has one actual bounded admitted packet and that the current honest verdict is `negative but useful`, not `zero-GPU hold` and not defense-positive; leader-facing and competition-facing wording are narrowed without changing admitted tables or consumer schema, `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the next live CPU-first lane becomes `I-A truth-hardening refresh after first actual negative I-B packet` |
| 2026-04-17 08:50 | Closed the `I-A` refresh after first actual negative `I-B` packet as `positive but stabilizing`: the new white-box falsifier narrows a competing near-term branch but does not weaken the admitted mechanistic `PIA + stochastic-dropout` packet, so `I-A` remains the strongest near-term innovation track; one stale `Finding NeMo` sentence in `mainline-narrative.md` was corrected, `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the next live CPU-first lane moves to `X-18 non-graybox next-lane reselection after I-A refresh from negative actual I-B packet` |
| 2026-04-17 09:00 | Closed `X-18` as `positive`: after the refreshed `I-A` ordering, the next honest non-graybox `CPU-first` lane is now `XB-CH-2 transfer / portability blocker refresh review`, because gray-box hold items still lack a new signal, white-box same-family rescue remains below release, and transfer / portability remains the highest-value unresolved branch even though it is still blocker-shaped; `active_gpu_question` stays `none` and `next_gpu_candidate` stays `none` |
| 2026-04-17 09:15 | Closed `XB-CH-2` refresh as `negative but useful`: recent repo progress sharpens the blocker rather than resolving it, because the current gray-box portability evidence is still only an intra-contract `PIA GPU128/GPU256` support pair, while cross-dataset, cross-model, and cross-threat-model portability still lack paired model contracts, paired split contracts, and one bounded shared-surface hypothesis; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the next live CPU-first lane moves to `X-19 non-graybox next-lane reselection after refreshed transfer blocker review` |
| 2026-04-17 09:20 | Closed `X-19` as `positive`: once the refreshed transfer branch is confirmed to remain `needs-assets`, the honest immediate move is another non-graybox CPU-first reselection rather than pretending the blocked branch is executable; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane remains the reselection pass itself until a stronger executable branch is chosen |
| 2026-04-17 09:40 | Closed `X-20` as `positive`: the bounded higher-layer stale-entry sync is now complete, because `mainline-narrative`, `challenger-queue`, `comprehensive-progress`, and the root control board no longer drift on `Finding NeMo = actual bounded falsifier`, `XB-CH-2 = needs-assets`, and `no active GPU`; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live CPU-first lane now moves to `X-21 non-graybox next-lane reselection after X-20 stale-entry sync` |
| 2026-04-17 10:05 | Closed `X-21` as `positive`: after the stale-entry sync closed, there is still no honest executable reopen in `XB-CH-2`, `GB-CH-2`, or other blocked/hold queue items; the strongest immediate non-graybox move is therefore `X-22 I-A higher-layer truth-hardening residue audit after X-21 reselection`, while `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the CPU sidecar narrows to `PIA provenance maintenance` |
| 2026-04-17 10:20 | Closed `X-22` as `positive`: the current `I-A` residue sat in higher-layer presentation strength rather than in the underlying mechanistic packet, so leader/material entry docs were upgraded to carry the four-metric low-FPR contract and bounded repeated-query adaptive boundary explicitly; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live CPU-first lane now moves to `X-23 non-graybox next-lane reselection after X-22 I-A residue audit` |
| 2026-04-17 10:35 | Closed `X-23` as `positive`: after the `I-A` residue audit, there is still no stronger executable reopen in blocked non-graybox challenger branches, but one bounded stale layer remains in `mainline-narrative` and the root control board; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live CPU-first lane now moves to `X-24 residual stale-entry cleanup after X-23 reselection` |
| 2026-04-17 10:50 | Closed `X-24` as `positive`: the final residual stale execution-order layer in `mainline-narrative` and the root `ROADMAP` is now aligned to current truth, so `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live CPU-first lane now moves to `X-25 non-graybox next-lane reselection after X-24 cleanup` |
| 2026-04-17 11:10 | Closed `X-25` as `positive`: after the stale-entry cleanup, blocked and hold non-graybox branches still do not expose a stronger executable reopen, so the honest next move is to promote `PIA provenance maintenance` from carry-forward sidecar into the main CPU-first slot; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-26 PIA provenance maintenance main-lane review after X-25 reselection` while the CPU sidecar narrows to `I-A higher-layer boundary maintenance` |
| 2026-04-17 11:30 | Closed `X-26` as `positive`: the current `PIA` provenance blocker is already sharp enough and does not require a new manifest field, schema bump, or GPU release; the honest maintenance action is to freeze `workspace-verified` intake reading, the explicit `paper-aligned blocked by checkpoint/source provenance` boundary, and the real reopen triggers, while `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-27 non-graybox next-lane reselection after X-26 provenance review` |
| 2026-04-17 11:45 | Closed `X-27` as `positive`: after `X-26` froze the provenance blocker, the strongest unresolved non-graybox branch becomes `XB-CH-2` again, but still only as a CPU-side blocker/contract review because no paired contracts or shared execution surface were released; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-28 XB-CH-2 shared-surface contract freeze review after X-27 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 12:00 | Closed `X-28` as `positive`: the current repo still cannot freeze one honest cross-box shared execution surface, but the blocker is now explicit and compact: `paired model contract + paired split contract + shared metric hypothesis + bounded packet budget`; `XB-CH-2` therefore remains `needs-assets`, `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-29 non-graybox next-lane reselection after X-28 shared-surface contract freeze review` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 12:15 | Closed `X-29` as `positive`: after `X-28`, `XB-CH-2` is sharper but still not executable, while gray-box and white-box hold branches still do not expose a stronger immediate lane; the honest next move is therefore a bounded return to `I-A` as `X-30 I-A carry-forward truth-hardening audit after X-29 reselection`, with `active_gpu_question = none`, `next_gpu_candidate = none`, and the CPU sidecar shifting to `cross-box / system-consumable wording maintenance` |
| 2026-04-17 12:30 | Closed `X-30` as `positive but stabilizing`: the current mechanistic `I-A` wording is now stable across leader/materials/higher-layer docs, with four-metric low-FPR reporting, bounded repeated-query adaptive reading, and provenance caveat all still visible; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-31 non-graybox next-lane reselection after X-30 I-A audit` while the CPU sidecar remains `cross-box / system-consumable wording maintenance` |
| 2026-04-17 12:45 | Closed `X-31` as `positive`: the only remaining post-`X-30` drift was stale control-plane wording in the root board and one long-form progress sentence; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-32 non-graybox next-lane reselection after X-31 stale-entry sync` while the CPU sidecar remains `cross-box / system-consumable wording maintenance` |
| 2026-04-17 13:00 | Closed `X-32` as `positive`: after `X-31`, all blocked/hold non-graybox branches still remain below honest execution release and `I-A` has already returned to stable sidecar status, so the strongest main-slot choice is to promote `cross-box / system-consumable wording maintenance` back into the main lane specifically to clear active stale intake/system surfaces that still encode `Finding NeMo` and `Phase E` with old queue truth; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-33 cross-box / system-consumable stale intake sync after X-32 reselection` while the CPU sidecar narrows to `I-A higher-layer boundary maintenance` |
| 2026-04-17 13:15 | Closed `X-33` as `positive`: active higher-layer and machine-readable intake surfaces now stop encoding `Finding NeMo` as the current `Phase E` intake-only `zero-GPU hold` candidate, and instead preserve `PIA paper-aligned confirmation` as document-layer conditional only while moving `Finding NeMo` fully to `non-admitted actual bounded falsifier` boundary truth; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-34 non-graybox next-lane reselection after X-33 stale intake sync` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 13:30 | Closed `X-34` as `positive`: after both control-plane and intake-plane stale surfaces were cleared, the visible non-graybox pool still contains no honest ready main-slot lane above blocked/hold branches or stable sidecar maintenance, so the correct next move is bounded non-graybox candidate-surface expansion rather than another fake reopen; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-35 non-graybox candidate-surface expansion after X-34 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 13:45 | Closed `X-35` as `positive`: the first honest way to expand the stale non-graybox pool is to restore `I-D` as an active candidate surface, because black-box still lacks a new family contract, white-box still lacks a distinct defended-family import lane, and `I-C` still lacks a genuinely new bounded cross-box hypothesis beyond its frozen translated-contract falsifier; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-36 I-D conditional future-surface successor selection after X-35 expansion` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 14:20 | Closed `X-36` as `positive`: the restored `I-D` surface does not currently contain one genuinely new bounded successor lane, because same-contract scale tuning would be parameter churn, hidden-guidance-jitter salvage would be failed-contract rescue, and broader conditional-family widening would break the frozen `I-D.1` contract; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-37 non-graybox next-lane reselection after X-36 I-D successor freeze` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 14:35 | Closed `X-37` as `positive`: after `X-36`, no blocked/hold non-graybox branch honestly reopened and no new `I-A` successor question appeared, so the strongest immediate main-slot move is one bounded cross-box / system-consumable stale-surface sync pass on the remaining active `I-D` material wording drift; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-38 cross-box / system-consumable stale-surface sync after X-37 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 14:45 | Closed `X-38` as `positive`: the remaining active material-facing `I-D` wording surface is now aligned to `X-36`, so higher-layer readers no longer see `bounded hidden-guidance defense idea` without the sharper `negative actual runner-level rerun + no honest bounded successor lane` boundary; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-39 non-graybox next-lane reselection after X-38 stale-surface sync` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 15:05 | Closed `X-39` as `positive`: after `X-38`, the visible non-graybox pool still contains no honest ready main-slot lane above blocked/hold branches or stable sidecar maintenance, so the correct next move is another bounded candidate-surface expansion rather than a fake reopen; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-40 non-graybox candidate-surface expansion after X-39 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 15:20 | Closed `X-40` as `positive`: the first honest way to expand the stale non-graybox pool again is to restore `I-C`, but only as fresh bounded cross-box hypothesis generation rather than same-pair translated-contract hardening; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-41 I-C fresh bounded cross-box hypothesis generation after X-40 expansion` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 15:40 | Closed `X-41` as `positive`: the first genuinely new bounded `I-C` hypothesis after the translated-falsifier freeze is a bounded multi-pair agreement-first cross-box hypothesis, not another translated intervention retry; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-42 I-C bounded multi-pair agreement-first contract review after X-41 hypothesis generation` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 15:55 | Closed `X-42` as `blocked but useful`: the new agreement-first `I-C` idea survives, but the first executable board contract is still missing one second member/nonmember pair freeze under the same overlap authority; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-43 I-C secondary pairboard identity freeze after X-42 contract review` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 16:15 | Closed `X-43` as `positive but bounded`: the second pairboard identity is now frozen deterministically to member `canonical_index = 8` and nonmember `canonical_index = 23`, giving the fresh `I-C` agreement-first line a complete `2 member + 2 nonmember` identity board under the same overlap authority without metric-based cherry-picking; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-44 I-C bounded multi-pair agreement-board contract freeze after X-43 pairboard identity freeze` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 16:35 | Closed `X-44` as `blocked but useful`: the fresh `I-C` identity board is now complete and gray-box already exposes object-level per-sample scores, but the first honest agreement-board contract still cannot be frozen because current white-box admitted metrics remain pair-local deltas and no board-wide object-local concentration scalar plus selector policy has been frozen yet; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-45 I-C white-box board-local concentration scalar contract freeze after X-44 agreement-board contract review` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 17:05 | Closed `X-45` as `positive but bounded`: the white-box contract blocker is now resolved by freezing one board-local scalar, `selected_channel_abs_profile_mean`, on the board-wide selected-channel set `[5, 471, 1, 135, 360, 215, 394, 425]` inherited from the already-frozen pair-A selector, and a bounded four-object CPU probe confirms that scalar is not degenerate on the `965 / 1278 / 8 / 23` board; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-46 I-C first bounded four-object agreement-board read after X-45 scalar contract freeze` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 17:25 | Closed `X-46` as `negative but useful`: the first honest four-object agreement board is now fully readable on both surfaces, but while both white-box and gray-box still place members above nonmembers on average, they do not preserve a clean enough same-object broad order to count as positive agreement-first support; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-47 non-graybox next-lane reselection after X-46 first bounded agreement-board read` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 17:40 | Closed `X-47` as `positive`: after that first fresh `I-C` board read landed as `negative but useful`, the most honest immediate next move is one bounded cross-box / system-consumable stale-entry sync pass because higher-layer readers still see the pre-`X-46` control-plane state; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-48 cross-box / system-consumable stale-entry sync after X-47 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 17:50 | Closed `X-48` as `positive`: the active higher-layer entry docs are now aligned again to the post-`X-46` control-plane truth, so higher-layer readers no longer stop at the older `X-45` blocker state; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-49 non-graybox next-lane reselection after X-48 stale-entry sync` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 18:00 | Closed `X-49` as `positive`: once the stale-entry sync is cleared, no stronger ready non-graybox branch reopens above `I-A`, because fresh `I-C` just landed a negative-but-useful first board read, `XB-CH-2` still remains `needs-assets`, and no new defended-family or black-box challenger question appeared; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-50 I-A higher-layer boundary maintenance audit after X-49 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 18:20 | Closed `X-50` as `positive`: the remaining `I-A` residue sat in higher-layer carry-forward rather than in the packet itself, and `mainline-narrative`, `comprehensive-progress`, `admitted-results-summary`, and the current control-plane summary now again carry the bounded repeated-query adaptive reading plus mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` reporting; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-51 non-graybox next-lane reselection after X-50 I-A boundary audit` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 18:35 | Closed `X-51` as `positive`: once the `I-A` boundary audit is cleared, no stronger blocked/hold non-graybox branch reopens above one remaining materials-facing stale-entry sync, because `competition-evidence-pack` still encodes `SecMI` as `blocked baseline` and `TMIA-DM` as `intake only`; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-52 cross-box / materials stale-entry sync after X-51 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 19:10 | Closed `X-52` as `positive`: the active admitted/material-facing evidence pack now preserves `SecMI = same-asset independent corroboration line` and `TMIA-DM = strongest packaged gray-box challenger`, so higher-layer materials no longer lag behind current gray-box truth; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-53 non-graybox next-lane reselection after X-52 materials stale-entry sync` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 19:35 | Closed `X-53` as `positive`: after the materials-facing stale-entry sync, no blocked/hold non-graybox branch honestly reopened above the stable `I-A` sidecar, but `I-B` had become the strongest innovation surface without an explicit post-falsifier successor review; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-54 I-B post-falsifier successor selection after X-53 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 20:10 | Closed `X-54` as `negative but useful`: the restored `I-B` surface still has no honest bounded successor lane above its `actual bounded falsifier` freeze, because same-family rescue remains forbidden, distinct-family import is absent, and no new bounded localization-defense hypothesis is visible; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-55 non-graybox next-lane reselection after X-54 I-B successor freeze` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 20:35 | Closed `X-55` as `positive`: once `I-B` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but fresh `I-C` is now the strongest innovation surface still lacking an explicit post-negative successor review after `X-46`; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-56 I-C post-negative-agreement-board successor selection after X-55 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 21:00 | Closed `X-56` as `negative but useful`: the fresh `I-C` surface still has no honest bounded successor lane above the first negative agreement-board read, because same-board salvage remains forbidden, black-box corroboration still lacks a frozen bridge surface, and no new bounded cross-permission hypothesis is visible; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-57 non-graybox next-lane reselection after X-56 I-C successor freeze` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 21:20 | Closed `X-57` as `positive`: once `I-C` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but one active higher-layer entry doc still lags behind the current control-plane state, so the next honest move is one bounded stale-entry sync pass rather than immediate candidate-surface expansion; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-58 cross-box / system-consumable stale-entry sync after X-57 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 21:30 | Closed `X-58` as `positive`: the remaining active higher-layer entry doc is now aligned again to the post-`X-56` control-plane truth, so higher-layer readers no longer stop at the old `X-53` lane state; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-59 non-graybox next-lane reselection after X-58 stale-entry sync` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 21:45 | Closed `X-59` as `positive`: once the stale-entry sync is cleared, no blocked/hold branch honestly reopens above the stable `I-A` sidecar and no fresh successor lane appears inside `I-B` or `I-C`, so the next honest move is bounded non-graybox candidate-surface expansion rather than a forced same-family return; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-60 non-graybox candidate-surface expansion after X-59 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 22:10 | Closed `X-60` as `positive`: the restored non-graybox candidate surface is `black-box paper-backed next-family scouting`, because black-box is the only remaining non-graybox area with honest CPU-only expansion room that does not violate the current frozen `I-B / I-C` negatives, white-box distinct-family closure, or `XB-CH-2 needs-assets` boundary; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-61 black-box paper-backed next-family scoping review after X-60 expansion` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 22:35 | Closed `X-61` as `negative but useful`: the remaining paper-backed black-box backlog still does not expose one genuinely new promotable family, because the face-image LDM route is domain-specific, collection-level, and structurally overlaps with the already-landed `semantic-auxiliary-classifier` and gray-box-owned `CDI`; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-62 non-graybox next-lane reselection after X-61 black-box scoping` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 22:45 | Closed `X-62` as `positive`: once black-box scouting also closes negative and the `X-60 / X-61` changes were already synced directly into active higher-layer docs, no blocked/hold non-graybox branch honestly reopens and no stale-entry sync pass remains above the stable carry-forward line, so the strongest next live lane returns to `I-A` truth-hardening; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-63 I-A formal/adaptive low-FPR residue audit after X-62 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 23:05 | Closed `X-63` as `positive`: the remaining `I-A` residue was materials-facing rather than experiment-facing, because the active PIA visual prompt still suggested `AUC-only` comparison text; it now explicitly preserves `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` plus the bounded repeated-query adaptive boundary, so the current live lane moves to `X-64 non-graybox next-lane reselection after X-63 I-A residue audit` while `active_gpu_question` and `next_gpu_candidate` both stay `none` and the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 23:35 | Closed `X-64` as `positive`: once `X-63` cleared the last visible `I-A` residue, no blocked/hold branch honestly reopened and no stale-entry sync pass remained above sidecar maintenance, so the honest next move became another bounded non-graybox candidate-surface expansion rather than a forced `I-A` rerun or white-box reopen; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-65 non-graybox candidate-surface expansion after X-64 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-17 23:55 | Closed `X-65` as `positive`: the restored non-graybox candidate surface is `I-B paper-backed localization-defense successor scouting`, because black-box already closed negative, white-box still should not take the next slot, and the broader `Finding NeMo + local memorization + FB-Mem` mechanism stack still contains CPU-only hypothesis-generation room above same-family rescue churn; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-66 I-B paper-backed localization-defense scoping review after X-65 expansion` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-18 00:20 | Closed `X-66` as `negative but useful`: the broadened `Finding NeMo + local memorization + FB-Mem` stack still does not expose one genuinely new bounded successor hypothesis on top of the current `actual bounded falsifier`, because the additional material remains either historical intake scaffolding, observability plumbing, or paper-faithful `SD1.4 / LAION` context rather than a current admitted-surface hypothesis; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-67 non-graybox next-lane reselection after X-66 I-B paper-backed scoping` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-18 00:30 | Closed `X-67` as `positive`: once the broadened `I-B` stack also freezes, no stronger blocked/hold branch honestly reopens above the stable sidecar line, so the strongest next live lane returns to `I-A` truth-hardening rather than another immediate expansion; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live lane now moves to `X-68 I-A formal-adaptive-lowfpr carry-forward audit after X-67 reselection` while the CPU sidecar remains `I-A higher-layer boundary maintenance` |
| 2026-04-18 01:05 | Closed `X-68` as `positive but stabilizing`: the current `I-A` formal/adaptive/low-FPR contract still had one live carry-forward task, but it was only a top-summary read-path residue in `leader-research-ready-summary.md`, whose one-page table still foregrounded `AUC / ASR`; once that table was upgraded to carry the low-FPR pair explicitly, `I-A` returned to sidecar-only status, `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-69 non-graybox next-lane reselection after X-68 I-A carry-forward audit` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 01:15 | Closed `X-69` as `positive`: once the final top-summary `I-A` residue was cleared, no blocked/hold branch honestly reopened above sidecar maintenance, so the next honest move became `X-70 non-graybox candidate-surface expansion after X-69 reselection` rather than another `I-A` turn; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 01:45 | Closed `X-70` as `positive`: the next honest restored non-graybox candidate surface is `WB-CH-4 white-box loss-feature challenger family`, because black-box remains exhausted or asset-blocked, visible white-box candidates remain exhausted, and the paper-backed white-box loss-feature family (`LSA* / LiRA / Strong LiRA`) is distinct from the current `GSA` gradient family while also being directly relevant to `SMP-LoRA / DP-LoRA`; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-71 white-box loss-feature challenger scoping review after X-70 expansion` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 02:00 | Closed `X-71` as `positive but bounded`: the restored `WB-CH-4` surface does contain one honest near-term lane, but only as a bounded same-asset `LSA*`-style loss-feature contract review on current admitted `DDPM/CIFAR10` white-box assets; `LiRA / Strong LiRA` remain above current bounded host-fit budget, so `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-72 white-box same-asset loss-feature contract review after X-71 scoping` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 02:35 | Closed `X-72` as `positive but bounded`: current admitted `DDPM/CIFAR10` white-box assets do support one bounded same-asset `LSA*`-style loss-feature contract because the same runtime path already computes denoising loss on the frozen admitted datasets/checkpoints, but current runtime/mainline still exports gradients only and does not emit per-sample loss-score artifacts, so execution remains blocked on a bounded loss-score export surface review; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-73 white-box same-asset loss-score export surface review after X-72 contract review` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 03:10 | Closed `X-73` as `positive but bounded`: one honest bounded loss-score export surface does exist on current admitted `DDPM/CIFAR10` white-box assets, but the preferred path is a separate in-repo internal helper / CLI surface built on top of the existing in-process GSA extraction logic rather than patching the upstream external extractor or mutating current admitted `run-gsa-runtime-mainline` semantics first; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-74 white-box bounded internal loss-score export implementation after X-73 surface review` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 03:45 | Closed `X-74` as `positive but bounded`: the repository now exposes one separate bounded internal loss-score export surface on the admitted `DDPM/CIFAR10` white-box asset family via `export-gsa-loss-score-packet`, and one real-asset `cpu / ddpm_num_steps = 20 / sampling_frequency = 2 / extraction_max_samples = 1` smoke succeeded on target plus three shadow pairs without mutating admitted gradient-mainline semantics; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-75 white-box bounded loss-score first packet selection after X-74 implementation` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 04:10 | Closed `X-75` as `positive but bounded`: the first honest bounded loss-score packet is now frozen to a `threshold-style`, `shadow-oriented`, `shadow-threshold-transfer` board on exported scalar loss scores with `extraction_max_samples = 64` per target/shadow split, while `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` remain mandatory but the packet still stays below release-grade low-FPR honesty at that bounded scale; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-76 white-box bounded loss-score threshold evaluator implementation after X-75 packet selection` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 04:40 | Closed `X-76` as `positive but bounded`: the repository now exposes `evaluate-gsa-loss-score-packet`, which reads the exported white-box loss-score packet, freezes score orientation plus operating threshold from pooled shadow scores only, transfers that frozen board onto target evaluation, and emits the mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` bundle; one real bounded smoke on the existing export packet returned `shadow member-lower` but `target transfer = AUC 0.0 / ASR 0.0 / TPR@1%FPR 0.0 / TPR@0.1%FPR 0.0` while `target self-board` stayed positive, which confirms that the self-board must remain diagnostic-only rather than verdict truth; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-77 white-box bounded loss-score first actual packet after X-76 evaluator implementation` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 05:30 | Closed `X-77` as `positive but bounded`: the first real bounded `64`-per-split loss-score packet now exists on current admitted white-box assets via `export-gsa-loss-score-packet` plus `evaluate-gsa-loss-score-packet`; under the frozen shadow-only transfer contract, the target board lands at `AUC = 0.699463 / ASR = 0.632812 / TPR@1%FPR = 0.03125 / TPR@0.1%FPR = 0.03125`, while the diagnostic target self-board differs only by threshold and does not rescue the low-FPR boundary; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-78 white-box bounded loss-score post-first-actual-packet boundary review after X-77 actual packet` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 05:50 | Closed `X-78` as `positive but stabilizing`: the first actual white-box loss-score packet is now boundary-reviewed and should be frozen as bounded auxiliary evidence rather than promoted or immediately extended, because it is execution-positive but still only `AUC = 0.699463 / ASR = 0.632812 / TPR@1%FPR = 0.03125 / TPR@0.1%FPR = 0.03125` on the transferred target board, with weak low-FPR behavior and no genuinely new same-family follow-up hypothesis; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-79 non-graybox next-lane reselection after X-78 white-box loss-score boundary review` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 06:05 | Closed `X-79` as `positive`: once the white-box loss-score branch is frozen as bounded auxiliary evidence and current higher-layer sync is already aligned, no stronger blocked/hold non-graybox branch honestly reopens above the carry-forward sidecar, so the strongest next live lane returns to `X-80 I-A formal-adaptive-lowFPR carry-forward audit after X-79 reselection`; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 06:20 | Closed `X-80` as `positive`: one real active `I-A` carry-forward residue still existed in `docs/mainline-narrative.md`, whose current-state paragraph was still frozen at `X-76`; once that doc was updated to carry `X-77 / X-78 / X-79` and the current `X-80` lane, the higher-layer read path again aligned with repo truth; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-81 non-graybox next-lane reselection after X-80 I-A carry-forward audit` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 06:35 | Closed `X-81` as `positive`: once the active `I-A` residue was cleared, the strongest next move was still not a fresh black-box or white-box reopen, because black-box scouting remains closed negative, white-box loss-score is already boundary-frozen, and `XB-CH-2` remains `needs-assets`; the honest immediate move is one bounded `X-82 cross-box / system-consumable stale-entry sync after X-81 reselection`, because active higher-layer docs still expose stale lane state; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 06:50 | Closed `X-82` as `positive`: the active higher-layer stale-entry surfaces are now aligned again to current control-plane truth, including the current live lane and post-`X-78 / X-79 / X-80 / X-81` read-path; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-83 non-graybox next-lane reselection after X-82 stale-entry sync` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 07:00 | Closed `X-83` as `positive`: once the stale-entry sync cleared and no new active `I-A` residue remained, the strongest next move was no longer another wording-only sync pass and still not any blocked/hold reopen, because black-box scouting remains closed negative, white-box loss-score remains frozen as bounded auxiliary evidence, and `XB-CH-2` remains `needs-assets`; the honest next move is therefore `X-84 non-graybox candidate-surface expansion after X-83 reselection`; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 07:15 | Closed `X-84` as `positive`: the restored non-graybox candidate surface is `cross-box admitted-summary quality/cost read-path hardening`, because all immediate box-local reopens remain frozen or blocked while the unified comparison table already carries richer `quality_cost / evidence_level / boundary` fields that higher-layer admitted summaries do not foreground enough; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-85 cross-box admitted-summary quality-cost sync review after X-84 expansion` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-18 07:30 | Closed `X-85` as `positive`: the admitted results summary now explicitly exposes `Evidence Level` and `Quality / Cost` alongside the admitted metric columns, so higher-layer readers no longer have to infer execution scale and evidence grade only from headline metrics plus free-text boundary notes; `active_gpu_question` stayed `none`, `next_gpu_candidate` stayed `none`, and the current live lane moved to `X-86 non-graybox next-lane reselection after X-85 admitted-summary sync review` while the CPU sidecar remained `I-A higher-layer boundary maintenance` |
| 2026-04-17 16:20 | Closed `X-87` as `bounded`: root `§4 B-M0` converged to candidate `A` rather than `B`, because the current `I-C` bridge still lacks an honest same-pair gray-box four-metric release surface on the frozen `965 / 1278` pair; on one fresh `cuda` re-export of the frozen `X-75 / X-77` `DDPM/CIFAR10` `64`-per-split loss-score packet, threshold-transfer lands at `AUC = 0.671143 / ASR = 0.59375 / TPR@1%FPR = 0.015625 / TPR@0.1%FPR = 0.015625`, while Gaussian LR-transfer lands at `AUC = 0.602051 / ASR = 0.59375 / TPR@1%FPR = 0.03125 / TPR@0.1%FPR = 0.03125`, so LR beats threshold only on the two low-FPR targets and the branch closes as `bounded` rather than promoted; `active_gpu_question` returns to `none`, `next_gpu_candidate` returns to `none`, and no Platform consumer-field handoff is required because the packet schema is unchanged |
| 2026-04-17 16:56 | Closed `X-88` as `bounded`: long-horizon post-`B-M0` scoping now selects `G1-A = gray-box tri-evidence audit scorer` as the next honest distinct-family candidate, because `I-C` still lacks a same-pair four-metric release surface on the frozen `965 / 1278` pair, broadened `I-B` still exposes no genuinely new admitted-surface successor above the current actual bounded falsifier, and fresh black-box/latent second-signal paths remain contract-blocked below honest comparability; the chosen lane reuses the aligned `PIA gpu256` plus `TMIA-DM long_window gpu256` packet surfaces and the frozen internal `CDI` paired-contract, freezes one CPU-first gate of `TMIA sample-identity export + tri-score offline canary`, commits to `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`, keeps `active_gpu_question = none`, keeps `next_gpu_candidate = none` until that CPU-first gate lands and is inspected, and requires no Platform consumer-field handoff because current contract-first consumer fields stay unchanged |
| 2026-04-17 17:55 | Closed `X-89` as `positive`: the `X-88` CPU gate is now landed honestly on the frozen `gpu256_undefended` plus `gpu256_defended` surfaces, because `TMIA-DM` sample identity can be frozen from current artifacts by reusing the aligned `PIA` packet order only after matching `dataset_root / member_split_root / model_dir / max_samples / num_samples / sample_count_per_split`, and the new internal tri-score canary is a real scorer extension rather than another switching-threshold restatement; the canonical canary summary at `Research/workspaces/gray-box/runs/x88-cdi-tmiadm-triscore-canary-20260417-175249/audit_summary.json` reports macro `component_auc = { pia = 0.835152, tmiadm = 0.777561, zscore_sum = 0.852928 }` and composite `AUC = 0.854515 / ASR = 0.790039 / TPR@1%FPR = 0.130859 / TPR@0.1%FPR = 0.048828`, while on `gpu256_defended` the composite beats both bounded baselines on both low-FPR targets with higher `AUC` (`0.837601 > 0.836227 > 0.832870`); `active_gpu_question` stays `none`, `next_gpu_candidate` becomes `G1-A larger shared-surface tri-score rerun review pending separate bounded GPU review`, and no Platform consumer-field handoff is required because the consumer schema remains unchanged |
| 2026-04-17 09:25 | Closed the actual `X-19` lane selection as `positive`: the strongest immediate executable branch after the refreshed transfer blocker is one bounded `X-20` higher-layer stale-entry sync pass, because `mainline-narrative` and `challenger-queue` still underexpose the current `Finding NeMo = actual bounded falsifier` and `XB-CH-2 = needs-assets` truth; `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and the current live CPU-first lane now becomes `X-20 cross-box / system-consumable stale-entry sync after X-19 reselection` |
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
| 2026-04-18 09:40 | Closed `R2-4` as `positive`: `04-defense` is no longer only a nominal next slot, because `H1 risk-targeted SISS / retain-forget mixture` is now the selected single-family pilot, a reusable `prepare-risk-targeted-unlearning-pilot` surface exists in-repo, and one real full-overlap prep run now exports `k = 16 / 32 / 64` forget/control ladders on the current `461 / 474` shared board; the top-10% `GSA ∩ PIA` member overlap is only `8`, so current selection truth is `aggregate-percentile` rather than pure intersection-only ranking |
| 2026-04-18 10:05 | Landed the first actual `04-H1` bounded retain+forget pilot on current admitted `DDPM/CIFAR10` assets: using the exported `k32` forget list, target `checkpoint-9600`, `32` CUDA steps, `batch_size = 4`, and `L_keep` vs `L_keep - 0.5 * L_forget`, the repo now produces one real defended checkpoint plus machine-readable training log; after fixing duplicate sample-id collapse in the target-member directory scan, the canonical run trains on `33` forget files for `32` unique forget IDs and `967` retain files for `933` unique retain IDs; this remains execution-positive rather than defense-positive because no forgotten-subset board or defense-aware attack rerun is attached yet |
| 2026-04-18 10:25 | Attached the first attack-side read to `04-H1`: a `forgotten members + matched nonmembers` `GSA` subset review now exists by borrowing undefended shadow exports from the current full-overlap loss-score packet and rerunning only the target subset on both baseline and defended checkpoints; the first `defense-unaware threshold-transfer` diagnostic is negative (`AUC 0.774691 -> 0.755401`, `TPR@1%FPR 0.222222 -> 0.027778`, `TPR@0.1%FPR 0.222222 -> 0.027778`), so the lane stays alive only provisionally and still needs retained/full-split boards plus a later defense-aware rerun before any real verdict |
| 2026-04-18 13:20 | Closed the first target-wide read on `04-H1`: after fixing a no-allowlist `GSA` dataset scan bug that incorrectly tried to decode `dataset.json` as an image, the repo now produces one real full-split borrowed-shadow review on `1000` target members and `1000` target nonmembers; the result stays unfavorable (`AUC 0.618043 -> 0.596696`, `ASR 0.5515 -> 0.5665`, `TPR@1%FPR 0.018 -> 0.011`, `TPR@0.1%FPR 0.006 -> 0.003`), so the current honest stack is `forgotten negative + retained mixed/weak + full-split negative`, still below any defense-positive or defense-aware wording |
| 2026-04-18 13:33 | Tightened the `04-H1` gate with a fairer target-side control: `GSA` review export now supports deterministic per-sample paired noise via `noise_seed`, and three paired-noise reruns are now landed; they reduce the severity of the old negative read but do not reverse the direction (`forgotten AUC 0.845679 -> 0.827932`, `retained AUC 0.601307 -> 0.597222`, `full split AUC 0.623331 -> 0.617696`), while paired-noise full-split score shifts remain broadly global rather than sharply forgotten-concentrated, so the current `k32` pilot is not worth a defense-aware rerun |
| 2026-04-18 13:45 | Landed the first changed pilot within `04-H1`: a `k16` retain+forget run with unchanged optimizer hyperparameters plus a full paired-noise tri-board now exists; compared with `k32`, it keeps forgotten-subset `AUC` slightly negative but recovers low-FPR tail (`0.315789 -> 0.368421`), makes the retained companion board flat-to-tail-positive, and keeps full-split nearly neutral (`AUC 0.623331 -> 0.622141`), so `k16` now replaces `k32` as the current working instantiation even though the lane is still not defense-positive |
| 2026-04-18 14:02 | Landed the first pure-intersection lower-bound pilot inside `04-H1`: `k8` now uses the exact `Top10%(GSA) ∩ Top10%(PIA)` member overlap as its forget set and completes a full paired-noise tri-board; it keeps forgotten-subset metrics almost exactly flat, retained-companion tails no longer improve, and full-split stays near-neutral with the cleanest drift profile, so the honest read is `cleaner but too weak`, meaning `k16` remains the current working instantiation |
| 2026-04-18 15:21 | Closed `X-108` as `bounded`: the next honest `04-H1` move is no longer another `k` reselection, because `k32` is already too broad, `k8` is already too tight, and `k16` is the only rung that still preserves some low-FPR tail lift while keeping full-split near neutral; pilot-side training statistics across `k8 / k16 / k32` also show branch-frequency noise is not the decisive variable, so the next bounded GPU candidate is now frozen to one single-variable `k16 alpha-up` pilot with `alpha = 0.75`, `mixture_lambda = 0.5`, and `32` steps, while `active_gpu_question` stays `none` until a slot is deliberately released |
| 2026-04-18 16:05 | Closed `X-109` as `negative but useful`: the `k16 alpha-up` follow-up is now fully executed with a paired-noise tri-board, and it does not improve the current working instantiation; forgotten-subset tails stay exactly at the old `k16` level while `AUC` gets slightly worse (`0.885965 -> 0.883041`), the retained companion board loses the old tail gain entirely (`TPR@1%FPR 0.294118 -> 0.235294`) and now also drops `AUC` (`0.781046 -> 0.774510`), and the full-split board is not materially better (`AUC 0.622141 -> 0.622931`, but `TPR@1%FPR 0.026 -> 0.024` and `ASR 0.5675 -> 0.5690`); therefore original `k16` remains the best working instantiation, `active_gpu_question` stays `none`, and `next_gpu_candidate` returns to `none` pending a fresh CPU-side parameter-selection review |
| 2026-04-18 16:18 | Closed `X-110` as `bounded`: post-`X-109`, the current same-family search space is no longer “increase pressure again”, because stronger forget pressure already failed without improving forgotten tails; the honest remaining same-family space, if any, is now selective-variable review only (`mixture_lambda`-style frequency control or shorter budget), so `active_gpu_question` stays `none`, `next_gpu_candidate` stays `none`, and `04` does not get another immediate GPU rerun without a new CPU-side selection argument |
| 2026-04-18 16:30 | Closed `X-111` as `positive hardening`: `Research` CLI now has one real module-level entrypoint, because `python -m diffaudit.cli unsupported-command` is covered by a new subprocess regression test and `src/diffaudit/cli.py` now explicitly exits through `main()` under `__main__`; this does not change any admitted result or schema, but it removes a silent failure mode from the `Research -> Runtime` execution surface and keeps future module-level invocation safe while `active_gpu_question` and `next_gpu_candidate` both stay `none` |
| 2026-04-18 16:43 | Closed `X-112` as `bounded`: the post-`X-110` same-family selective-variable space is now narrowed to one first conditional candidate rather than an open class; under Python `random.Random(0)` and a `32`-step branch schedule, `mixture_lambda = 0.4375` is the first rung that lowers forget-branch frequency materially relative to `0.5` without collapsing straight toward near-no-op, whereas `0.375` and below already cut too sharply; this does not release a GPU question by itself, but it freezes the next conditional same-family candidate to `k16 + mixture_lambda-down (0.4375)` while keeping the rest of the contract unchanged |
| 2026-04-18 17:05 | Closed `X-113` as `negative but useful`: the `k16 mixture_lambda-down` follow-up is now fully executed with a paired-noise tri-board, and it does not improve the current working instantiation; forgotten-subset tails regress (`TPR@1%FPR 0.368421 -> 0.263158`) without any `AUC` gain, retained companion tails regress even though `AUC` rises slightly (`0.294118 -> 0.176471`), and full-split only improves headline smoothness while losing low-FPR tail (`AUC 0.622141 -> 0.624224`, but `TPR@1%FPR 0.026 -> 0.021`), so original `k16` remains the best working instantiation and `next_gpu_candidate` returns to `none` again |
| 2026-04-18 17:35 | Closed `X-114` as `bounded`: after `X-113`, `04` does not honestly jump straight to `H2 privacy-aware adapter`, because the repo still has no dedicated `04-H2` implementation / CLI / test surface and no bounded packet for that fallback; therefore `H2` remains wording-level fallback only, `04` stays in CPU-first family-review mode, `active_gpu_question` and `next_gpu_candidate` both remain `none`, and `Research -> Runtime -> Platform` stays contract-stable without any schema change |
| 2026-04-18 18:05 | Closed `X-115` as `positive`: the post-`X-114` handoff review confirms that current `04` truth changes only the research-side control plane, not any admitted metric row, Runtime endpoint, Platform snapshot shape, or runner capability; therefore the correct integration action is to update the autonomous prompt and preserve the existing admitted read path (`Research unified table -> Runtime evidence endpoint -> Platform public snapshot`) without adding consumer fields or cross-repo code changes |
| 2026-04-18 18:25 | Closed `X-116` as `positive`: even after `X-115`, two live entry docs still drifted from current repo truth, because `docs/codex-roadmap-execution-prompt.md` still encoded the old `G1-A` next-GPU steering and `workspaces/implementation/challenger-queue.md` still presented a pre-`04` live lane; both are now resynced to `active GPU question = none`, `next_gpu_candidate = none`, `04-defense = CPU-first family review`, and `no new Runtime/Platform contract`, so fresh sessions will no longer be steered by stale control-plane wording |
| 2026-04-18 18:45 | Closed `X-117` as `positive`: current Platform public `catalog.json` still overstated admitted research truth in consumer-facing `risk_interpretation` copy, especially by turning recon evidence into prescriptive defense copy, omitting gray-box provisional/privacy-boundary language, and overstating white-box mitigation as risk elimination; the snapshot wording is now hardened to match admitted boundaries while keeping all metrics, Runtime endpoints, snapshot shape, and runner contracts unchanged |
| 2026-04-21 14:14 | Closed issue #10 as `positive hardening`: added `check-recon-stage0-paper-gate` so strict `recon Attack-I` now has an executable Stage 0 gate that returns `blocked` when the current public bundle is only `proxy-shadow-member / local-semantic-chain-ready`; canonical anchor is `workspaces/implementation/2026-04-21-issue10-recon-stage0-paper-gate.md`; `active_gpu_question` and `next_gpu_candidate` both stay `none`, and no Runtime or Platform schema change is required |
| 2026-04-21 14:33 | Closed `X-118` as `positive but bounded`: `04-H2 privacy-aware adapter` is no longer a wording-only fallback, because the repo already contains `lora_ddpm.py`, `smp_lora.py`, `train_smp_lora.py`, related smoke/runtime-tuning tests, and one real bounded CPU smoke artifact at `workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/summary.json`; however, it still lacks a canonical `diffaudit` asset probe / prep / run / review chain, so `H2` is now `prototype-implemented / contract-incomplete`, `active_gpu_question` and `next_gpu_candidate` both stay `none`, and the next honest CPU-first lane becomes `X-119 04-H2 canonical contract hardening` rather than immediate GPU release |
| 2026-04-21 14:41 | Closed `X-119` as `positive but bounded`: the minimum honest `04-H2` contract is now frozen to a four-stage canonical chain: `probe-h2-assets` for admitted-asset identity and image-root validation, `prepare-h2-contract` for frozen packet/manifest generation, `run-h2-defense-pilot` for bounded adapter training with workspace-root `summary.json`, and `review-h2-defense-pilot` for same-packet attack-side comparison against the undefended baseline with mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`; canonical anchor is `workspaces/implementation/2026-04-21-x119-04-h2-canonical-contract-hardening.md`; until that chain exists, `H2` remains `prototype-implemented / contract-incomplete`, `active_gpu_question = none`, `next_gpu_candidate = none`, and no Runtime or Platform schema change is required |
| 2026-04-21 14:52 | Closed `X-120` as `positive`: the two live bootstrap prompts still encoded pre-`X-118/X-119` stale steering (`H2` as wording-only, no executable surface), so both prompt surfaces are now synchronized to `prototype-implemented / contract-incomplete` plus the sharper “missing canonical `asset probe / prep / run / review` chain” boundary; canonical anchor is `workspaces/implementation/2026-04-21-x120-active-prompt-sync-after-x119.md`; this changes only research-side control truth, keeps `active_gpu_question = none`, `next_gpu_candidate = none`, and advances the live CPU sidecar back to `I-A higher-layer boundary maintenance` |
| 2026-04-21 14:45 | Closed `X-121` as `positive but stabilizing`: after the post-`H2` prompt sync, the active entry set still preserves the current `I-A` boundary correctly, including four-metric reporting, low-FPR emphasis, and bounded repeated-query adaptive reading; canonical anchor is `workspaces/implementation/2026-04-21-x121-ia-active-entry-residue-review.md`; no new stale `AUC-only` drift is visible, `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest CPU-first lane advances to `R2-5 02-H1 SimA sidecar second-signal review` |
| 2026-04-21 14:49 | Closed `X-122` as `negative but clarifying`: current `SimA` repo truth is no longer paper-only, because `sima_adapter.py`, `test_sima_adapter.py`, and two bounded CPU packets already exist, and local `unittest` now passes; however, the best bounded rescan still stops at `AUC = 0.584961 / TPR@1%FPR = 0.03125`, so plain `SimA` scorer reopen remains below challenger quality and below honest reopen threshold; canonical anchor is `workspaces/implementation/2026-04-21-x122-r25-sima-honest-reopen-review.md`; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest CPU-first support lane becomes `R2-5b PIA + SimA support-fusion contract review` |
| 2026-04-21 14:58 | Closed `X-123` as `blocked but useful`: `PIA + SimA` support-fusion remains a valid sidecar direction, but current repo truth does not yet permit an honest shared-packet review because `crossbox_pairboard.py` and `PIA` packet export are already pairboard-ready while `SimA` still emits only aggregate feasibility summaries with no `member_scores / nonmember_scores / member_indices / nonmember_indices`; canonical anchor is `workspaces/implementation/2026-04-21-x123-r25b-pia-sima-support-fusion-contract-review.md`; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest CPU-first support lane becomes `R2-5c SimA packet-score export contract review` |
| 2026-04-21 15:08 | Closed `X-124` as `positive but bounded`: `SimA` now exposes a real pairboard-ready packet surface, because `src/diffaudit/attacks/sima_adapter.py` gained `export_sima_packet_scores(...)`, `src/diffaudit/cli.py` gained `export-sima-packet-scores`, and `tests/test_sima_adapter.py` now covers both module and CLI paths while local `unittest` passes; this closes the `member_scores / nonmember_scores / member_indices / nonmember_indices` export gap without promoting plain `SimA` itself, so `SimA` remains `execution-feasible but weak`, `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest CPU-first support lane becomes `R2-5b PIA + SimA support-fusion / calibration review`; canonical anchor is `workspaces/implementation/2026-04-21-x124-r25c-sima-packet-score-export-contract-landing.md` |
| 2026-04-21 16:02 | Closed `X-125` as `positive but bounded`: `PIA + SimA` support-fusion is now honestly executable on the frozen `461 / 474` full-overlap packet, because the new `SimA` exact-index export was run at `timestep = 160` and one real `5x` repeated-holdout pairboard now exists at `workspaces/cross-box/runs/crossbox-pairboard-pia-sima-full-overlap-20260421-r1/summary.json`; the best bounded fused candidate is `logistic_2feature`, which beats `PIA` best-single on `AUC` and `ASR` in `5 / 5` repeated runs and wins `TPR@1%FPR` in `4 / 5`, but it does not deliver a stable `TPR@0.1%FPR` lift (`2 wins / 1 tie / 2 losses`, mean `0.041558 < 0.043290`), so the line stays auxiliary rather than promoted; `active_gpu_question = none`, `next_gpu_candidate = none`, gray-box should now yield the next `CPU-first` slot, and the next honest live lane becomes `X-126 non-graybox next-lane reselection after X-125 bounded PIA + SimA review`; canonical anchor is `workspaces/implementation/2026-04-21-x125-r25b-pia-sima-support-fusion-bounded-review.md` |
| 2026-04-21 16:18 | Closed `X-126` as `positive`: after `X-125`, the control-plane entry docs were already aligned to the new gray-box yield, but two higher-layer system-consumable surfaces still encoded pre-`X-125` steering by telling readers to “first do `SimA`, then `PIA + SimA`”; therefore the highest-value immediate move was not another empty reselection loop or a fresh `I-A` wording pass, but one bounded stale-entry sync on `docs/mainline-narrative.md` and `docs/future-phase-e-intake.md`; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest live lane becomes `X-127 cross-box / system-consumable stale-entry sync after X-126 reselection`; canonical anchor is `workspaces/implementation/2026-04-21-x126-non-graybox-next-lane-reselection-after-x125.md` |
| 2026-04-21 16:24 | Closed `X-127` as `positive`: the remaining active higher-layer stale-entry layer is now cleared, because `docs/mainline-narrative.md` and `docs/future-phase-e-intake.md` no longer steer readers back into pre-`X-125` gray-box wording and now both read `02` as an auxiliary sidecar with one landed bounded `PIA + SimA` review and no stable `TPR@0.1%FPR` lift; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest live lane becomes `X-128 non-graybox next-lane reselection after X-127 stale-entry sync`; canonical anchor is `workspaces/implementation/2026-04-21-x127-crossbox-system-sync-after-x126.md` |
| 2026-04-21 16:36 | Closed `X-128` as `positive`: after `X-127`, no active stale-entry layer remains above the control plane and no blocked/hold non-graybox branch has become more executable than the admitted-mainline maintenance path, because black-box candidate scouting remains closed-negative or `needs-assets`, white-box distinct-family work remains medium-horizon preparation, `I-B` remains a `non-admitted actual bounded falsifier`, `I-C` remains `translated-contract-only + negative falsifier`, and `I-D` still has no honest bounded successor lane; therefore the highest-value next live lane is now `X-129 I-A truth-hardening after X-128 reselection`, with `active_gpu_question = none` and `next_gpu_candidate = none`; canonical anchor is `workspaces/implementation/2026-04-21-x128-non-graybox-next-lane-reselection-after-x127.md` |
| 2026-04-21 16:47 | Closed `X-129` as `positive`: after `X-128`, the remaining `I-A` residue was no longer in core research/control surfaces but in competition-facing materials, where `competition-evidence-pack` and `competition-innovation-summary` still used the softer `workspace-verified + adaptive-reviewed` wording; both are now hardened back to `workspace-verified + bounded repeated-query adaptive-reviewed`, while preserving four-metric ordering and provenance-blocked language, so the current `I-A` material-facing claim is again aligned to the admitted gray-box contract; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest live lane becomes `X-130 non-graybox next-lane reselection after X-129 I-A hardening`; canonical anchor is `workspaces/implementation/2026-04-21-x129-ia-truth-hardening-after-x128.md` |
| 2026-04-21 17:02 | Closed `X-130` as `positive`: after `X-129`, another immediate candidate-surface expansion would still have been dishonest, because one active system-consumable stale-entry layer remained on the live read path: `README.md` still pointed sessions back to the old `GB-64 / PIA vs TMIA-DM confidence-gated switching` state, and `challenger-queue.md` still exposed `WB-CH-4` as `actual-packet-pending` together with a stale `Recommended Next Order` block headed by `X-86`; therefore the highest-value immediate move became `X-131 cross-box / system-consumable stale-entry sync after X-130 reselection`, while `active_gpu_question = none` and `next_gpu_candidate = none`; canonical anchor is `workspaces/implementation/2026-04-21-x130-non-graybox-next-lane-reselection-after-x129.md` |
| 2026-04-21 17:09 | Closed `X-131` as `positive`: the remaining active stale-entry layer is now cleared, because `README.md` no longer points fresh sessions back to `GB-64` and `challenger-queue.md` no longer preserves the stale `WB-CH-4 actual-packet-pending` read or the obsolete `X-86` recommended-order block; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next honest live lane becomes `X-132 non-graybox next-lane reselection after X-131 stale-entry sync`; canonical anchor is `workspaces/implementation/2026-04-21-x131-crossbox-system-sync-after-x130.md` |
| 2026-04-21 17:18 | Closed `X-132` as `positive`: after `X-131`, the control plane is now clean enough that another abstract non-graybox reselection pass would be lower value than returning to the current `04-defense` active slot, because `H2 privacy-aware adapter` is already prototype-implemented, test-backed, and smoke-backed while still missing the first canonical `diffaudit` contract stage frozen in `X-119`; therefore the highest-value next live lane becomes `X-133 04-H2 probe-h2-assets executable contract start`, with `active_gpu_question = none` and `next_gpu_candidate = none`; canonical anchor is `workspaces/implementation/2026-04-21-x132-non-graybox-next-lane-reselection-after-x131.md` |
| 2026-04-21 17:34 | Closed `X-133` as `positive but bounded`: `04-H2` now has the first landed canonical `diffaudit` stage, because `probe-h2-assets` is implemented in `src/diffaudit/defenses/h2_adapter.py`, exposed from `src/diffaudit/cli.py`, covered by `tests/test_h2_adapter.py`, and a real admitted-asset probe at `workspaces/implementation/runs/h2-probe-assets-20260421-r1/summary.json` resolves `checkpoint-9600/model.safetensors`, full-scan `1000 / 1000` image roots, and `32 x 32 x 3` `RGB` compatibility under `packet_cap = 1000`; `H2` therefore advances from `prototype-only` to `probe-landed`, but still stays below run/review readiness, `active_gpu_question = none`, `next_gpu_candidate = none`, and the next live lane becomes `X-134 04-H2 prepare-h2-contract minimal surface freeze`; canonical anchor is `workspaces/implementation/2026-04-21-x133-04-h2-probe-assets-contract-landing.md` |
| 2026-04-21 17:40 | Closed `X-134` as `positive but bounded`: `04-H2` now has the second landed canonical `diffaudit` stage, because `prepare-h2-contract` is implemented in `src/diffaudit/defenses/h2_adapter.py`, exposed from `src/diffaudit/cli.py`, covered by `tests/test_h2_adapter.py`, and the first real admitted-asset workspace at `workspaces/implementation/runs/h2-prepare-contract-20260421-r1/summary.json` freezes checkpoint identity, packet identity, and runtime defaults (`rank = 4`, `lambda_coeff = 0.5`, `num_epochs = 10`, `batch_size = 8`) without overstating itself as a training run; `H2` now reads `probe + prepare landed / run + review missing`, `active_gpu_question = none`, `next_gpu_candidate = none`, and the next live lane becomes `X-135 04-H2 run-h2-defense-pilot bounded execution contract start`; canonical anchor is `workspaces/implementation/2026-04-21-x134-04-h2-prepare-contract-landing.md` |
| 2026-04-21 17:56 | Closed `X-135` as `positive but bounded`: `04-H2` now has the third landed canonical `diffaudit` stage, because `run-h2-defense-pilot` is implemented in `src/diffaudit/defenses/h2_adapter.py`, exposed from `src/diffaudit/cli.py`, covered by `tests/test_h2_adapter.py`, and one real admitted-asset bounded pilot now exists at `workspaces/implementation/runs/h2-run-defense-pilot-20260421-r1/summary.json`; that packet keeps `active_gpu_question = none`, `next_gpu_candidate = none`, stages only `1 / 1` images under the frozen `1000 / 1000` asset identity, reuses `checkpoint-9600/model.safetensors`, and emits workspace-root config/log/checkpoint artifacts plus first-step training metrics, but still provides no attack-side board or low-FPR readout; `H2` therefore now reads `probe + prepare + run landed / review missing`, and the next live lane becomes `X-136 04-H2 review-h2-defense-pilot same-packet attack-side review contract start`; canonical anchor is `workspaces/implementation/2026-04-21-x135-04-h2-run-defense-pilot-contract-landing.md` |
| 2026-04-21 18:05 | Closed `X-136` as `negative but useful`: `04-H2` now has the fourth landed canonical `diffaudit` stage, because `review-h2-defense-pilot` is implemented in `src/diffaudit/defenses/h2_adapter.py`, exposed from `src/diffaudit/cli.py`, covered by `tests/test_h2_adapter.py`, and one real same-packet review now exists at `workspaces/implementation/runs/h2-review-defense-pilot-20260421-r1/summary.json`; that review reuses the frozen `X-135` `1 / 1` packet and a borrowed `GSA` shadow loss-score packet, compares the baseline target checkpoint against a merged review-compatible defended checkpoint, and emits the mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` bundle, but all transferred metrics remain `0.0`, so `H2` is now minimally contract-complete yet still null on the first bounded attack-side read; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next live lane becomes `X-137 non-graybox next-lane reselection after X-136 same-packet review`; canonical anchor is `workspaces/implementation/2026-04-21-x136-04-h2-review-defense-pilot-contract-landing.md` |
| 2026-04-21 18:08 | Closed `X-137` as `positive`: after `X-136`, `H2` no longer lacks contract stages, but the first same-packet review is still transfer-only and too small (`1 / 1`) to change project-level story, so the highest-value next live lane is not GPU release, stale-entry sync, or immediate lane-yield; it becomes `X-138 04-H2 bounded packet-scale follow-up selection after X-137 reselection`, with `active_gpu_question = none` and `next_gpu_candidate = none`; canonical anchor is `workspaces/implementation/2026-04-21-x137-non-graybox-next-lane-reselection-after-x136.md` |
| 2026-04-21 16:25 | Closed `X-138` as `positive`: after `X-137`, `H2` is no longer blocked on contract construction, but the first `1 / 1` transfer-only board is still too degenerate to justify immediate lane-yield; because `review-h2-defense-pilot` consumes the staged packet recorded by `run-h2-defense-pilot`, the honest next question is one minimal same-contract packet enlargement rather than document-only speculation, and `4 / 4` is the smallest bounded rung that can test whether the earlier all-zero read was pure packet sparsity; `active_gpu_question = none`, `next_gpu_candidate = none`, and the next live lane becomes `X-139 04-H2 4x4 bounded follow-up review`; canonical anchor is `workspaces/implementation/2026-04-21-x138-04-h2-bounded-packet-scale-followup-selection.md` |
| 2026-04-21 16:25 | Closed `X-139` as `negative but useful`: the selected minimal `4 / 4` follow-up is now real on admitted CIFAR10 assets, because `run-h2-defense-pilot` and `review-h2-defense-pilot` both land cleanly on the same frozen checkpoint/asset contract at `workspaces/implementation/runs/h2-run-defense-pilot-4x4-20260421-r1/summary.json` and `workspaces/implementation/runs/h2-review-defense-pilot-4x4-20260421-r1/summary.json`; this removes the pure-zero degeneracy of the first `1 / 1` board by producing non-null target-transfer metrics (`AUC = 0.5 / ASR = 0.375 / TPR@1%FPR = 0.5 / TPR@0.1%FPR = 0.5`), but baseline and defended metrics still remain exactly equal, so `H2` now reads `minimal contract-complete + bounded 4/4 follow-up negative but useful`, still below promotion and still below `next_gpu_candidate`; the next immediate move becomes `X-140 cross-box / system-consumable stale-entry sync after X-139`; canonical anchor is `workspaces/implementation/runs/h2-review-defense-pilot-4x4-20260421-r1/summary.json` |
| 2026-04-21 16:25 | Closed `X-140` as `positive`: after `X-139`, the active entry surfaces still pointed fresh sessions to `X-138` as if packet-scale follow-up were pending, so the highest-value immediate move was one bounded stale-entry sync across `README`, `comprehensive-progress`, `reproduction-status`, `mainline-narrative`, `research-autonomous-execution-prompt`, `challenger-queue`, and the roadmap itself; those surfaces now carry the sharper `H2` truth (`minimal contract-complete + bounded 4/4 follow-up negative but useful`, `no current next_gpu_candidate`, and `04` should yield the next `CPU-first` slot absent a genuinely new bounded hypothesis`), while `active_gpu_question = none` and `next_gpu_candidate = none`; the next live lane becomes `X-141 non-graybox next-lane reselection after X-140 stale-entry sync`; canonical anchor is `workspaces/implementation/2026-04-21-x140-crossbox-system-sync-after-x139.md` |
| 2026-04-21 16:25 | Closed storage-boundary audit as `positive`: the repository now has one explicit storage rule (`external` = upstream/exploratory code clones, `third_party` = minimal vendored code, `Download` = raw intake bundles, `workspaces/*/assets` = lane-normalized gateways, `workspaces/*/runs` = evidence), and `workspaces/README.md` is resynced to that rule; but the audit also freezes three real live inconsistencies as future cleanup targets: `external/recon-assets` is asset-heavy and belongs below `Download` rather than `external`, `external/SecMI` vs `third_party/secmi` still needs visible canonical/exploratory separation, and `external/CLiD` still mixes upstream code with supplementary-style outputs while `Download` also stores a CLiD supplementary mirror; canonical anchor is `workspaces/implementation/2026-04-21-research-storage-boundary-audit.md` |
| 2026-04-23 09:00 | Closed storage-boundary cleanup pass as `positive`: the sharpest live inconsistency is now actually resolved, because `external/recon-assets` has been physically migrated to `D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets`, active recon bundle references have been retargeted, `README` now points CLiD artifact summarization at the raw supplementary mirror under `Download`, and new role markers (`external/README.md`, `third_party/README.md`, `Download/README.md`, plus the tracked `third_party/secmi/LOCAL_ROLE.md`) now make the remaining dual-surface cases explicit instead of ambiguous; the repository storage rule is therefore materially more uniform even though some historical notes still preserve old path wording as historical context; canonical anchor is `workspaces/implementation/2026-04-21-research-storage-boundary-audit.md` |
| 2026-04-26 00:00 | Closed storage-layout finalization as `positive`: the previously documented storage boundary is now physically true, because `Research/external/downloads` has been removed and its raw-intake subtrees (`black-box`, `gray-box`, `manifests`, `shared`, `white-box`) now live directly under `D:\Code\DiffAudit\Download\`; verification confirmed `Download\black-box\supplementary\recon-assets`, `Download\manifests\research-download-manifest.json`, and the shared dataset/weight roots are present, while active path search no longer finds live `Research/external/downloads` or `external/CLiD/inter_output` references; canonical anchor is `workspaces/implementation/2026-04-26-research-storage-layout-finalization.md` |

---

## 11. Archived Roadmaps

- `legacy/2026-04-15-P0-P3-completed-roadmap.md`
- `legacy/2026-04-15-competition-sprint-roadmap-archived.md`

