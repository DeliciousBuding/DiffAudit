# DiffAudit Research — Challenger Queue

> **Last refreshed**: 2026-04-21 16:25
> **Purpose**: Keep the innovation funnel aligned with current repo truth after closure-round reviews
> **Rule**: This queue is for future candidate generation, bounded follow-up, or asset-triggered reopen only. It is not a substitute for admitted/mainline status.

---

## Current Queue Truth

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-141 non-graybox next-lane reselection after X-140 stale-entry sync` (active CPU-first control lane)
- `04` current control read is now:
  - `H1 risk-targeted SISS / retain-forget mixture` = executed family with real prep / pilot / review surface
  - original `k16` = current best working instantiation
  - `k16 + alpha-up` and `k16 + mixture_lambda-down` are both closed `negative but useful`
  - `H2 privacy-aware adapter` = `minimal contract-complete + bounded 4/4 follow-up negative but useful`
  - the minimal `4 / 4` follow-up removes the pure-zero degeneracy from the first `1 / 1` board, but baseline vs defended deltas remain exactly `0.0`
  - so `H2` should now yield the next `CPU-first` slot unless a genuinely new bounded hypothesis appears
- `05-cross-box` remains a promoted evidence line, not the next open rerun slot
- `02` remains sidecar only:
  - `SimA` = `execution-feasible but weak`
  - plain `SimA` scorer reopen = `not-requestable without new bounded hypothesis`
  - `SimA` packet-score export = landed and pairboard-ready
  - first bounded `PIA + SimA` full-overlap pairboard = landed
  - best fused candidate = `logistic_2feature`
  - `AUC / ASR` gains are real and `TPR@1%FPR` is partially improved
  - `TPR@0.1%FPR` still has no stable lift
  - gray-box should now yield the next `CPU-first` slot
- `03` remains medium-horizon activation-subspace / risky-subspace preparation; `01` remains parked; `06` remains governance fallback
- `X-114` and `X-115` are now closed: current `04` truth changes control-plane ordering only and still does not require any new Runtime / Platform consumer contract
- older `X-86→X-91` and prior queue history below should now be read as archived carry-forward context rather than the current live steering layer
- `I-D.1` is now closed as `positive but bounded` on the frozen `Stable Diffusion v1.5 base + celeba_partial_target/checkpoint-25000` local canary contract
- `I-D.2` is now closed as `positive but bounded` on a real single-GPU `8 / 8 / 8` packet, with `generation_guidance_scale = 7.5` widening raw separation over `3.5` but fixed thresholds failing to transfer across scales
- `I-D.3` is now closed as `positive but bounded`: hidden-guidance jitter disrupts attack calibration on average, but current mixed-packet evidence is still seed-sensitive and below low-FPR / adaptive release
- `X-14` is now closed as `positive`: the first bounded `I-D` packet pair is synchronized into higher-layer truth
- `I-D.4` is now closed as `negative but useful`: the first actual runner-level deterministic hidden-guidance-jitter rerun is reproducible, but its honest frozen-packet operating point (`accuracy = 0.5 / TPR = 0.625 / FPR = 0.625`) falls below low-FPR and adaptive-release honesty
- the `I-A` refresh after negative actual `I-D.4` is now closed as `positive but stabilizing`: the failed conditional-defense rerun narrows a future-surface branch but does not weaken the mechanistic `PIA + stochastic-dropout` packet, so `I-A` remains the strongest near-term innovation track
- `X-15` is now closed as `positive`: the next honest non-graybox `CPU-first` lane is `I-B.5 first bounded localization/intervention packet selection`
- `I-B.5` is now closed as `positive`: the first executable `I-B` packet is a white-box-only, CPU-only, in-model attenuation review on the native `965 / 467` pair rather than an imported `I-C` bridge packet
- `I-B.6` is now closed as `positive but bounded`: the first executable packet runs successfully and emits local movement plus downstream drift metrics, but it is still execution-positive rather than defense-positive because no attack-side evaluation bundle is attached yet
- `I-B.7` is now closed as `positive`: the first honest attack-side review should stay on admitted `GSA` assets with one bounded evaluation-size override rather than full-scale rerender or duplicate subset assets
- `I-B.8` is now closed as `positive but bounded`: the admitted `GSA` surface now has a real bounded attack-side evaluation control via `--max-samples`, and the first CPU-only `max_samples = 64` review packet on reused admitted gradients lands at `AUC = 0.988159 / ASR = 0.90625 / TPR@1%FPR = 0.453125 / TPR@0.1%FPR = 0.0` with `target_eval_size = 128`, which is control-positive rather than defense-positive
- `I-B.9` is now closed as `positive`: the first honest intervention-on/off review is a target-anchored fixed-mask dual-run bounded board on the same `max_samples = 64` packet, with one frozen mask object replayed across target and shadow extractions and no per-model reselection
- `I-B.10` is now closed as `positive but bounded`: the repo now exposes `run-gsa-runtime-intervention-review`, which assembles one bounded baseline/intervened dual-run review surface with `metric_deltas` and `locality_anchor`, but no admitted real-asset packet has been executed yet
- `I-B.11` is now closed as `blocked but useful`: current boundedness is evaluation-only; the first admitted packet would still traverse the full admitted `8000`-image board twice, so it is not yet an honest GPU release
- `I-B.12` is now closed as `positive but bounded`: extraction-side boundedness now exists on the dual-run review surface via `extraction_max_samples`, with fallback from `max_samples`, so the execution-budget blocker is cleared even though the admitted packet itself is still unrun
- `I-B.13` is now closed as `positive`: the first truly bounded admitted `64 / 64` dual-run packet is host-fit and honest to release on admitted `GSA` assets
- `I-B.14` is now closed as `negative but useful`: the first actual bounded admitted packet is real and cleanly executed, but it nudges attack-side metrics upward rather than downward
- `I-B.15` is now closed as `negative but clarifying`: the current `Finding NeMo / I-B` branch should now be read as an `actual bounded falsifier`, not `zero-GPU hold` and not defense-positive
- `X-16` is now closed as `positive`: after the sharper `I-B` falsifier, the next honest non-graybox main lane is cross-box / system-consumable sync rather than another box-local reopen
- `X-17` is now closed as `positive`: higher-layer leader/materials entry points now carry the updated `Finding NeMo` boundary without changing admitted tables
- the `I-A` refresh after the first actual negative `I-B` packet is now closed as `positive but stabilizing`: the new white-box falsifier narrows a competing branch but does not weaken the mechanistic `PIA + stochastic-dropout` packet
- `XB-CH-2` refresh is now closed as `negative but useful`: the cross-box transfer / portability branch is still blocked on missing paired model contracts, paired split contracts, and one bounded shared-surface hypothesis
- `X-19` is now closed as `positive`: the next honest move after the refreshed transfer blocker is a bounded stale-entry sync pass rather than another empty reselection loop
- `X-20` is now closed as `positive`: Research-side entry docs and the root control board are now aligned to the same current truth, so the next honest move is another non-graybox reselection pass rather than more wording-only sync
- `X-21` is now closed as `positive`: after `X-20`, there is still no honest executable reopen in `XB-CH-2`, `GB-CH-2`, or other blocked/hold queue items, so the strongest next non-graybox lane is a bounded return to `I-A`
- `X-22` is now closed as `positive`: the current `I-A` residue sat in higher-layer presentation strength rather than the underlying packet, and leader/material entry docs now carry the stronger four-metric plus bounded-adaptive reading
- `X-23` is now closed as `positive`: after `X-22`, the remaining issue is no longer mechanism truth but one last stale execution-order layer in higher docs, so the next honest move is a bounded residual stale-entry cleanup pass
- `X-24` is now closed as `positive`: the last visible stale execution-order layer in `mainline-narrative` and the root control board is now aligned, so the next honest move is another non-graybox reselection pass
- `X-25` is now closed as `positive`: after `X-24`, no blocked or hold non-graybox branch gained a stronger executable reopen, so `PIA provenance maintenance` should be promoted from sidecar into the next bounded main lane
- `X-26` is now closed as `positive`: `PIA` provenance maintenance no longer needs a new consumer field or execution release, and the blocker is now frozen to explicit carry-forward boundary + reopen-trigger maintenance
- `X-27` is now closed as `positive`: once provenance maintenance is frozen, `XB-CH-2` becomes the highest-value unresolved non-graybox branch again, but only as a CPU-side blocker/contract review rather than an execution reopen
- `X-28` is now closed as `positive`: `XB-CH-2` still lacks one honest shared execution surface, but its reopen contract is now frozen to four explicit requirements: paired model contract, paired split contract, shared metric hypothesis, and bounded packet budget
- `X-29` is now closed as `positive`: once `XB-CH-2` is blocker-frozen, the strongest immediate executable non-graybox lane becomes a bounded return to `I-A`, while cross-box wording maintenance downgrades back to sidecar status
- `X-30` is now closed as `positive but stabilizing`: the current `I-A` mechanistic / low-FPR / bounded-adaptive wording is stable enough that no new higher-layer residue is visible right now
- `X-31` is now closed as `positive`: the remaining post-`X-30` drift was only stale control-plane wording, not a new research-side uncertainty
- `X-32` is now closed as `positive`: once `I-A` is back to sidecar and blocked/hold non-graybox branches still do not reopen honestly, the strongest executable main-lane move is to promote `cross-box / system-consumable wording maintenance` back into the main slot because active intake/system surfaces still drift on `Finding NeMo` and `Phase E`
- `X-33` is now closed as `positive`: active higher-layer and machine-readable intake surfaces no longer encode `Finding NeMo` as the current `Phase E` intake-only `zero-GPU hold` candidate; `PIA paper-aligned confirmation` remains document-layer conditional only, and there is now no active `Phase E` intake review priority order
- `X-34` is now closed as `positive`: after both control-plane and intake-plane stale surfaces were cleared, the visible non-graybox candidate pool still contains no honest ready main-slot lane above blocked/hold branches or stable sidecar maintenance, so the next correct move is bounded candidate-surface expansion rather than another fake reopen
- `X-35` is now closed as `positive`: the first honest way to expand the stale non-graybox pool is to restore `I-D` as an active candidate surface, because black-box still lacks a new family contract, white-box still lacks a distinct defended-family import lane, and `I-C` still lacks a genuinely new bounded cross-box hypothesis beyond its frozen translated-contract falsifier
- `X-36` is now closed as `positive`: the restored `I-D` surface does not currently contain one genuinely new bounded successor lane, because same-contract scale tuning would be parameter churn, hidden-guidance-jitter salvage would be failed-contract rescue, and wider conditional families would break the frozen `I-D.1` boundary; the honest result is therefore to freeze `I-D` back to future-surface support only and yield to non-graybox reselection
- `X-37` is now closed as `positive`: after that `I-D` yield, no blocked/hold non-graybox branch honestly reopened and no new `I-A` successor question appeared, so the strongest immediate move became one bounded cross-box / system-consumable stale-surface sync pass on the remaining active `I-D` material wording drift
- `X-38` is now closed as `positive`: the remaining active material-facing `I-D` wording surface is now aligned to current truth, so higher-layer readers no longer see the old `bounded hidden-guidance defense idea` phrasing without the sharper negative runner-level and no-successor boundary
- `X-39` is now closed as `positive`: after `X-38`, the visible non-graybox pool still contains no honest ready main-slot lane above blocked/hold branches or stable sidecar maintenance, so the correct next move is another bounded candidate-surface expansion rather than a fake reopen
- `X-40` is now closed as `positive`: the first honest way to expand the stale non-graybox pool again is to restore `I-C`, but only as fresh bounded cross-box hypothesis generation rather than same-pair translated-contract hardening
- `X-41` is now closed as `positive`: the first genuinely new bounded `I-C` hypothesis after the translated-falsifier freeze is a bounded multi-pair agreement-first cross-box hypothesis, not another translated intervention retry
- `X-42` is now closed as `blocked but useful`: the agreement-first idea survives, but the first executable board contract still lacks one second member/nonmember pair freeze under the same overlap authority
- `X-43` is now closed as `positive but bounded`: the second pairboard identity is now frozen deterministically to member `8` and nonmember `23`, so the new `I-C` line now has a complete `2 member + 2 nonmember` identity board under the same overlap authority, but still no executable agreement-board contract
- `X-44` is now closed as `blocked but useful`: the identity board is now complete and gray-box already exposes object-level scores, but the first honest agreement-board contract still lacks one frozen white-box board-local concentration scalar and selector policy
- `X-45` is now closed as `positive but bounded`: the white-box blocker is now resolved by freezing `selected_channel_abs_profile_mean` on one board-wide selected-channel set inherited from the pair-A selector, so the fresh `I-C` line can now return to an actual four-object board read
- `X-46` is now closed as `negative but useful`: the first honest four-object board read still shows class-mean same-direction residue, but it does not preserve a clean enough same-object broad order to count as positive agreement-first support
- `X-47` is now closed as `positive`: after that negative-but-useful first fresh `I-C` board read, the most honest immediate next move is one bounded stale-entry sync pass rather than same-board salvage or immediate `I-A` promotion
- `X-48` is now closed as `positive`: the active higher-layer entry docs are now aligned again to the post-`X-46` control-plane truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only work
- `X-49` is now closed as `positive`: once stale-entry sync is cleared, no stronger ready non-graybox branch reopens above `I-A`, so the next honest main-slot move is a bounded return to `I-A` truth-hardening
- `X-50` is now closed as `positive`: the remaining `I-A` residue sat in higher-layer carry-forward rather than in the packet itself, and the active higher-layer entry docs now again carry the bounded repeated-query adaptive reading plus mandatory four-metric low-FPR reporting
- `X-51` is now closed as `positive`: once that `I-A` residue is cleared, no blocked/hold non-graybox branch reopens above one remaining materials-facing stale-entry sync, because `competition-evidence-pack` still encodes `SecMI` and `TMIA-DM` with pre-refresh gray-box status
- `X-52` is now closed as `positive`: the active admitted/material-facing evidence pack is now aligned again to current gray-box truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only sync
- `X-53` is now closed as `positive`: once the materials-facing stale entry is cleared, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but `I-B` is now the strongest innovation surface that has not yet received an explicit post-falsifier successor review, so the next honest move is to inspect `I-B` for any genuinely new bounded successor lane rather than to re-promote `I-A` mechanically
- `X-54` is now closed as `negative but useful`: the restored `I-B` surface still does not contain one honest bounded successor lane above its `actual bounded falsifier` freeze, because same-family rescue remains forbidden, distinct-family import is unavailable, and no new bounded localization-defense hypothesis is visible
- `X-55` is now closed as `positive`: once `I-B` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but fresh `I-C` is now the strongest innovation surface still lacking an explicit post-negative successor review after `X-46`, so the next honest move is to inspect `I-C` for any genuinely new bounded successor lane rather than to force a return to `I-A`
- `X-56` is now closed as `negative but useful`: the fresh `I-C` surface still does not contain one honest bounded successor lane above the first negative agreement-board read, because same-board salvage remains forbidden, black-box corroboration still lacks a frozen bridge surface, and no new bounded cross-permission hypothesis is visible
- `X-57` is now closed as `positive`: once `I-C` is successor-frozen, no blocked/hold branch honestly reopens above the stable `I-A` sidecar, but one active higher-layer entry doc still lags behind the current control-plane state, so the next honest move is one bounded stale-entry sync pass rather than immediate candidate-surface expansion
- `X-58` is now closed as `positive`: the remaining active higher-layer entry doc is now aligned again to the post-`X-56` control-plane truth, so the next honest move is a fresh non-graybox reselection rather than more wording-only work
- `X-59` is now closed as `positive`: once the stale-entry sync is cleared, no blocked/hold branch honestly reopens above the stable `I-A` sidecar and no fresh successor lane appears inside `I-B` or `I-C`, so the next honest move is bounded non-graybox candidate-surface expansion rather than a forced same-family return
- `X-60` is now closed as `positive`: the restored non-graybox candidate surface is not another gray-box return or same-family reopen, but `black-box paper-backed next-family scouting`, because black-box is the only remaining non-graybox area with honest CPU-only expansion room that does not violate current frozen negative or `needs-assets` boundaries
- `X-61` is now closed as `negative but useful`: the paper-backed black-box backlog still does not expose one genuinely new promotable family, because the remaining face-image LDM route is domain-specific, collection-level, structurally overlaps with `semantic-auxiliary-classifier` and gray-box-owned `CDI`, and lacks a bounded local execution contract
- `X-62` is now closed as `positive`: once black-box scouting also closes negative, no blocked/hold non-graybox branch honestly reopens and no immediate stale-entry sync need remains, so the strongest next live lane returns to `I-A` truth-hardening rather than another artificial candidate-surface expansion
- `X-63` is now closed as `positive`: the remaining `I-A` residue was materials-facing rather than experiment-facing, and the active PIA visual prompt no longer invites `AUC-only` defense storytelling without four-metric and bounded-adaptive carry-forward
- `X-64` is now closed as `positive`: once `X-63` clears the last visible `I-A` residue and no blocked/hold branch or stale-entry sync pass reopens above sidecar maintenance, the honest next move is another bounded candidate-surface expansion rather than a forced `I-A` rerun or white-box reopen
- `X-65` is now closed as `positive`: the restored non-graybox candidate surface is `I-B paper-backed localization-defense successor scouting`, because black-box already closed negative, white-box still should not take the next slot, and the broader `Finding NeMo + local memorization + FB-Mem` mechanism stack still contains CPU-only hypothesis-generation room above same-family rescue churn
- `X-66` is now closed as `negative but useful`: the broadened `Finding NeMo + local memorization + FB-Mem` stack still does not expose one genuinely new bounded successor hypothesis on top of the current `actual bounded falsifier`, because the extra material remains historical intake scaffolding, observability plumbing, or paper-faithful `SD1.4/LAION` context rather than a current admitted-surface hypothesis
- `X-67` is now closed as `positive`: once the broadened `I-B` stack also freezes, no stronger blocked/hold branch honestly reopens above the stable sidecar line, so the strongest next live lane returns to `I-A` truth-hardening rather than another immediate expansion
- `X-68` is now closed as `positive but stabilizing`: the current `I-A` contract still had one real carry-forward task, but it was only a narrow higher-layer residue in the `Leader` one-page summary table, which still foregrounded `AUC / ASR` before low-FPR and bounded-adaptive reading
- `X-69` is now closed as `positive`: once that last top-summary `I-A` residue is cleared, no blocked/hold branch honestly reopens above sidecar maintenance, so the next honest move is bounded non-graybox candidate-surface expansion rather than another `I-A` turn
- `X-70` is now closed as `positive`: the next honest restored non-graybox candidate surface is `WB-CH-4 white-box loss-feature challenger family`, because the visible white-box pool was exhausted but the paper-backed loss-feature family (`LSA* / LiRA / Strong LiRA`) had never been promoted into the candidate queue even though it is distinct from the current `GSA` gradient family
- `X-71` is now closed as `positive but bounded`: the restored `WB-CH-4` surface does contain one honest near-term lane, but only as a bounded same-asset `LSA*`-style contract review; `LiRA / Strong LiRA` remain above current bounded host-fit budget
- `X-72` is now closed as `positive but bounded`: current admitted `DDPM/CIFAR10` white-box assets do support one bounded same-asset `LSA*`-style loss-feature contract, but the current admitted runtime/mainline still emits gradients only, so the next honest lane is a bounded loss-score export surface review rather than direct execution
- `X-73` is now closed as `positive but bounded`: one honest bounded loss-score export surface does exist, and the preferred first path is a separate in-repo internal helper / CLI surface rather than patching the upstream external extractor or mutating current admitted runtime-mainline semantics
- `X-74` is now closed as `positive but bounded`: the separate in-repo internal loss-score export surface is now implemented and one bounded real-asset smoke succeeded, so the next blocker is first packet selection / evaluation contract rather than export capability
- `X-75` is now closed as `positive but bounded`: the first honest bounded packet is now frozen to a threshold-style, shadow-oriented, shadow-threshold-transfer board on exported scalar loss scores with `extraction_max_samples = 64` per split, while low-FPR reporting remains mandatory but still below release-grade honesty at that bounded scale
- `X-76` is now closed as `positive but bounded`: the separate threshold evaluator surface now exists, and one real bounded smoke proves why verdicts must come from pooled-shadow orientation/threshold transfer rather than the target self-board
- `X-77` is now closed as `positive but bounded`: the first real `64`-per-split actual packet now exists and stays positive under the intended shadow-only transfer contract, but its low-FPR values remain too weak for release-grade promotion
- `X-78` is now closed as `positive but stabilizing`: the first actual packet is now boundary-frozen as bounded auxiliary evidence, because the branch is real and positive but still weak at low FPR and lacks a genuinely new same-family follow-up hypothesis
- `X-79` is now closed as `positive`: once the white-box loss-score branch is frozen and current higher-layer sync is already aligned, no stronger non-graybox branch honestly reopens above the carry-forward sidecar, so the next live lane returns to `I-A`
- `X-80` is now closed as `positive`: one active higher-layer residue still existed in `mainline-narrative.md`, and it is now cleared
- `X-81` is now closed as `positive`: once that `I-A` residue is cleared, the strongest next move is one bounded stale-entry sync pass rather than another immediate candidate-surface reopen
- `X-82` is now closed as `positive`: the active higher-layer stale-entry surfaces are now aligned again to current control-plane truth
- `X-83` is now closed as `positive`: once the stale-entry sync clears, no blocked/hold branch honestly reopens and no new active `I-A` residue is visible, so the next honest move becomes bounded non-graybox candidate-surface expansion
- `X-84` is now closed as `positive`: the restored candidate surface is `cross-box admitted-summary quality/cost read-path hardening`
- `X-85` is now closed as `positive`: the admitted summary now explicitly exposes evidence level and quality/cost fields alongside admitted metrics
- gray-box currently has no immediate next-family execution lane
- white-box currently has no immediate next-hypothesis execution lane
- white-box now does have one fresh paper-backed candidate surface below release:
  - `WB-CH-4 white-box loss-feature challenger family`
  - but its honest near-term entry point is now `bounded first actual packet execution on the frozen shadow-transfer contract`, not immediate promotion into admitted mainline execution
- the next live CPU-first reselection step is no longer pending:
  - `X-12` already froze the next live lane after the translated `I-C` falsifier, and the current translated packet family has now produced both:
    - one executability canary
    - one negative falsifier
  - current `I-C` truth is therefore:
    - `translated-contract-only`
    - below support
    - below GPU release
    - reopen only on a genuinely new bounded hypothesis

This queue should now be read with three distinctions:

1. `ready-for-selection`
   - honest CPU-first candidate-generation or review work
2. `hold / not-requestable`
   - real line exists, but current repo truth says do not schedule execution
3. `needs-assets`
   - may still be valuable, but cannot honestly progress without new data/models/contracts

---

## Top 3 Priorities

### 1. `X-141` non-graybox next-lane reselection after `X-140` stale-entry sync

- `status`: `active / cpu-first control lane`
- `expected value`: ⭐⭐
- `mode`: `reselection / post-H2 lane review`
- `why now`:
  - `X-139` already shows that one minimal `4 / 4` follow-up still carries zero defended-vs-baseline deltas
  - `X-140` has already cleared the stale entry-doc layer
  - the next question is now which non-graybox lane should take the freed `CPU-first` slot

### 2. `X-140` cross-box / system-consumable stale-entry sync after `X-139`

- `status`: `completed / positive`
- `expected value`: ⭐⭐
- `mode`: `stale-entry sync / higher-layer alignment`
- `why it mattered`:
  - active entry docs still pointed to `X-138` as if the follow-up were pending
  - the sharper `H2` truth is now visible again across README, progress, narrative, prompt, queue, and roadmap

### 3. `X-139` `04-H2` 4x4 bounded follow-up review

- `status`: `completed / negative but useful`
- `expected value`: ⭐⭐
- `mode`: `bounded packet-scale follow-up review`
- `why it mattered`:
  - it proved the lane is not only blocked by `1 / 1` degeneracy
  - it also showed that even after minimal enlargement the defended-vs-baseline deltas are still exactly `0.0`

### 4. `X-135` `04-H2` run-h2-defense-pilot bounded execution contract start

- `status`: `completed / positive but bounded`
- `expected value`: ⭐⭐
- `mode`: `contract implementation / prepare`
- `why it mattered`:
  - it froze checkpoint identity, packet identity, and runtime defaults into one machine-readable workspace manifest
  - it moved `H2` from `contract-incomplete` to `probe + prepare landed / run + review missing`

### 4. `X-133` `04-H2` probe-h2-assets executable contract start

- `status`: `completed / positive but bounded`
- `expected value`: ⭐⭐
- `mode`: `contract implementation / asset probe`
- `why it mattered`:
  - it landed the first canonical `diffaudit` H2 stage on admitted assets
  - it froze `checkpoint-9600/model.safetensors` plus full-scan `1000 / 1000` `32 x 32 x 3` `RGB` compatibility under `packet_cap = 1000`

---

## Current Candidates

### Black-box

#### `BB-CH-1` Caption/semantic-family refresh

- `status`: `reviewed / closed-negative`
- `expected value`: ⭐
- `reason`:
  - the refresh review found no honest ready next-family promotion candidate
  - visible options collapse into same-family continuation, boundary-only work, needs-assets, or gray-box-owned audit expansion
  - do not reopen black-box candidate generation unless a real new family or asset/boundary shift appears

#### `BB-CH-4` Paper-backed next-family scouting

- `status`: `reviewed / closed-negative`
- `expected value`: ⭐⭐
- `reason`:
  - `X-61` already audited the remaining paper backlog and found no genuinely new promotable family
  - the remaining face-image LDM route is still domain-specific, collection-level, and below local bounded execution contract
  - keep this lane closed unless a distinct bounded family contract or new assets appear

#### `BB-CH-2` Variation real-asset unblock

- `status`: `needs-assets`
- `expected value`: ⭐⭐
- `blocker`:
  - `query_image_root / query images`
  - plus endpoint/proxy, budget, and frozen parameters

#### `BB-CH-3` Semantic-auxiliary-classifier scoring follow-up

- `status`: `hold`
- `expected value`: ⭐
- `reason`:
  - current scoring/fusion review already closed as `negative but useful`
  - do not reopen without a genuinely new feature-family hypothesis

### Gray-box

#### `GB-CH-1` Second gray-box defense mechanism

- `status`: `reviewed / selected`
- `expected value`: ⭐⭐
- `note`:
  - selected mechanism is `TMIA-DM late-window + temporal-striding(stride=2)`
  - keep it as the second gray-box defense mechanism, not as a project-wide replacement defense
  - do not reopen gray-box defense selection unless a genuinely different mechanism appears

#### `GB-CH-2` Ranking-sensitive variable search

- `status`: `hold`
- `expected value`: ⭐⭐
- `note`:
  - the concrete switching hypothesis has now been executed as a bounded offline packet
  - current packet is `negative but useful`
  - do not reopen unless a genuinely new gating signal appears

#### `GB-CH-3` Noise as a Probe latent-diffusion promotion path

- `status`: `hold`
- `expected value`: ⭐
- `reason`:
  - promotion-gap review already closed negatively
  - contract-shift review also closed negatively
  - do not reopen unless a real contract shift appears

### White-box

#### `WB-CH-1` Distinct defended-family import / selection

- `status`: `reviewed / closed-negative`
- `expected value`: ⭐
- `note`:
  - no distinct import-ready defended family is available in the current round
  - `GSA2` stays same-family corroboration
  - `DP-LoRA` stays bounded branch continuation
  - `Finding NeMo` stays a non-admitted executed falsifier branch, not a distinct defended family

#### `WB-CH-2` `Finding NeMo`

- `status`: `not-requestable`
- `expected value`: ⭐⭐
- `reason`:
  - protocol mismatch on current admitted assets
  - still held under separate future reconsideration boundary

#### `WB-CH-3` `GSA2`

- `status`: `hold`
- `expected value`: ⭐
- `reason`:
  - positive same-family corroboration already exists
  - does not create a new white-box family

#### `WB-CH-4` White-box loss-feature challenger family

- `status`: `bounded-auxiliary / actual-packet-landed / boundary-frozen`
- `expected value`: ⭐⭐
- `reason`:
  - this paper-backed family (`LSA* / LiRA / Strong LiRA`) is distinct from the current `GSA` gradient family
  - it has not yet been promoted into the candidate queue even though the main white-box paper benchmarks it directly
  - `X-72` already confirmed that current admitted `DDPM/CIFAR10` assets support one bounded same-asset `LSA*`-style contract
  - `X-73` already selected the preferred surface as a separate in-repo internal helper / CLI path
  - `X-74` already implemented and smoke-validated that surface on the current admitted asset family
  - `X-75` already froze the first honest packet/evaluation contract
  - `X-77` already landed the first actual packet on top of the exported-score evaluator surface
  - `X-78` already boundary-froze that packet as bounded auxiliary evidence rather than a promoted new line
  - do not escalate the whole family into execution release; `LiRA / Strong LiRA` remain above current bounded host-fit budget and no new same-family bounded hypothesis is currently frozen

### Cross-box

#### `XB-CH-1` Cross-threat-model agreement follow-up

- `status`: `hold`
- `expected value`: ⭐⭐
- `reason`:
  - current agreement review already landed positively
  - reopen only if a new box-level verdict changes the project narrative

#### `XB-CH-2` Transfer / portability probes

- `status`: `needs-assets`
- `expected value`: ⭐⭐
- `reason`:
  - still lacks honest paired model/dataset contracts

---

## Negative / Closed Items To Not Reopen Blindly

- `PIA + SecMI` naive fusion
- `Recon + CLiD` fusion
- `structural memorization` under current faithful approximation
- `Noise as a Probe` defended-extension on current `SD1.5` contract
- immediate latent-diffusion same-surface board for `Noise as a Probe`
- white-box defense breadth on the current candidate set

---
