# DiffAudit Research — Challenger Queue

> **Last refreshed**: 2026-04-17
> **Purpose**: Keep the innovation funnel aligned with current repo truth after closure-round reviews
> **Rule**: This queue is for future candidate generation, bounded follow-up, or asset-triggered reopen only. It is not a substitute for admitted/mainline status.

---

## Current Queue Truth

- `active GPU question = none`
- gray-box currently has no immediate next-family execution lane
- white-box currently has no immediate next-hypothesis execution lane
- the next live CPU-first priority should therefore move to:
  - `black-box next-family candidate-generation refresh review`

This queue should now be read with three distinctions:

1. `ready-for-selection`
   - honest CPU-first candidate-generation or review work
2. `hold / not-requestable`
   - real line exists, but current repo truth says do not schedule execution
3. `needs-assets`
   - may still be valuable, but cannot honestly progress without new data/models/contracts

---

## Top 3 Priorities

### 1. `BB-CH-1` Black-box next-family candidate-generation refresh

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐⭐
- `mode`: `CPU-only review`
- `why now`:
  - black-box still needs a second truly different family
  - current box-local lane is stable enough that candidate-generation is higher value than maintenance churn
  - this is the best immediate place to reopen method diversity without spending GPU first

### 2. `GB-CH-1` Second gray-box defense mechanism selection

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐⭐
- `mode`: `CPU-first design / selection`
- `why not immediate`:
  - gray-box mainline is already stable
  - immediate latent-diffusion follow-ups just closed negatively
  - only a materially different new mechanism should reopen this box

### 3. `WB-CH-1` Distinct white-box defended-family import / selection

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐
- `mode`: `CPU-only candidate generation`
- `why not immediate`:
  - `DP-LoRA` is already bounded and frozen below new GPU release
  - `GSA2` is same-family corroboration only
  - `Finding NeMo` is still `not-requestable`

---

## Current Candidates

### Black-box

#### `BB-CH-1` Caption/semantic-family refresh

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐⭐
- `scope`:
  - caption-space
  - feature-space
  - other genuinely new semantic black-box families
- `note`:
  - do not assume old CLIP/BLIP blocker wording is still authoritative without a fresh repo-grounded check

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

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐⭐
- `note`:
  - only reopen for a materially different mechanism
  - do not spend budget on more `TMIA-DM` or `Noise as a Probe` churn under current contracts

#### `GB-CH-2` Ranking-sensitive variable search

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐
- `note`:
  - keep below release until a concrete disagreement/ranking hypothesis is written

#### `GB-CH-3` Noise as a Probe latent-diffusion promotion path

- `status`: `hold`
- `expected value`: ⭐
- `reason`:
  - promotion-gap review already closed negatively
  - contract-shift review also closed negatively
  - do not reopen unless a real contract shift appears

### White-box

#### `WB-CH-1` Distinct defended-family import / selection

- `status`: `ready-for-selection`
- `expected value`: ⭐⭐
- `note`:
  - this is candidate generation only
  - not `DP-LoRA` reruns
  - not `GSA2` scale-up
  - not `Finding NeMo` reopen under current contract

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

## Recommended Next Order

1. `BB-CH-1` black-box next-family candidate-generation refresh review
2. `GB-CH-1` second gray-box defense mechanism selection
3. `WB-CH-1` distinct white-box defended-family import / selection

This order is deliberately CPU-first and does **not** authorize any new GPU run by itself.
