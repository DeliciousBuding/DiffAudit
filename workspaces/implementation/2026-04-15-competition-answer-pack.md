# 2026-04-15 Competition Answer Pack

## 1. Core Claim

DiffAudit shows that diffusion-model membership risk is observable under three attacker models:

- black-box: externally observable generation-only behavior already leaks signal;
- gray-box: limited model-side access keeps that signal strong even after simple lightweight defenses;
- white-box: privileged access makes membership inference close to trivial.

## 2. The Cleanest Story To Tell

### Black-box

- Main line: `Recon DDIM public-100 step30`, `AUC 0.849`.
- Best single-metric rung: `Recon DDIM public-50 step10`, `AUC 0.866`.
- Independent corroboration: `CLiD clip` now runs locally on the same CelebA asset family.
- Cross-check: `CLiD clip` stays perfectly separated on both `celeba_target` and `celeba_partial_target` local `100 / 100` rungs.

How to say it:
`Recon` is the admitted public-subset headline.
`CLiD` is the independent corroboration line proving the black-box signal is not tied to one scoring mechanism.

### Gray-box

- Main controlled line: `PIA`.
- Scale stability:
  - `512 / 512`: `AUC 0.841339`
  - `1024 / 1024`: `AUC 0.83863`
- Independent corroboration:
  - `SecMI stat`: `AUC 0.885833`
  - `SecMI NNS`: `AUC 0.946286`

How to say it:
`PIA` is the more controlled local runtime ladder.
`SecMI` is the stronger alternate scorer that proves the gray-box signal is not an artifact of one attack objective.

### White-box

- Main line: `GSA 1k-3shadow`, `AUC 0.998192`.

How to say it:
Once privileged access is available, membership signal becomes near-trivial. This is the upper-bound risk line.

## 3. Defense Claim

The strongest current lightweight gray-box defense is stochastic dropout at inference time.

- `PIA 512 / 512`: `0.841339 -> 0.828075`
- `PIA 1024 / 1024`: `0.83863 -> 0.825966`

How to say it:
The defense helps, but only mildly. It weakens the ranking signal and shifts thresholds, yet does not neutralize leakage.

For the strongest visual attack-vs-defense contrast:

- white-box `GSA`: `0.998192`
- defended white-box `DPDM W-1 strong-v3`: `0.488783`

## 4. What Is Actually Novel Here

- We are not presenting a single lucky attack result.
- We now have converging local evidence across:
  - two black-box mechanisms;
  - two gray-box mechanisms;
  - a white-box upper bound;
  - repeated same-scale defense comparisons.
- We also have a negative result that matters:
  sample-level `Recon + CLiD` late fusion is feasible on aligned local assets, but it does not improve over already-saturated `CLiD`.

How to say it:
The value is not “we reproduced one paper.”
The value is “we built an audit stack that shows the risk from multiple access assumptions and tested where simple mitigation stops helping.”

## 5. Boundaries We Must State Clearly

- Local `CLiD` runs are `workspace-verified local corroboration`, not paper-faithful full CLiD benchmarks.
- `PIA` and `SecMI` are strong local runtime evidence, but still bounded by checkpoint/source provenance and protocol interpretation.
- `Recon` admitted evidence is still a controlled public-subset black-box line, not generalized internet exploitability.
- `GSA` should be framed as upper-bound privileged-risk evidence, not a normal product KPI.

## 6. 3-Minute Demo Flow

1. Start with the threat-model ladder:
   black-box -> gray-box -> white-box.
2. Show the admitted black-box headline:
   admitted `Recon AUC 0.849`, then mention the stronger single-metric rung `0.866`.
3. Immediately reinforce it with:
   local `CLiD` corroboration across two target-family checkpoints.
4. Move to gray-box:
   `PIA` stable from `512` to `1024`.
5. Add `SecMI`:
   strongest alternate gray-box scorer.
6. Show defense:
   dropout helps but does not solve.
7. End on white-box:
   `GSA` near-saturated, therefore privileged access is catastrophic.

## 7. Likely Judge Questions

### Q1. Are these just reproductions of prior papers?

Answer:
No. Some lines originate from published attacks, but the value here is the integrated audit evidence:
multiple access levels, multiple attack objectives, same-asset corroboration, scale-up checks, and defense comparisons in one local stack.

### Q2. Why should we trust the black-box result if one method could be brittle?

Answer:
Because we do not rely on one method alone.
`Recon` is the admitted main line, and `CLiD` independently lights up on the same asset family across two target-family checkpoints.

### Q3. Did the defense work?

Answer:
Partially.
Stochastic dropout consistently weakens `PIA`, but the attack remains materially above random at both `512` and `1024` scale.

### Q4. Which result is strongest?

Answer:
For raw attack strength, white-box `GSA`.
For gray-box corroboration, `SecMI NNS`.
For admitted black-box mainline, `Recon`.

### Q5. What is the most honest single takeaway?

Answer:
Membership risk in diffusion systems is not confined to one attack setting, and lightweight mitigations are currently insufficient to erase it.

## 8. Slide Titles We Can Reuse

- `DiffAudit: Membership Risk Across Access Levels`
- `Black-Box Leakage Is Observable And Multi-Mechanism`
- `Gray-Box Signal Survives Scale-Up`
- `Lightweight Defenses Help, But Do Not Solve`
- `Privileged Access Makes Leakage Near-Trivial`

## 9. If We Only Get One Closing Sentence

DiffAudit’s strongest message is that diffusion-model privacy risk persists across attacker capabilities, survives method changes and local scale-up, and currently outpaces simple lightweight defenses.
