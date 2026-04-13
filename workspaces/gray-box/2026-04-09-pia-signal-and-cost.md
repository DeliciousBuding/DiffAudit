# 2026-04-09 Gray-Box Note: PIA Attack Signal And Cost

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `attack signal and quality/cost wording clarified for the admitted gray-box mainline`
- `evidence_level`: `runtime-mainline`

## A. What Signal PIA Uses In This Repo

Current `Research` integration uses the official `external/PIA/DDPM/components.py` `PIA` attacker.

The actual signal is not a generic reconstruction score. It is the discrepancy between:

1. the initial `eps` predicted directly from `x0` at `t=0`
2. the `eps_back` predicted again after the same image is re-noised to each attack step

In the upstream attacker, this is implemented as:

- cache one `eps = eps_getter(x0, ..., t=0)`
- replay that same `eps` across all `attack_num` steps
- at each step, rebuild `x_t = get_xt(x0, step, eps)`
- query the model again to get `eps_back`
- accumulate `|eps - eps_back|^lp`

So the gray-box score in our mainline is really an `epsilon-trajectory consistency` signal.

In `Research/src/diffaudit/attacks/pia_adapter.py`, we then:

- average the attacker output over the attack-step dimension
- negate the distance
- treat higher final scores as more member-like

This means the current member/non-member separation comes from one concrete hypothesis:

- member samples produce more self-consistent `epsilon` predictions under the current DDPM and split
- non-member samples produce larger stepwise `eps -> eps_back` drift

## B. Why Stochastic-Dropout Can Lower PIA

Current defense toggle is:

- `--stochastic-dropout-defense`

In the adapter this changes the model mode from `eval()` to `train()` before scoring, so dropout remains active during the attack.

That matters because the current PIA signal depends on stable `epsilon` predictions across repeated stepwise queries. If dropout is active:

- the first `eps` estimate becomes noisier
- later `eps_back` estimates also become noisier
- the attack's own consistency assumption weakens

So the present `G-1` story is not "privacy proven". The defensible claim is narrower:

- `stochastic-dropout` injects inference-time instability into the exact `epsilon-trajectory consistency` signal used by this PIA path
- on the current CIFAR-10 + DDPM line, that instability is enough to reduce the admitted attack metrics

## C. Current Cost Reading

All admitted gray-box runs below share the same attack shape:

- attacker: `PIA`
- `attack_num = 30`
- `interval = 10`
- `batch_size = 8`
- execution: `single GPU, serial`

Observed elapsed time:

| rung | baseline seconds | defense seconds |
| --- | --- | --- |
| `GPU128` | `39.446178` | `44.112771` |
| `GPU256` | `77.248308` | `92.490869` |
| `GPU512` | `171.214752` | `131.89636` |
| `GPU512 rerun1` | `61.80132` | `69.352488` |

Interpretation:

- cost grows with sample count as expected, but not linearly enough to treat one run as a universal wall-clock estimator
- repeat cost is materially lower than the first `GPU512` pass for both baseline and defense, so the current table should keep both first-run and rerun timing instead of pretending there is one canonical elapsed number
- the only robust cost claim today is:
  - `PIA` mainline is a single-GPU serial attack with `30` stepwise `epsilon` consistency checks per sample path

## D. Allowed Mainline Claim

After this note, the gray-box mainline can be stated as:

- attack: `PIA` on CIFAR-10 DDPM measures stepwise `epsilon` consistency under the official DDPM attacker path
- defense: `stochastic-dropout` weakens that consistency signal at inference time
- evidence: favorable defense movement is supported at `128 / 256 / 512` and once more by a same-scale `GPU512` repeat
- limit: this is still `provisional G-1`, not a validated privacy win and not `paper-aligned`

## E. Immediate Next Step

1. Carry this signal wording into gray-box status pages and the unified table.
2. Keep current gray-box GPU line frozen unless a genuinely new research question appears.
3. Do not reopen `SecMI` until real `flagfile + checkpoint root` exist.
