# 2026-04-15 Gray-Box Note: PIA Scale Stability vs SecMI Full-Split Strength

## Scope

This note consolidates the latest local gray-box runtime evidence after adding the new `PIA 1024 / 1024` pair and the already-complete `SecMI` full-split run.

## Evidence Rungs

| Line | Run | Split Size | AUC | ASR | Notes |
|------|-----|------------|-----|-----|------|
| PIA baseline | `pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive` | `512 / 512` | `0.841339` | `0.786133` | Current promoted runtime-mainline baseline |
| PIA baseline | `pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive` | `1024 / 1024` | `0.838630` | `0.782715` | Same recipe, larger local rung |
| PIA defended | `pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive` | `512 / 512` | `0.828075` | `0.767578` | Existing defended reference |
| PIA defended | `pia-cifar10-runtime-mainline-dropout-defense-20260415-gpu-1024-allsteps-adaptive` | `1024 / 1024` | `0.825966` | `0.770508` | Same-scale comparator for new baseline |
| SecMI stat | `secmi-cifar10-gpu-full-stat-20260415-r2` | `25000 / 25000` | `0.885833` | `0.815400` | Full local split execution |
| SecMI NNS | `secmi-cifar10-gpu-full-stat-20260415-r2` | `25000 / 25000` | `0.946286` | `0.879275` | Auxiliary head on same run |

## Main Takeaways

- `PIA` is now scale-stable across the local runtime ladder: going from `512` to `1024` does not collapse the attack signal.
- `all_steps` stochastic dropout still only mildly degrades `PIA` ranking at larger scale. It weakens the attack, but does not neutralize it.
- `SecMI` remains the stronger raw scorer on the current CIFAR-10 asset stack, especially on the `NNS` head.

## Recommended Competition Wording

- Foreground `PIA` as the more controlled local gray-box runtime mainline, because it now has both baseline and defended ladders at `512` and `1024`.
- Use `SecMI` as independent corroboration that the gray-box privacy signal is not an artifact of the PIA objective alone.
- Do not claim paper-faithful reproduction for either line. Keep the boundary as `workspace-verified / local runtime evidence`.

## Next Step

- If we spend more gray-box budget, prefer consumer-facing consolidation or a materially different defense over yet another same-family `PIA` rung.
